import time
import cv2
import numpy as np
#import torch
from keras import backend as K

from lib.FSANet import *
from keras.layers import Average


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def get_detector(choix):

    if (choix == 2):
        from mtcnn import MTCNN
        detector = MTCNN(
            #keep_all=True, device=device
            )
    elif (choix == 1):
        detector = cv2.dnn.readNetFromCaffe(
            './lib/deploy.prototxt',
            './lib/res10_300x300_ssd_iter_140000.caffemodel'
        )
        
    return detector
    
def mtcnn_cb(detector, func, img: np.ndarray) -> np.ndarray:
    #boxes, probs = detector.detect(img)
    boxes = detector.detect_faces(img)
    boxes = boxes[0]
    box = boxes.get('box')
    
    if (func == 'crop'):
        imgc = img[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])]
        return imgc
    elif (func == 'box'):
        imgb = cv2.rectangle(img, (box[0], box[1]), 
                        (box[0] + box[2], box[1] + box[3]), 
                        (255, 0, 0), 2)
        return imgb
    else: return img

def ssd_cb(detector, func, img: np.ndarray) -> np.ndarray:
    import pandas as pd
    size_og = img.shape
    img_cp = img.copy()
    img_cp = cv2.resize(img_cp, (300, 300))
    arx = size_og[1]/300
    ary = size_og[0]/300
    
    imgBlob = cv2.dnn.blobFromImage(image=img_cp)
    detector.setInput(imgBlob)
    detections = detector.forward()
    
    colabls = ["img_id", "is_face", "confidence", 
               "left", "top", "right", "bottom"]
    
    detections_df = pd.DataFrame(detections[0][0], columns = colabls)
    detections_df = detections_df[detections_df['is_face'] == 1]
    detections_df = detections_df[detections_df['confidence'] >= 0.90]
    detections_df['left'] = (detections_df['left'] * 300).astype(int)
    detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
    detections_df['right'] = (detections_df['right'] * 300).astype(int)
    detections_df['top'] = (detections_df['top'] * 300).astype(int)
    
    for i in detections_df.iterrows():
        left = i[1]['left']
        right = i[1]['right']
        bottom = i[1]['bottom']
        top = i[1]['top']

        imgb = cv2.rectangle(img, (int(left*arx), int(top*ary)), 
                      (int(right*arx), int(bottom*ary)), 
                      (255, 255, 255), 1)
        imgc = img[int(top*ary):int(bottom*ary),
                   int(left*arx):int(right*arx)]
    
    if (func == 'crop'): return imgc
    elif (func == 'box'): return imgb
    else: return detections

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=80):
    import math
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll)
                 * math.sin(pitch) * math.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch)
                 * math.sin(yaw) * math.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def draw_results_mtcnn(detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network, time_plot):

    if len(detected) > 0:
        for i, d in enumerate(detected):
            #x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            if d['confidence'] > 0.95:
                x1, y1, w, h = d['box']

                x2 = x1+w
                y2 = y1+h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)

                faces[i, :, :, :] = cv2.resize(
                    input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i, :, :, :] = cv2.normalize(
                    faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                face = np.expand_dims(faces[i, :, :, :], axis=0)
                p_result = model.predict(face)

                face = face.squeeze()
                img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :],
                                p_result[0][0], p_result[0][1], p_result[0][2])

                input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img

    #cv2.imshow("result", input_img)
    
    return input_img  # ,time_network,time_plot

def draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot):
    
    # loop over the detections
    if detected.shape[2]>0:
        for i in range(0, detected.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detected[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                (h0, w0) = input_img.shape[:2]
                box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                (startX, startY, endX, endY) = box.astype("int")
                # print((startX, startY, endX, endY))
                x1 = startX
                y1 = startY
                w = endX - startX
                h = endY - startY
                
                x2 = x1+w
                y2 = y1+h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)
                
                faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)        
                
                face = np.expand_dims(faces[i,:,:,:], axis=0)
                p_result = model.predict(face)
                
                face = face.squeeze()
                img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], p_result[0][0], p_result[0][1], p_result[0][2])
                
                input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img
                
    #cv2.imshow("result", input_img)
    
    return input_img #,time_network,time_plot

def cam(choix, func):
    print("Appareil Photo Vidéo")
    
    detector = get_detector(choix=choix)
    fps = 30
    start = time.time()
    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600*1)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900*1)
    
    if not vidcap.isOpened():
        raise ValueError(f"Failed to open cam")
    while True:
        temppasse = time.time() - start
        ret, frame = vidcap.read()
        
        if not ret:
            print("error in retrieving frame")
            break
        
        if temppasse > 1./fps:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (choix == 2): img = mtcnn_cb(detector=detector, func=func, img=img)
            elif (choix == 1): img = ssd_cb(detector=detector, func=func, img=img)
            cv2.imshow('cam', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            
        if cv2.waitKey(1) == ord('q'):
            break

def camml(choix):
    print("Appareil Photo Vidéo")
    
    # Models & weights
    n_primcaps = 7*3
    S_set = [3, 16, 2, n_primcaps, 5]

    model1 = FSA_net_Capsule(64, 3, [3,3,3], 1, S_set)()
    model2 = FSA_net_Var_Capsule(64, 3, [3,3,3], 1, S_set)()
    
    n_primcaps = 8*8*3
    S_set = [3, 16, 2, n_primcaps, 5]
    model3 = FSA_net_noS_Capsule(64, 3, [3,3,3], 1, S_set)()
    
    # Loading models
    print('Loading models ...')
    weight_file1 = './pretrained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')
    weight_file2 = './pretrained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')
    weight_file3 = './pretrained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')
    
    # Average model
    inputs = Input(shape=(64,64,3))
    x1 = model1(inputs) #1x1
    x2 = model2(inputs) #var
    x3 = model3(inputs) #w/o
    avg_model = Average()([x1,x2,x3])
    model = Model(inputs=inputs, outputs=avg_model)
    
    # Get detector
    detector = get_detector(choix=choix)
    
    # Paramètres du temp
    img_idx = 0
    detected = '' #make this not local variable
    time_detection = 0; time_network = 0; time_plot = 0; ad = 0.6
    skip_frame = 1 # every 5 frame do 1 detection and network forward propagation
    
    # Paramètres de la caméra
    fps = 30
    start = time.time()
    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600*1)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900*1)
    
    print('Start detecting pose ...')
    if (choix == 2): detected_pre = []
    elif (choix == 1): detected_pre = np.empty((1,1,1))
    
    if not vidcap.isOpened():
        raise ValueError(f"Failed to open cam")
    while True:
        temppasse = time.time() - start
        ret, frame = vidcap.read()
        
        if not ret:
            print("error in retrieving frame")
            break
        
        if temppasse > 1./fps:
            # Index
            img_idx +=1
            img_h, img_w, _ = np.shape(frame)
            if (choix == 2):
                if img_idx == 1 or img_idx % skip_frame == 0:
                    time_detection = 0; time_network = 0; time_plot = 0
                    
                    detected = detector.detect_faces(frame)

                    if len(detected_pre) > 0 and len(detected) == 0:
                        detected = detected_pre

                    faces = np.empty((len(detected), 64, 64, 3))

                    input_img = draw_results_mtcnn(
                        detected, frame, faces, ad, 64, 
                        img_w, img_h, model, time_detection, 
                        time_network, time_plot
                    )
                else:
                    input_img = draw_results_mtcnn(
                    detected, frame, faces, ad, 
                    64, img_w, img_h, model, 
                    time_detection, time_network, time_plot
                    )

                if len(detected) > len(detected_pre) or img_idx % (skip_frame*3) == 0:
                    detected_pre = detected
            elif (choix == 1):
                if img_idx==1 or img_idx%skip_frame == 0:
                    time_detection = 0; time_network = 0; time_plot = 0
            
                    blob = cv2.dnn.blobFromImage(
                        cv2.resize(
                        frame, (300, 300)),1.0, 
                        (300, 300), (104.0, 177.0, 123.0)
                        )
                    detector.setInput(blob)
                    detected = detector.forward()

                    if detected_pre.shape[2] > 0 and detected.shape[2] == 0:
                        detected = detected_pre

                    faces = np.empty((detected.shape[2], 64, 64, 3))
                    input_img = draw_results_ssd(
                        detected, frame, faces, ad, 
                        64, img_w, img_h, model, 
                        time_detection, time_network, time_plot
                    )
            
                else:
                    input_img = draw_results_ssd(
                        detected, frame, faces, ad, 
                        64, img_w, img_h, model, 
                        time_detection, time_network, time_plot
                        )
                if detected.shape[2] > detected_pre.shape[2] or img_idx%(skip_frame*3) == 0:
                    detected_pre = detected
                
            cv2.imshow('cam', 
                       input_img
                       #cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                       )

        if cv2.waitKey(1) == ord('q'):
            break
        
# User input
import argparse
def get_args():
    parser = argparse.ArgumentParser(description="This script opens cam input for face detection functions.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--detector", type=int, default=1, 
        help="1. for SSD (faster), 2. for MTCNN (more accurate)"
        )
    parser.add_argument(
        '--function', default='box',
        help= '(box)es for faces detected, or (crop)ping faces',
        )
    parser.add_argument(
        '--purpose', type=int, default=1,
        help="1. for detect only, 2. for face angles"
    )
    args = parser.parse_args()
    return args



def main():
    print("Appareil Photo Vidéo")
    #img = cv2.imread('./input/360795..jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cam(2, 'box')
    #camml(1)
    
    args = get_args()
    
    detr = args.detector
    func = args.function
    purr = args.purpose
    
    if (purr == 1): cam(detr, func)
    elif (purr == 2): 
        K.clear_session()
        K.set_learning_phase(0) # make sure its testing mode
        camml(detr)
    else: print('Suck my balls!')

if __name__ == '__main__':
    main()