import logging
import os
from keras.optimizers import SGD, Adam
import sys
import numpy as np
#from moviepy.editor import *
logging.basicConfig(level=logging.DEBUG)

from lib.SSRNet import SSR_net
from lib.utilsInput import *
from lib.utilsOutput import *
from lib.utilsTrain import *


def test_mae(model, testNPZ):
    logging.debug("Loading data...")
    image, age, image_size = load_data_npz(testNPZ)
    
    x_data = image
    y_data_a = age
    
    y = model.predict(x_data)
    mae = 0
    for i in range(len(x_data)):
      mae = mae + abs(y[i] - y_data_a[i])
    mae = mae/len(x_data)
    
    return mae

def main():
    
    test_path = './data/megaage_test.npz'
    
    logging.debug("Loading data...")
    image, age, image_size = load_data_npz(test_path)
    
    x_data = image
    y_data_a = age
    optMethod = Adam()
    
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    
    model = SSR_net(image_size,stage_num, lambda_local, lambda_d)()
    model.compile(optimizer=optMethod, loss=["mae"], metrics={'pred_a':'mae'})
    
    for fname in os.listdir('./weights/'):
        if fname.endswith('.hdf5'):
            model.load_weights('./weights/' + str(fname))
            break
    else:
        print('No existing weights found.')
        
    model.summary()
    
    mae = test_mae(model, test_path)
    print(mae)
    
if __name__ == '__main__':
    main()