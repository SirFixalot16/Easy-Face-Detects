import logging
import os
from keras.optimizers import SGD, Adam
import sys
import numpy as np
#from moviepy.editor import *
logging.basicConfig(level=logging.DEBUG)
import random
import pandas as pd
from PIL import Image

from lib.SSRNet import SSR_net
from lib.utilsInput import *
from lib.utilsOutput import *
from lib.utilsTrain import *


def test_acc(model, path):
    anno = pd.read_csv(
        ('./data/afadanno.csv'), 
        sep = ",", 
        header = None, 
        names=['name', 'age', 'gender'])
    
    anno2018 = anno.loc[:, ['name', 'gender']]
    
    annoR = np.asarray(anno2018.values)

    for i in range(len(annoR)):
       if (annoR[i][1] == 'm'):
           annoR[i][1] = 1
       elif (annoR[i][1] == 'f'): annoR[i][1] = 0

    random.shuffle(annoR)
    passer = annoR[:160000]
    
    # Load images
    i = 0
    f = i + 40000
    passs = passer[i:f]
    
    x_data = np.empty((40000, 64, 64, 3))
    y_data_g = np.empty((40000), dtype=int)
    
    for i in range(40000):
        image = Image.open(path+str(passs[i][0]))
        x_data[i] = np.asarray(image)
        y_data_g[i] = (passs[i][1])
        print("pass: ", i)
        
    y = model.predict(x_data)
    acc = 0
    for i in range(40000):
        if(int(y[i]) == int(y_data_g[i])):
            acc = acc + 1
    acc = acc/40000
    return acc
    

def main():
    
    ipath = './data/AFADF/AFADFR/content/data/AFADF/'
    
    logging.debug("Loading data...")

    optMethod = Adam()
    
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    
    model = SSR_net(64 ,stage_num, lambda_local, lambda_d)()
    model.compile(optimizer=optMethod, loss=["mae"], metrics={'pred_a':'mae'})
    
    for fname in os.listdir('./weights_sex/'):
        if fname.endswith('.hdf5'):
            model.load_weights('./weights_sex/' + str(fname))
            break
    else:
        print('No existing weights found.')
        
    model.summary()
    
    acc = test_acc(model, ipath)
    print(acc)
    
if __name__ == '__main__':
    main()