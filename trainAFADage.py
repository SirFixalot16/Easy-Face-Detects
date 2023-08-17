import pandas as pd
import logging
import random
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from PIL import Image
import numpy as np
from keras.utils import plot_model
#from moviepy.editor import *
logging.basicConfig(level=logging.DEBUG)

from lib.SSRNet import SSR_net
from lib.utilsInput import *
from lib.utilsOutput import *
from lib.utilsTrain import *

def main():
    # Modify for use
    train_path = './data/AFADF/'
    db_name = 'afadage'
    batch_size = 512
    nb_epochs = 20
    validation_split = 0.1 

    logging.debug("Loading data...")
    
    # Load annotations
    anno = pd.read_csv(
        ('./data/afadage.csv'), 
        sep = ",", 
        header = None, 
        names=['name', 'age'])
    afadfr = np.asarray(anno.values)
    random.shuffle(afadfr)
    passer = afadfr[:160000]
    
    # Load images
    i = 0
    f = i + 40000
    passs = passer[i:f]
    
    x_data = np.empty((40000, 64, 64, 3))
    y_data_a = np.empty((40000))
    
    for i in range(40000):
        image = Image.open(train_path+str(passs[i][0]))
        x_data[i] = np.asarray(image)
        y_data_a[i] = passs[i][1]
        print("pass: ", i)
    
    
    start_decay_epoch = [30,60]

    optMethod = Adam()
    
    # I just bullshited this and it worked I ain't gonna lie
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1


    model = SSR_net(64, stage_num, lambda_local, lambda_d)()
    save_name = 'ssrnet_%d_%d_%d_%d_%s_%s' % (stage_num[0],stage_num[1],stage_num[2], 64, lambda_local, lambda_d)
    model.compile(optimizer=optMethod, loss=["mae"], metrics={'pred_a':'mae'})
    
    # Check for existing weights, I'm lazy so leave only the best weight in there
    for fname in os.listdir('./weights/'):
        if fname.endswith('.hdf5'):
            model.load_weights('./weights/' + str(fname))
            break
    else:
        print('No existing weights found. Proceed with fresh training.')
        
    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    plot_model(model, to_file="./model/"+save_name+".png")
    
    # Saving to formats
    model.save("model/"+ save_name+ '.h5')
    model.save("model/"+ save_name+ '.tf')
    with open(os.path.join("./model/", save_name+'.json'), "w") as f:
        f.write(model.to_json())


    decaylearningrate = DecayLearningRate(start_decay_epoch)

    callbacks = [ModelCheckpoint(db_name+"_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"), decaylearningrate
                        ]

    logging.debug("Running training...")



    data_num = len(x_data)
    #indexes = np.arange(data_num)
    #np.random.shuffle(indexes)
    #x_data = x_data[indexes]
    #y_data_a = y_data_a[indexes]
    train_num = int(data_num * (1 - validation_split))

    x_train = x_data[:train_num]
    x_test = x_data[train_num:]
    y_train_a = y_data_a[:train_num]
    y_test_a = y_data_a[train_num:]
    
    # Clear RAM
    x_data = None
    del x_data
    y_data_a = None
    del y_data_a

    hist = model.fit(data_generator_reg(X=x_train, Y=y_train_a, batch_size=batch_size),
                               steps_per_epoch=train_num // batch_size,
                               validation_data=(x_test, [y_test_a]),
                               epochs=nb_epochs, verbose=1,
                               callbacks=callbacks)

    logging.debug("Saving weights...")
    model.save_weights(os.path.join(db_name+"_models/"+save_name, save_name+'.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(db_name+"_models/"+save_name, 'history_'+save_name+'.h5'), "history")


if __name__ == '__main__':
    main()