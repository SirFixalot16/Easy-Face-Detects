import tensorflow
import keras
from keras import backend as K
from keras.utils import get_custom_objects
import numpy as np

class DecayLearningRate(keras.callbacks.Callback):
	def __init__(self, startEpoch):
		self.startEpoch = startEpoch

	def on_train_begin(self, logs={}):
		return
	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):

		if epoch in self.startEpoch:
			if epoch == 0:
				ratio = 1
			else:
				ratio = 0.1
			LR = K.get_value(self.model.optimizer.lr)
			K.set_value(self.model.optimizer.lr,LR*ratio)
		return

	def on_epoch_end(self, epoch, logs={}):
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return

def get_initial_weights(output_size):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights


def register_keras_custom_object(cls):
    """ A decorator to register custom layers, loss functions etc in global scope """
    get_custom_objects()[cls.__name__] = cls
    return cls