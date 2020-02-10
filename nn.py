from augmentation import *

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tensorflow as tf


def resize(img):
    return tf.image.resize_images(img, [66, 200])


def nn_model():
    model = Sequential()  # linear stack of layers

    model.add(Lambda(lambda imgs: imgs[:, 80:, :, :], input_shape=(160, 320, 3)))

    model.add(Lambda(lambda imgs: (imgs/255.0) - 0.5))
    model.add(Lambda(resize))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1164, activation='relu'))
    model.add(BatchNormalization())  # normalise output from activation function

    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())  # Makes sure weights don't become imbalanced with high or low values

    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())  # Can increase speed at with training occurs.

    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())  # Reduce ability of outlying large weights that over influence training process

    model.add(Dense(1))

    model.compile(loss= "MSE", optimizer=Adam(lr=0.001))
    return model


m = nn_model()
#  m.fit_generator()

#  Activation function finds a value for weighted sum of inputs. eg: Sigmoid finds value between 0 - 1
#  Epoch is the number of pass through same data
#  Learning rate is multiplied with difference ......
#  Optimiser is ADAM, a variation of SGD
#  Shuffle = True shuffles order of data that's passed each epoch.
#  Verbose is how much output we want to see
#  Batch size is the number of data passed through to the network at one time.

#  Loss is calculated at the end of each epoch, and tries to reduce it

