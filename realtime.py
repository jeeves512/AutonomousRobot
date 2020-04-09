import cv2
import numpy as np
import json
import tensorflow.keras.models as tf
import os
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split


def create_model():
    model = Sequential()  # linear stack of layers

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu', input_shape=(76, 160, 3)))
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

    model.compile(loss="MSE", optimizer=Adam(lr=0.001))
    return model


def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    return model


cam = cv2.VideoCapture(0)  # captures webcam image
model = load_trained_model('my_model.h5')

while True:
    ret, QueryImgBGR = cam.read()  # captures frame from camera
    QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    img1 = cv2.resize(QueryImgBGR, (76, 160))
    img = np.asarray(img1, dtype=np.float32)
    img = img.reshape(1, 76, 160, 3)
    print(model.predict(img))
    cv2.imshow('result', img1)
    cv2.waitKey(10)
