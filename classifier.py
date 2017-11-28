#!/usr/bin/env python

import random
import numpy as np
import json

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Stackoverflow to deal with python versions
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str
    
    #from keras.models import Sequential

# Tensorflow ordering of input data: samples, height, width, channel

if __name__ == "__main__":
    with open('train.json') as data_file:
        data_loaded = json.load(data_file)

    
    third_channel = np.zeros((75,75))
    images = []
    labels = []
    for j in data_loaded:
        first_channel = np.array(j["band_1"]).reshape(third_channel.shape)
        second_channel = np.array(j["band_2"]).reshape(third_channel.shape)
        image = np.array([first_channel, second_channel, third_channel])
        image = np.rollaxis(image,0,3)
        images.append(image)
        labels.append(j["is_iceberg"])

    images = np.array(images)
    labels = np.array(labels)

    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(75,75,3)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(images, labels, batch_size=32, epochs=1000)

