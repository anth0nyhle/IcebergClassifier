#!/usr/bin/env python

import random
import numpy as np
import json
import csv
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold

try:
    to_unicode = unicode
except NameError:
    to_unicode = str


if __name__ == "__main__":
    with open('train.json') as data_file:
        data_loaded = json.load(data_file)

    third_channel = np.zeros((75, 75))

    images = []
    labels = []
    inc_angles = []
    for j in data_loaded:
        first_channel = np.array(j["band_1"]).reshape(third_channel.shape)
        second_channel = np.array(j["band_2"]).reshape(third_channel.shape)
        image = np.array([first_channel, second_channel, third_channel])
        image = np.rollaxis(image, 0, 3)
        images.append(image)
        labels.append(j["is_iceberg"])
        inc_angles.append(j["inc_angle"])

    with open("test.json") as test_file:
        test_loaded = json.load(test_file)

    third_channel_test = np.zeros((75, 75))
    images_test = []
    inc_angles_test = []
    ids = []
    for k in test_loaded:
        first_channel_test = np.array(k["band_1"]).reshape(third_channel_test.shape)
        second_channel_test = np.array(k["band_2"]).reshape(third_channel_test.shape)
        image_test = np.array([first_channel_test, second_channel_test, third_channel_test])
        image_test = np.rollaxis(image_test, 0, 3)
        images_test.append(image_test)
        ids.append(k["id"])
        inc_angles_test.append(k["inc_angle"])

    images = np.array(images)
    labels = np.array(labels)
    inc_angles = np.array(inc_angles)

    images_test = np.array(images_test)
    inc_angles_test = np.array(inc_angles_test)

    cvscores = []

    # datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest')
    # datagen.fit(images)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(75, 75, 3)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.25))

    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    csv_logger = callbacks.CSVLogger("test12_8lay_epoch_results.log", separator=",", append=False)
    model.fit([images, inc_angles], labels, validation_split=.10, batch_size=32, epochs=50, verbose=1, callbacks=[csv_logger])
    # model.fit_generator(datagen.flow(images, labels, batch_size=32), steps_per_epoch=len(images) / 32, epochs=20, verbose=1, callbacks=[csv_logger], shuffle=True)

    scores = model.evaluate([images, inc_angles], labels, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    labels_test = model.predict([images_test, inc_angles_test], batch_size=32, verbose=1)

    np.savetxt("test12_8lay_default_predlabels.csv", labels_test, delimiter=",")
    np.savetxt("test12_8lay_default_dev_acc.csv", cvscores, delimiter=",")

    with open("test12_submission.csv", "w") as submission_file:
        wr = csv.writer(submission_file, delimiter=",")
        wr.writerow(["id", "is_iceberg"])
        for i, p in zip(ids, labels_test):
            wr.writerow((i, p[0]))