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
from keras.utils import plot_model
from keras import callbacks
from sklearn.model_selection import StratifiedKFold

# Stackoverflow to deal with python versions
import io

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

    # from keras.models import Sequential

# Tensorflow ordering of input data: samples, height, width, channel

if __name__ == "__main__":
    with open('train.json') as data_file:
        data_loaded = json.load(data_file)

    third_channel = np.zeros((75, 75))

    images = []
    labels = []
    for j in data_loaded:
        first_channel = np.array(j["band_1"]).reshape(third_channel.shape)
        second_channel = np.array(j["band_2"]).reshape(third_channel.shape)
        image = np.array([first_channel, second_channel, third_channel])
        image = np.rollaxis(image, 0, 3)
        images.append(image)
        labels.append(j["is_iceberg"])

    with open("test.json") as test_file:
        test_loaded = json.load(test_file)

    third_channel_test = np.zeros((75, 75))
    images_test = []
    ids = []
    for k in test_loaded:
        first_channel_test = np.array(k["band_1"]).reshape(third_channel_test.shape)
        second_channel_test = np.array(k["band_2"]).reshape(third_channel_test.shape)
        image_test = np.array([first_channel_test, second_channel_test, third_channel_test])
        image_test = np.rollaxis(image_test, 0, 3)
        images_test.append(image_test)
        ids.append(k["id"])

    images = np.array(images)
    labels = np.array(labels)

    images_test = np.array(images_test)

    seed = 7
    np.random.seed(seed)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []

    count = 0

    for train, test in kfold.split(images, labels):
        count += 1
        print("Iteration:", count, "--------------------------------------------------------------------------")

        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(75, 75, 3)))

        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        # model.add(Dense(64, activation='softplus'))
        # model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        csv_logger = callbacks.CSVLogger("test2_6lay_10crossval_epoch_results.log", separator=",", append=True)
        model.fit(images[train], labels[train], batch_size=32, epochs=20, verbose=1, callbacks=[csv_logger])

        scores = model.evaluate(images[test], labels[test], verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    labels_test = model.predict(images_test, batch_size=32, verbose=1)
    labels_test = np.array(labels_test)
    labels_test = np.round(labels_test, decimals=1)

    np.savetxt("test2_6lay_10crossval_predlabels.csv", labels_test, delimiter=",")
    np.savetxt("test2_6lay_10crossval_dev_acc.csv", cvscores, delimiter=",")

    with open("test2_submission.csv", "w") as submission_file:
        wr = csv.writer(submission_file, delimiter=",")
        wr.writerow(["id", "is_iceberg"])
        for i, p in zip(ids, labels_test):
            wr.writerow((i, p))

    # submission = []
    #
    # for i, p in zip(ids, labels_test):
    #     submission.append((i, p))
    #
    # np.savetxt("test2_submission.csv", submission, delimiter=",")
