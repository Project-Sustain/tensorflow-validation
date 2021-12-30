#!/bin/python3

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from pprint import pprint
#from sklearn.model_selection import train_test_split
#from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_io as tfio
from pymongo import MongoClient

URI = "mongodb://lattice-100:27018/"
DATABASE = "sustaindb"
COLLECTION = "mpb_cypress_hill_sk_100m"


def main():

    print("tensorflow version: {}".format(tf.__version__))
    print("tensorflow-io version: {}".format(tfio.__version__))

    # Fetch collection from MongoDB as training dataset
    dataset = tfio.experimental.mongodb.MongoDBIODataset(
        uri=URI, database=DATABASE, collection=COLLECTION
    )

    dataset

    # learning_rate = 0.01
    # epochs = 200
    # n_samples = 30
    # train_x = np.linspace(0, 20, n_samples)
    # train_y = 3 * train_x + np.random.randn(n_samples)
    #
    # plt.plot(train_x, train_y, 'o')
    # plt.plot(train_x, 3 * train_x)
    # plt.show()
    #
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(1, input_shape=[1]))
    # model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate))
    # model.summary()
    #
    # history = model.fit(train_x, train_y, epochs=300)
    # plt.plot(history.history['loss'])
    # plt.show()


if __name__ == '__main__':
    main()
