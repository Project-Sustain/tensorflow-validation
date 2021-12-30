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

# MongoDB Stuff
URI = "mongodb://lattice-100:27018/"
DATABASE = "sustaindb"
COLLECTION = "mpb_cypress_hill_sk_100m"

# Modeling Stuff
LEARNING_RATE = 0.01
EPOCHS = 20


def main():

    print("tensorflow version: {}".format(tf.__version__))
    print("tensorflow-io version: {}".format(tfio.__version__))

    # Fetch collection from MongoDB as training dataset
    dataset = tfio.experimental.mongodb.MongoDBIODataset(
        uri=URI, database=DATABASE, collection=COLLECTION
    )

    # Numeric features.
    numerical_cols = ['T_MAX']

    SPECS = {
        "dense_input": tf.TensorSpec(tf.TensorShape([]), tf.float32, name="T_MAX"),
        "target": tf.TensorSpec(tf.TensorShape([]), tf.float32, name="T_MIN_SUMMER"),
    }

    pprint(SPECS)

    BATCH_SIZE = 32
    train_ds = dataset.map(
        lambda x: tfio.experimental.serialization.decode_json(x, specs=SPECS)
    )

    # Prepare a tuple of (features, label)
    train_ds = train_ds.map(lambda v: (v, v.pop("target")))
    train_ds = train_ds.batch(BATCH_SIZE)

    pprint(train_ds)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=[1]))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    model.summary()

    history = model.fit(train_ds, epochs=EPOCHS, name=numerical_cols)
    # plt.plot(history.history['loss'])
    # plt.show()


if __name__ == '__main__':
    main()
