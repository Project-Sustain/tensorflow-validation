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
COLLECTION = "noaa_nam"

# Modeling Stuff
LEARNING_RATE = 0.01
EPOCHS = 20
BATCH_SIZE = 32


def using_tfio_dataset():
    # Fetch collection from MongoDB as training dataset
    dataset = tfio.experimental.mongodb.MongoDBIODataset(
        uri=URI, database=DATABASE, collection=COLLECTION
    )

    tensor_specs = {
        "feature": tf.TensorSpec(tf.TensorShape([]), tf.float32, name="TEMPERATURE_AT_SURFACE_KELVIN"),  # feature
        "label": tf.TensorSpec(tf.TensorShape([]), tf.float32, name="TEMPERATURE_TROPOPAUSE_KELVIN")  # label
    }
    pprint(tensor_specs)

    dataset = dataset.map(
        lambda x: tfio.experimental.serialization.decode_json(x, specs=tensor_specs)
    )

    # Prepare a tuple of (features, label)
    dataset = dataset.map(lambda v: (v, v.pop("label")))
    dataset = dataset.batch(BATCH_SIZE)
    pprint(dataset)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,), name="feature"))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    model.summary()

    history = model.fit(dataset, epochs=EPOCHS)

    res = model.evaluate(dataset)
    print("test loss, test acc:", res)
    # plt.plot(history.history['loss'])
    # plt.show()


def main():
    print("tensorflow version: {}".format(tf.__version__))
    print("tensorflow-io version: {}".format(tfio.__version__))

    client = MongoClient(URI)
    database = client[DATABASE]
    collection = database[COLLECTION]

    feature = 'TEMPERATURE_AT_SURFACE_KELVIN'
    label = 'TEMPERATURE_TROPOPAUSE_KELVIN'

    pprint(collection.find_one({'GISJOIN': 'G4802970'}, {'_id': 0, feature: 1, label: 1}))
    client.close()


if __name__ == '__main__':
    main()
