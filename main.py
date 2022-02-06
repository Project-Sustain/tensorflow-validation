#!/bin/python3
import numpy
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

from sklearn.preprocessing import normalize, MinMaxScaler

# MongoDB Stuff
URI = "mongodb://lattice-100:27018/"
DATABASE = "sustaindb"
COLLECTION = "noaa_nam"

# Modeling Stuff
LEARNING_RATE = 0.001
EPOCHS = 3
BATCH_SIZE = 32


def main():
    print("tensorflow version: {}".format(tf.__version__))

    m = 2

    features = ['PRESSURE_AT_SURFACE_PASCAL', 'RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT']
    label = 'TEMPERATURE_AT_SURFACE_KELVIN'
    projection = {
        "_id": 0,
    }
    for feature in features:
        projection[feature] = 1
    projection[label] = 1

    pprint(projection)

    client = MongoClient(URI)
    database = client[DATABASE]
    collection = database[COLLECTION]
    documents = collection.find({'COUNTY_GISJOIN': 'G2000010'}, projection)

    features_and_labels_list = list(map(lambda x: list(x.values()), documents))
    features_and_labels_numpy = np.array(features_and_labels_list)
    features_and_labels_numpy_transposed = features_and_labels_numpy.T
    features_numpy = features_and_labels_numpy_transposed[:m].T
    labels_numpy = features_and_labels_numpy_transposed[m:].T

    print(f"features_numpy: {features_numpy}, min={np.min(features_numpy, axis=1)}, max={np.max(features_numpy, axis=1)}")
    print(f"labels_numpy: {labels_numpy}, min={np.min(labels_numpy, axis=1)}, max={np.max(labels_numpy, axis=1)}")

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(features_numpy, labels_numpy)
    print(scaler.data_min_)



    normalized_features = tf.keras.utils.normalize(
        features_numpy, axis=-1, order=2
    ).transpose()

    normalized_labels = tf.keras.utils.normalize(
        labels_numpy, axis=-1, order=2
    ).transpose()

    pprint(normalized_labels)
    pprint(normalized_features)

    print(f"normalized_features shape: {normalized_features.shape}")
    print(f"normalized_labels shape: {normalized_labels.shape}")

    print(f"normalized_features: max={np.max(normalized_features)}, min={np.min(normalized_features)}")
    print(f"normalized_labels: max={np.max(normalized_labels)}, min={np.min(normalized_labels)}")

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(m,)))
    model.add(tf.keras.layers.Dense(units=1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    model.summary()

    history = model.fit(normalized_features, normalized_labels, epochs=EPOCHS, validation_split=0.2)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    pprint(hist)

    results = model.evaluate(normalized_features, normalized_labels, batch_size=128)
    print("test loss, test acc:", results)

    # Save model
    model.save('saved_model/my_model')

    # Reload model
    new_model = tf.keras.models.load_model('saved_model/my_model')

    # Check its architecture
    new_model.summary()

    new_results = new_model.evaluate(normalized_features, normalized_labels, batch_size=128)
    print("RELOADED test loss, test acc:", new_results)

    # first = np.array(np_features[:1])
    #
    # with np.printoptions(precision=2, suppress=True):
    #     print('First example:', first)
    #     print()
    #     print('Normalized:', normalizer(first).numpy())

    client.close()


if __name__ == '__main__':
    main()
