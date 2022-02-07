#!/bin/python3
import numpy
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas
import time
from pprint import pprint
#from sklearn.model_selection import train_test_split
#from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_io as tfio
from pymongo import MongoClient
from sklearn.preprocessing import normalize, MinMaxScaler

import validation

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

    validation.validate_model(
        "saved_model"
        "my_model",
        "Linear Regression",
        documents,
        features,
        label,
        "RMSE",
        True
    )

    # # load into Pandas DF
    # dataframe = pandas.DataFrame(list(documents))
    # pprint(dataframe)
    #
    # scaler = MinMaxScaler(feature_range=(0, 1)).fit(dataframe)
    # scaled = scaler.transform(dataframe)
    # scaled_df = pandas.DataFrame(scaled, columns=dataframe.columns)
    # pprint(scaled_df)
    # client.close()
    #
    # features_df = scaled_df[features]
    # label_df = scaled_df.pop(label)
    #
    # model = tf.keras.Sequential()
    # model.add(tf.keras.Input(shape=(m,)))
    # model.add(tf.keras.layers.Dense(units=1, activation='relu'))
    # model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    # model.summary()
    #
    # history = model.fit(features_df, label_df, epochs=EPOCHS, validation_split=0.2)
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch
    # pprint(hist)
    #
    # results = model.evaluate(features_df, label_df, batch_size=128)
    # print("test loss, test acc:", results)
    #
    # # Save model
    # model.save('saved_model/my_model')
    #
    # # Reload model
    # new_model = tf.keras.models.load_model('saved_model/my_model')
    #
    # # Check its architecture
    # new_model.summary()
    #
    # new_results = new_model.evaluate(features_df, label_df, batch_size=128)
    # print("RELOADED test loss, test acc:", new_results)


if __name__ == '__main__':
    main()
