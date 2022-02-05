import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import pymongo
from pprint import pprint
from logging import info, error


def validate_model(job_id, model_type, documents, feature_fields, label_field, validation_metric, normalize=True):
    features = []
    labels = []

    m = len(feature_fields)

    # Takes documents, a MongoDB Cursor() for N Document objects, containing m=2 features and 1 label in the form:
    # {
    #   'PRESSURE_AT_SURFACE_PASCAL': 98053.425,                                    // feature_1
    #   'RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT': 63.84692993164063,      // feature_2
    #   'TEMPERATURE_AT_SURFACE_KELVIN': 296.07669921875                            // label
    #  }
    # ... and maps the documents to lists in the form:
    # [
    #   [98053.425, 63.84692993164063, 296.07669921875],
    #   [feature_1_1, feature_2_1, label_1],
    #   ...
    #   [feature_1_M, feature_2_M, label_N]
    # ]
    features_and_labels_list = list(map(lambda x: list(x.values()), documents))

    # Convert this list of lists into a Numpy array in the following form:
    # array([[9.80534250e+04, 2.96076699e+02, 6.38469299e+01],
    #        [9.80246250e+04, 2.95956699e+02, 6.41469299e+01],
    #        [9.79838250e+04, 2.95736699e+02, 6.52469299e+01],
    #        ...,
    #        [9.80545813e+04, 2.84013379e+02, 9.23746105e+01],
    #        [9.80313813e+04, 2.83813379e+02, 9.23746105e+01],
    #        [9.81185813e+04, 2.83713379e+02, 9.21746105e+01]])
    # ... with a shape of (N, 3)
    features_and_labels_numpy = np.array(features_and_labels_list)

    # Transpose this 2D array into the following form:
    # array([[9.80534250e+04, 9.80246250e+04, 9.79838250e+04, ..., // feature_1
    #         9.80545813e+04, 9.80313813e+04, 9.81185813e+04],
    #        [2.96076699e+02, 2.95956699e+02, 2.95736699e+02, ..., // feature_2
    #         2.84013379e+02, 2.83813379e+02, 2.83713379e+02],
    #        [6.38469299e+01, 6.41469299e+01, 6.52469299e+01, ..., // label
    #         9.23746105e+01, 9.23746105e+01, 9.21746105e+01]])
    features_and_labels_numpy_transposed = features_and_labels_numpy.T

    # Then slice the label array from the feature arrays:
    # array([[98053.425, 98024.625, 97983.825, ...,          // feature_1
    #         98054.58125, 98031.38125, 98118.58125],
    #        [296.07669922, 295.95669922, 295.73669922, ..., // feature_2
    #         284.01337891, 283.81337891, 283.71337891]])
    # ...and label:
    # array([[63.84692993, 64.14692993, 65.24692993, ..., 92.37461052, // label
    #         92.37461052, 92.17461052]])
    features_numpy = features_and_labels_numpy_transposed[:m]
    labels_numpy = features_and_labels_numpy_transposed[m:]






    num_processed = 0
    for document in documents:
        features.append(document[feature_fields[0]])
        labels.append(document[label_field])
        num_processed += 1

        if num_processed % 1000 == 0:
            info(f"Processed {num_processed} documents...")

    np_features = np.array(features)
    np_labels = np.array(labels)

    info(f"np_features shape: {np_features.shape}")
    info(f"np_labels shape: {np_labels.shape}")

    normalized_features = tf.keras.utils.normalize(
        np_features, axis=-1, order=2
    ).transpose()

    normalized_labels = tf.keras.utils.normalize(
        np_labels, axis=-1, order=2
    ).transpose()

    pprint(normalized_features)
    info(f"normalized_features shape: {normalized_features.shape}")
    info(f"normalized_labels shape: {normalized_labels.shape}")

    # Reload model
    new_model = tf.keras.models.load_model(f"{path}/{job_id}")

    # Check its architecture
    new_model.summary()

    new_results = new_model.evaluate(normalized_features, normalized_labels, batch_size=128)
    info(f"Test loss, test acc: {new_results}")
