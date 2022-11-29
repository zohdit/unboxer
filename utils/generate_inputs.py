
import json
import os
import pickle
import random
import warnings


import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from config.config_data import DATASET_LOADER, EXPECTED_LABEL, USE_RGB, INPUT_MAXLEN
from config.config_dirs import IMDB_INPUTS, MNIST_INPUTS
from utils.dataset import get_train_test_data_mnist, get_train_test_data_imdb
from config.config_featuremaps import CASE_STUDY
from config.config_data import classifier


# Ignore warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def save_inputs():
    if CASE_STUDY == "MNIST":
        # Get the train and test data and labels
        (train_data, train_labels), (test_data, test_labels) = get_train_test_data_mnist(
            dataset_loader=DATASET_LOADER,
            rgb=USE_RGB,
            verbose=False
        )

        _, test_data_gs = (
            tf.image.rgb_to_grayscale(train_data).numpy(),
            tf.image.rgb_to_grayscale(test_data).numpy()
        )

        # Get the predictions
        try:
            predictions = classifier.predict(test_data).argmax(axis=-1)
        except ValueError:
            # The model expects grayscale images
            predictions = classifier.predict(test_data_gs).argmax(axis=-1)


        generated_data = []

        # read generated inputs
        for subdir, dirs, files in os.walk(MNIST_INPUTS, followlinks=False):
            # Consider only the files that match the pattern
            for sample_file in [os.path.join(subdir, f) for f in files if f.endswith(".npy")]:
                generated_data.append(np.load(sample_file))

        generated_data = np.array(generated_data)

        generated_labels = np.full(generated_data.shape[0], EXPECTED_LABEL)


        generated_data_gs = generated_data.squeeze()
        generated_data_gs = tf.expand_dims(generated_data_gs, -1)
        # Get the predictions
        try:
            generated_predictions = classifier.predict(generated_data).argmax(axis=-1)
        except ValueError:
            # The model expects grayscale images
            generated_predictions = classifier.predict(generated_data_gs).argmax(axis=-1)

        generated_data = tf.image.grayscale_to_rgb(tf.expand_dims(generated_data.squeeze(), -1)).numpy()

        

        # Get the mask for the data
        mask_miss_mnist = np.array(test_labels != predictions)

        generated_mask_miss = np.array(generated_labels != generated_predictions)

        test_data = test_data[mask_miss_mnist]
        test_labels = test_labels[mask_miss_mnist]
        predictions = predictions[mask_miss_mnist]

        generated_data = generated_data[generated_mask_miss]
        generated_labels = generated_labels[generated_mask_miss]
        generated_predictions = generated_predictions[generated_mask_miss]


        generated_data = np.concatenate((test_data, generated_data))
        generated_labels = np.concatenate((test_labels, generated_labels))
        generated_predictions = np.concatenate((predictions, generated_predictions))

        np.save(f"in/data/mnist/input/x_test.npy", generated_data)
        np.save(f"in/data/mnist/input/y_test.npy", generated_labels)
        np.save(f"in/data/mnist/input/pred_test.npy", generated_predictions)

    if CASE_STUDY == "IMDB":
        # Get the train and test data and labels
        (train_data, train_labels), (test_data, test_labels) = get_train_test_data_imdb(
            verbose=False
        )
        
        # loading
        with open('in/models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)


        x_train = tokenizer.texts_to_sequences(train_data)
        x_test = tokenizer.texts_to_sequences(test_data)

        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=INPUT_MAXLEN)
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=INPUT_MAXLEN)


        DATASET_DIR = "in/data"

        if not os.path.exists(f"{DATASET_DIR}/imdb-cached"):
            os.makedirs(f"{DATASET_DIR}/imdb-cached")
            np.save(f"{DATASET_DIR}/imdb-cached/x_train.npy", x_train)
            np.save(f"{DATASET_DIR}/imdb-cached/y_train.npy", train_labels)
            np.save(f"{DATASET_DIR}/imdb-cached/x_test.npy", x_test)
            np.save(f"{DATASET_DIR}/imdb-cached/y_test.npy", test_labels)
            
            predictions = classifier.predict(x_test).argmax(axis=-1)
            np.save(f"{DATASET_DIR}/imdb-cached/y_prediction.npy", predictions)

        test_labels = np.load(f"in/data/imdb-cached/y_test.npy")
        predictions = np.load("in/data/imdb-cached/y_prediction.npy")
        predictions_cat = to_categorical(predictions, 2)

        generated_data = []

        # read generated inputs
        for subdir, dirs, files in os.walk(IMDB_INPUTS, followlinks=False):
            # Consider only the files that match the pattern
            for sample_file in [os.path.join(subdir, f) for f in files if f.endswith(".json")]:
                    with open(sample_file, "r") as json_file:           
                        input_data = json.load(json_file)
                        generated_data.append(input_data["text"])


        generated_data = np.array(generated_data)
        generated_labels = np.full(generated_data.shape[0], EXPECTED_LABEL)
        generated_data_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(generated_data), maxlen=INPUT_MAXLEN)
        generated_predictions = classifier.predict(generated_data_padded).argmax(axis=1)


        # Get the mask for the data
        predictions = np.array(predictions)
        mask_miss = np.array(test_labels != predictions)
        generated_mask_miss = np.array(generated_labels != generated_predictions)

        test_data = np.array(test_data)[mask_miss]
        test_data = test_data.squeeze()
        test_labels = test_labels[mask_miss]
        predictions = predictions[mask_miss]


        generated_data = generated_data[generated_mask_miss]
        generated_labels = generated_labels[generated_mask_miss]
        generated_predictions = generated_predictions[generated_mask_miss]


        generated_data = np.concatenate((test_data, generated_data))
        generated_labels = np.concatenate((test_labels, generated_labels))
        generated_predictions = np.concatenate((predictions, generated_predictions))

        sample_indices = random.sample(range(len(generated_data)), 600)

        np.save(f"in/data/imdb/input/x_test.npy", generated_data[sample_indices])
        np.save(f"in/data/imdb/input/y_test.npy", generated_labels[sample_indices])
        np.save(f"in/data/imdb/input/pred_test.npy", generated_predictions[sample_indices])
            
            
def load_inputs():
    if CASE_STUDY == "IMDB":
        x_train = np.load(f"in/data/imdb-cached/x_train.npy")
        y_train = np.load(f"in/data/imdb-cached/y_train.npy")
        x_test = np.load(f"in/data/imdb/input/x_test.npy")
        y_test = np.load(f"in/data/imdb/input/y_test.npy")
        pred_test = np.load(f"in/data/imdb/input/pred_test.npy")

    if CASE_STUDY == "MNIST":
        (x_train, y_train) , _ = get_train_test_data_mnist(
            dataset_loader=DATASET_LOADER,
            rgb=USE_RGB,
            verbose=False
        )
        x_test = np.load(f"in/data/mnist/input/x_test.npy")
        y_test = np.load(f"in/data/mnist/input/y_test.npy")
        pred_test = np.load(f"in/data/mnist/input/pred_test.npy")

    return x_train, y_train, x_test, y_test, pred_test


if __name__ == '__main__':
    save_inputs()