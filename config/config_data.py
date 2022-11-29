from config.config_dirs import MODEL
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras import datasets
import tensorflow as tf

# DATASET_LOADER = lambda: (extract_training_samples('letters'), extract_test_samples('letters'))
DATASET_LOADER = lambda: datasets.mnist.load_data()
USE_RGB = True
EXPECTED_LABEL = 5
NUM_INPUTS = 250
INPUT_MAXLEN = 2000
VOCAB_SIZE = 10000
classifier = tf.keras.models.load_model(MODEL)