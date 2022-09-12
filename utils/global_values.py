import os
import pickle
import warnings
from config.config_featuremaps import VOCAB_SIZE

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from config.config_data import DATASET_LOADER, EXPECTED_LABEL, USE_RGB, MISBEHAVIOR_ONLY
from config.config_dirs import MNIST_INPUTS, MODEL
from utils.dataset import get_train_test_data


# datasets_dir=r"/usr/local/lib/python3.7/site-packages/datasets"

# import sys
# sys.path.append(datasets_dir)


from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

# Taken from https://keras.io/examples/nlp/text_classification_with_transformer/
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim), ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = embed_dim, num_heads, ff_dim, rate

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config["embed_dim"] = self.embed_dim
        config["num_heads"] = self.num_heads
        config["ff_dim"] = self.ff_dim
        config["rate"] = self.rate
        config["att"] = self.att.get_config()
        config["ffn"] = self.ffn.get_config()
        config["layernorm1"] = self.layernorm1.get_config()
        config["layernorm2"] = self.layernorm2.get_config()
        config["dropout1"] = self.dropout1.get_config()
        config["dropout2"] = self.dropout2.get_config()
        return config

    @classmethod
    def from_config(cls, config):
        instance = cls(config["embed_dim"], config["num_heads"], config["ff_dim"], config["rate"])
        instance.att = tf.keras.layers.MultiHeadAttention.from_config(config["att"])
        instance.ffn = tf.keras.Sequential.from_config(config["ffn"])
        instance.layernorm1 = tf.keras.layers.LayerNormalization.from_config(config["layernorm1"])
        instance.layernorm2 = tf.keras.layers.LayerNormalization.from_config(config["layernorm2"])
        instance.dropout1 = tf.keras.layers.Dropout.from_config(config["dropout1"])
        instance.dropout2 = tf.keras.layers.Dropout.from_config(config["dropout2"])
        return instance

# Taken from https://keras.io/examples/nlp/text_classification_with_transformer/
@tf.keras.utils.register_keras_serializable()
class MyTokenAndPositionEmbedding(tf.keras.layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        # super(MyTokenAndPositionEmbedding, self).__init__()
        super(MyTokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen, self.vocab_size, self.embed_dim = maxlen, vocab_size, embed_dim

    def build(self, input_shape):
        self.token_emb = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        return {'maxlen': self.maxlen,
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim}


# Ignore warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get the classifier
classifier = tf.keras.models.load_model(MODEL)


# Get the train and test data and labels
(train_data, train_labels), (test_data, test_labels) = get_train_test_data(
    dataset_loader=DATASET_LOADER,
    rgb=USE_RGB,
    verbose=False
)

train_data_gs, test_data_gs = (
    tf.image.rgb_to_grayscale(train_data).numpy(),
    tf.image.rgb_to_grayscale(test_data).numpy()
)

# Get the predictions
try:
    predictions = classifier.predict(test_data).argmax(axis=-1)
except ValueError:
    # The model expects grayscale images
    predictions = classifier.predict(test_data_gs).argmax(axis=-1)
predictions_cat = to_categorical(predictions)

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
generated_predictions_cat = to_categorical(generated_predictions)


generated_data = tf.image.grayscale_to_rgb(tf.expand_dims(generated_data.squeeze(), -1)).numpy()
# Get the mask for the data
mask_miss_mnist = np.array(test_labels != predictions)

generated_mask_miss = np.array(generated_labels != generated_predictions)

# Filter the data if configured
if MISBEHAVIOR_ONLY:
    test_data = test_data[mask_miss_mnist]
    test_data_gs = test_data_gs[mask_miss_mnist]
    test_labels = test_labels[mask_miss_mnist]
    predictions = predictions[mask_miss_mnist]
    predictions_cat = predictions_cat[mask_miss_mnist]

    generated_data = generated_data[generated_mask_miss]
    generated_data_gs = generated_data_gs[generated_mask_miss]
    generated_labels = generated_labels[generated_mask_miss]
    generated_predictions = generated_predictions[generated_mask_miss]
    generated_predictions_cat = generated_predictions_cat[generated_mask_miss]


# classifier = tf.keras.models.load_model(MODEL, custom_objects={'KerasLayer': MyTokenAndPositionEmbedding, 'TransformerBlock': TransformerBlock})

# Get the train and test data and labels

# from datasets import load_dataset
# train_ds = load_dataset('imdb', cache_dir=f"in/data", split='train')
# train_data, train_label = train_ds['text'], train_ds['label']

# test_ds = load_dataset('imdb', cache_dir=f"in/data", split='test')
# test_data, test_labels = test_ds['text'], test_ds['label']

# train_data_padded = np.load(f"in/data/imdb-cached/x_train.npy")
# train_labels = np.load(f"in/data/imdb-cached/y_train.npy")
# test_data_padded = np.load(f"in/data/imdb-cached/x_test.npy")
# test_labels = np.load(f"in/data/imdb-cached/y_test.npy")
# predictions = np.load("in/data/imdb-cached/y_prediction.npy")


# predictions = classifier.predict(test_data_padded).argmax(axis=-1)


# predictions_cat = to_categorical(predictions, 2)

# # Get the mask for the data
# predictions = np.array(predictions)
# mask_miss = np.array(test_labels != predictions)
# predictions_cat = np.array(predictions_cat)
# test_data = np.array(test_data)

# # Filter the data if configured
# if MISBEHAVIOR_ONLY:
#     test_data = test_data[mask_miss]
#     test_data_padded = test_data_padded[mask_miss]
#     test_labels = test_labels[mask_miss]
#     predictions = predictions[mask_miss]
#     predictions_cat = predictions_cat[mask_miss]

