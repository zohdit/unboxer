
import os
from config.config_featuremaps import INPUT_MAXLEN, VOCAB_SIZE
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from datasets import load_dataset
from utils.global_values import MyTokenAndPositionEmbedding, TransformerBlock
from utils import global_values

SA_ACTIVATION_LAYERS = [5]

NC_ACTIVATION_LAYERS = [
    (1, lambda x: x.token_emb),  # Embedding layers
    (1, lambda x: x.pos_emb),  # Embedding layers
    (2, lambda x: x.ffn[0]),  # Dense feed forward layers in transformer
    (2, lambda x: x.ffn[1]),  # Dense feed forward layers in transformer
    3, 5  # Dense layers in classifier
]

BADGE_SIZE = 128




def create_model(x_train, y_train, test_data, test_labels_cat, name):

    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = tf.keras.layers.Input(shape=(INPUT_MAXLEN,))
    embedding_layer = MyTokenAndPositionEmbedding(INPUT_MAXLEN, VOCAB_SIZE, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    # As opposed to the keras tutorial, we use categorical_crossentropy,
    #   and we run 10 instead of 2 epochs, but with early stopping.
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        x_train, y_train, batch_size=32, epochs=10, validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    loss, acc1 = model.evaluate(test_data, test_labels_cat)
    print(f'Accuracy on whole test for {name}: {acc1}, {loss}')
    # model.save(f"in/models/{name}.h5")

if __name__ == '__main__':
    create_model(global_values.train_data_padded,  tf.keras.utils.to_categorical(global_values.train_labels), global_values.test_data_padded, global_values.test_labels_cat, f"text_classifier")

    # accuracy:  0.8614400029182434, 0.34638524055480957