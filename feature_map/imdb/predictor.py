import tensorflow as tf
from tensorflow import keras
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from config.config_data import EXPECTED_LABEL
from config.config_dirs import MODEL
from config.config_featuremaps import VOCAB_SIZE


class Predictor:

    # Load the pre-trained model. ,custom_objects={'KerasLayer':hub.KerasLayer}
    model = tf.keras.models.load_model(MODEL)
    print("Loaded model from disk")

    # loading
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    @staticmethod
    def predict(text):
        # #Predictions vector
        seq = Predictor.tokenizer.texts_to_sequences([text])

        padded_texts = pad_sequences(seq, maxlen=VOCAB_SIZE)

        explabel = (np.expand_dims(EXPECTED_LABEL, 0))

        # Convert class vectors to binary class matrices
        explabel = keras.utils.to_categorical(explabel, 2)
        explabel = np.argmax(explabel.squeeze())

        #Predictions vector
        predictions = Predictor.model.predict(padded_texts)

        prediction1, prediction2 = np.argsort(-predictions[0])[:2]

        # Activation level corresponding to the expected class
        confidence_expclass = predictions[0][explabel]

        if prediction1 != EXPECTED_LABEL:
            confidence_notclass = predictions[0][prediction1]
        else:
            confidence_notclass = predictions[0][prediction2]

        confidence = confidence_expclass - confidence_notclass

        return prediction1, confidence