import pickle
from config.config_data import INPUT_MAXLEN, VOCAB_SIZE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import imdb
from utils.dataset import get_train_test_data_imdb



def create_model(x_train, y_train, test_data, test_labels_cat, name):

    embedding_dims = 50  # Embedding size for each token
    filters = 250
    kernel_size = 3
    hidden_dims = 250

    inputs = Input(shape=(INPUT_MAXLEN,), dtype=tf.int32)
    embedded_sequences = Embedding(VOCAB_SIZE,
                                   embedding_dims)(inputs)
    out = Conv1D(filters, 
                 kernel_size, 
                 padding='valid', 
                 activation='relu', 
                 strides=1)(embedded_sequences)
    out = Dropout(0.4)(out)
    out = GlobalMaxPooling1D()(out)
    out = Dense(hidden_dims, 
                activation='relu')(out)
    out = Dropout(0.4)(out)
    outputs = Dense(2, activation='softmax')(out)
        
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

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
    model.save(f"in/models/{name}.h5")

if __name__ == '__main__':
    # Get the train test data and convert the labels to categorical
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data_imdb(
        verbose=True
    )
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train_data)

    import pickle

    # saving
    with open('in/models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    x_train = tokenizer.texts_to_sequences(train_data)
    x_test = tokenizer.texts_to_sequences(test_data)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=INPUT_MAXLEN)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=INPUT_MAXLEN)

    create_model(x_train, to_categorical(train_labels), x_test, to_categorical(test_labels), f"text_classifier")

    # accuracy:    0.8801599740982056, 0.30222010612487793