import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

from utils.dataset import get_train_test_data_mnist
from config.config_data import USE_RGB

def create_model(train_data, train_labels, test_data, test_labels, name):

    mask_label = np.array(test_labels == 5)
    test_data_masked = test_data[mask_label]
    test_labels_masked = test_labels[mask_label]


    print('Creating the classifier ...')
    print("train_data:" + str(len(train_data)))

    # Set up the sequential classifier
    classifier = Sequential(layers=[
        layers.Lambda(
            lambda images: tf.image.rgb_to_grayscale(images)
            if len(images.shape) > 3 and images.shape[-1] > 1
            else images
        ),
        layers.Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1)),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(.5),
        layers.Flatten(),
        layers.Dense(128),
        layers.Activation('relu'),
        layers.Dropout(.5),
        layers.Dense(10),
        layers.Activation('softmax')
    ])
    classifier.compile(
        loss='categorical_crossentropy',
        # optimizer='adadelta',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Get the train test data and convert the labels to categorical

    train_labels_cat, test_labels_cat, test_labels_cat_masked = to_categorical(train_labels, 10), to_categorical(test_labels, 10), to_categorical(test_labels_masked, 10)

    # Train the classifier
    classifier.fit(
        train_data,
        train_labels_cat,
        epochs=12,
        batch_size=128,
        # shuffle=True,
        # verbose=True
    )
    loss, acc1 = classifier.evaluate(test_data, test_labels_cat)
    print(f'Accuracy on whole test for {name}: {acc1}')

    loss, acc2 = classifier.evaluate(test_data_masked, test_labels_cat_masked)
    print(f'Accuracy on 5s for {name}: {acc2}')

    # Save the classifier
    # classifier.save("out/models/"+name+".h5")

    return acc1, acc2


if __name__ == "__main__":

    # Get the train test data and convert the labels to categorical
    mnist_loader = lambda: mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data_mnist(
        dataset_loader=mnist_loader,
        rgb=USE_RGB,
        verbose=True
    )

    train_data_gs, test_data_gs = (
    tf.image.rgb_to_grayscale(train_data).numpy(),
    tf.image.rgb_to_grayscale(test_data).numpy())

    create_model(train_data_gs, train_labels, test_data_gs, test_labels, f"digit_classifier")

    # digit_classifier 99.24, 98.76