import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from config.config_data import EXPECTED_LABEL, USE_RGB
from utils import global_values
from utils.dataset import get_train_test_data
from steps.train_model import create_model

def train_initial():
    mnist_loader = lambda: mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(
        dataset_loader=mnist_loader,
        rgb=USE_RGB,
        verbose=True
    )
    train_data, test_data = (
        tf.image.rgb_to_grayscale(train_data).numpy(),
        tf.image.rgb_to_grayscale(test_data).numpy()
    )
    mask_label = np.array(train_labels == global_values.EXPECTED_LABEL)
    train_data_digit = train_data[mask_label]
    train_label_digit = train_labels[mask_label]
    
    with open("logs/mnist/orig_acc.txt", "w") as f:
        for ind in range(0, 1):
            k = 100
            # randomly select k samples from train_data of expected label only                
            selected_train_index = random.sample(range(0, len(train_data_digit)), k)
            selected_train_labels = []
            selected_train_data  = []

            remaining_train_data = []
            remaining_train_labels = []
            remaining_train_indices = []

            for i in range(0, len(train_data_digit)): 
                # keep the selected indices with expected label 
                if i in selected_train_index:
                    selected_train_data.append(train_data_digit[i])
                    selected_train_labels.append(train_label_digit[i])
                else:
                    remaining_train_indices.append(i)
                    remaining_train_data.append(train_data_digit[i])
                    remaining_train_labels.append(train_label_digit[i])

            # keep the rest of training set
            for i in range(0, len(train_labels)):
                if train_labels[i] != EXPECTED_LABEL:
                    selected_train_labels.append(train_labels[i])
                    selected_train_data.append(train_data[i])

            np.save(f"logs/mnist/retrain_data/selected_data_{ind}", selected_train_data)
            np.save(f"logs/mnist/retrain_data/selected_labels_{ind}", selected_train_labels)

            np.save(f"logs/mnist/retrain_data/remaining_data_{ind}", remaining_train_data)
            np.save(f"logs/mnist/retrain_data/remaining_labels_{ind}", remaining_train_labels)
            np.save(f"logs/mnist/retrain_data/remaining_indices_{ind}", remaining_train_indices)

                    
            acc1, acc2 = create_model(np.array(selected_train_data), np.array(selected_train_labels), np.array(test_data), np.array(test_labels), f"orig_model_{ind}")
            
            f.write(f"orig_model_{ind}: {acc1} , {acc2} \n")


if __name__ == "__main__":
    train_initial()