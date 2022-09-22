
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from config.config_data import USE_RGB
from steps.train_model import create_model
from utils.dataset import get_train_test_data


def retrain_by_random():
    mnist_loader = lambda: mnist.load_data()
    (_, _), (test_data, test_labels) = get_train_test_data(
        dataset_loader=mnist_loader,
        rgb=USE_RGB,
        verbose=True
    )
    test_data = tf.image.rgb_to_grayscale(test_data).numpy()
    accs1 = []
    accs2 = []    
    with open("logs/mnist/random_acc.txt", "w") as f:
        # repeat the experiment n times
        part = 0
        for rep in range(0, 30):
            f.write(f"Run {rep}: \n")
            remaining_train_data = np.load(f"logs/mnist/retrain_data/remaining_data_{part}.npy")
            remaining_train_labels = np.load(f"logs/mnist/retrain_data/remaining_labels_{part}.npy")
            remaining_train_indices = list(range(0, len(remaining_train_data)))

            selected_train_data = np.load(f"logs/mnist/retrain_data/selected_data_{part}.npy")
            selected_train_labels = np.load(f"logs/mnist/retrain_data/selected_labels_{part}.npy") 
            
            k = 100  
            
            new_train_data = []
            new_train_labels = []
            # randomly select k samples from remaining train data

            new_train_indices = random.sample(remaining_train_indices, k)
            for index in new_train_indices:
                new_train_data.append(remaining_train_data[index])
                new_train_labels.append(remaining_train_labels[index])
                remaining_train_indices.remove(index)

            selected_train_data = np.concatenate((new_train_data, selected_train_data), axis=0)  
            selected_train_labels = np.concatenate((new_train_labels, selected_train_labels), axis=0)  

            acc1, acc2 = create_model(np.array(selected_train_data), np.array(selected_train_labels), test_data, test_labels, f"retrained_random_{rep}")
                
            f.write(f"retrained_random_{rep}: {acc1} , {acc2} \n")
            accs1.append(acc1)
            accs2.append(acc2)
    
    np.save("logs/mnist/random_acc_all", accs1)
    np.save("logs/mnist/random_acc_5", accs2)


if __name__ == "__main__":
    retrain_by_random()