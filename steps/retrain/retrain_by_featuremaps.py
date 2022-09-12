import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


from config.config_data import EXPECTED_LABEL, USE_RGB

from steps.train_model import create_model
import feature_map.mnist.feature_map as feature_map_generator
from utils.clusters.postprocessing import select_data_by_cluster
from utils.featuremaps.postprocessing import process_featuremaps_data
from utils.dataset import get_train_test_data


def retrain_by_featuremap():
    mnist_loader = lambda: mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(
        dataset_loader=mnist_loader,
        rgb=USE_RGB,
        verbose=True
    )
    test_data = tf.image.rgb_to_grayscale(test_data).numpy()

    with open("logs/mnist/fm_acc.txt", "w") as f:
            # repeat the experiment 5 times
            for rep in range(1, 6):
                f.write(f"Run {rep}: \n")
                remaining_train_data = np.load(f"logs/mnist/retrain_data/remaining_data_{rep}.npy")
                remaining_train_labels = np.load(f"logs/mnist/retrain_data/remaining_labels_{rep}.npy")

                remaining_train_data = tf.image.rgb_to_grayscale(remaining_train_data).numpy()

                selected_train_data = np.load(f"logs/mnist/retrain_data/selected_data_{rep}.npy")
                selected_train_labels = np.load(f"logs/mnist/retrain_data/selected_labels_{rep}.npy") 

                selected_train_data =  tf.image.rgb_to_grayscale(selected_train_data).numpy()
                
                n = 5
            
                k = int(len(remaining_train_data)/n)   
                
                for part in range(1, n+1):  
                    new_train_data = []
                    new_train_labels = []
                    # using FM select k samples from remaining train data
                    if part != n:
                        featuremaps_df, samples = feature_map_generator.generate_featuremap_by_data(remaining_train_data, remaining_train_labels)
                        df = process_featuremaps_data(featuremaps_df, samples)
                        new_train_data, new_train_labels, remaining_train_data, remaining_train_labels, _ = select_data_by_cluster(df, remaining_train_data, remaining_train_labels, k)
                        selected_train_data = np.concatenate((new_train_data, selected_train_data), axis=0)  
                        selected_train_labels = np.concatenate((new_train_labels, selected_train_labels), axis=0)  
                    # last batch of data
                    else:
                        selected_train_data = tf.image.rgb_to_grayscale(train_data).numpy()
                        selected_train_labels = train_labels

                    acc1, acc2 = create_model(np.array(selected_train_data), np.array(selected_train_labels), test_data, test_labels, f"retrained_fm_{part}")
                    
                    f.write(f"retrained_fm_{part}: {acc1} , {acc2} \n")


if __name__ == "__main__":
    retrain_by_featuremap()