from itertools import product
import random
import numpy as np
from steps.process_heatmaps import APPROACHES, get_perplexity
from steps.retrain.retrain_by_heatmaps import generate_clusters_by_heatmap_indices
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from IPython.display import display
import pandas as pd


from config.config_data import EXPECTED_LABEL, USE_RGB
from config.config_heatmaps import DIMENSIONALITY_REDUCTION_TECHNIQUES, RETRAIN_EXPLAINERS
from steps.train_model import create_model
from utils import global_values
from utils.clusters.Approach import Approach, OriginalMode
from utils.clusters.compare import compare_approaches_by_data, compare_approaches_by_index
from utils.clusters.postprocessing import select_data_by_cluster, select_data_by_cluster_both
from utils.dataset import get_train_test_data
from config.config_heatmaps import APPROACH, DIMENSIONALITY_REDUCTION_TECHNIQUES
from utils.featuremaps.postprocessing import process_featuremaps_data
import feature_map.mnist.feature_map as feature_map_generator

def retrain_by_both():
    mnist_loader = lambda: mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(
        dataset_loader=mnist_loader,
        rgb=USE_RGB,
        verbose=True
    )
    test_data = tf.image.rgb_to_grayscale(test_data).numpy()
    accs1 = []
    accs2 = []
    with open("logs/mnist/both_acc.txt", "w") as f:
        # repeat the experiment n times
        part = 0
        for rep in range(0, 30):
            f.write(f"Run {rep}: \n")
            classifier = tf.keras.models.load_model(f"out/models/orig_model_{part}.h5")
            remaining_train_data = np.load(f"logs/mnist/retrain_data/remaining_data_{part}.npy")
            remaining_train_labels = np.load(f"logs/mnist/retrain_data/remaining_labels_{part}.npy")
            remaining_train_indices = np.load(f"logs/mnist/retrain_data/remaining_indices_{part}.npy")

            selected_train_data = np.load(f"logs/mnist/retrain_data/selected_data_{part}.npy")
            selected_train_labels = np.load(f"logs/mnist/retrain_data/selected_labels_{part}.npy") 

        
            k = 100   
            
            new_train_data = []
            new_train_labels = []
            # using HM select k samples from remaining train data
            heatmaps_df = generate_clusters_by_heatmap_indices(remaining_train_indices, classifier)       
            featuremaps_df, samples, features = feature_map_generator.generate_featuremap_by_data(remaining_train_data, remaining_train_labels)
            df = process_featuremaps_data(featuremaps_df, samples, features)

            new_train_data, new_train_labels, remaining_train_data, remaining_train_labels, remaining_train_indices = select_data_by_cluster_both(heatmaps_df, df, remaining_train_data, remaining_train_labels, k)
            selected_train_data = np.concatenate((new_train_data, selected_train_data), axis=0)  
            selected_train_labels = np.concatenate((new_train_labels, selected_train_labels), axis=0)  

        
            acc1, acc2 = create_model(np.array(selected_train_data), np.array(selected_train_labels), test_data, test_labels, f"retrained_both_{rep}")
            
            f.write(f"retrained_both_{rep}: {acc1} , {acc2} \n")
            accs1.append(acc1)
            accs2.append(acc2)
    
    np.save("logs/mnist/both_acc_all", accs1)
    np.save("logs/mnist/both_acc_5", accs2)

if __name__ == "__main__":
    retrain_by_both()