
from itertools import product
import random
from heatmap.mnist.heatmap import generate_heatmaps_by_classifier
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import pandas as pd

from steps.process_heatmaps import APPROACHES, get_perplexity
from config.config_data import EXPECTED_LABEL, USE_RGB
from config.config_heatmaps import DIMENSIONALITY_REDUCTION_TECHNIQUES, RETRAIN_EXPLAINERS
from steps.train_model import create_model
from utils import global_values
from utils.clusters.Approach import OriginalMode
from utils.clusters.compare import compare_approaches_by_data, compare_approaches_by_index
from utils.clusters.postprocessing import select_data_by_cluster
from utils.dataset import get_train_test_data
from config.config_heatmaps import APPROACH, DIMENSIONALITY_REDUCTION_TECHNIQUES
from utils import global_values
from keras.utils.np_utils import to_categorical


def retrain_by_heatmap():
    mnist_loader = lambda: mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(
        dataset_loader=mnist_loader,
        rgb=USE_RGB,
        verbose=True
    )

    predictions_cat = to_categorical(train_labels, 10)

    mask_label = np.array(train_labels == global_values.EXPECTED_LABEL)
    train_data_digit = train_data[mask_label]

    train_indices_digit = []
    for idx in range(0, len(train_labels)):
        if train_labels[idx] == EXPECTED_LABEL:
            train_indices_digit.append(idx)

    # all test set
    accuracies = []
    # only 5s
    accuracies2 = []

    n = 6
    k = 100  
                     
    selected_train_index = random.sample(train_indices_digit, k)
    selected_train_labels = []
    selected_train_data  = []

    remaining_train_data = []
    remaining_train_labels = []
    remaining_train_predictions_cat = []

    for i in range(0, len(train_labels)):
        if i in selected_train_index:
            selected_train_labels.append(train_labels[i])
            selected_train_data.append(train_data[i])
        else:
            if train_labels[i] == EXPECTED_LABEL:
                remaining_train_labels.append(train_labels[i])
                remaining_train_data.append(train_data[i])
                remaining_train_predictions_cat.append(predictions_cat[i])
            else:
                selected_train_labels.append(train_labels[i])
                selected_train_data.append(train_data[i])
            
    acc1, acc2 = create_model(np.array(selected_train_data), np.array(selected_train_labels), np.array(test_data), np.array(test_labels), f"orig_model_HM")

    for j in range(1, n):  
        if j != n - 1:
            heatmaps_df = generate_clusters_by_heatmap(np.array(remaining_train_data), np.array(remaining_train_predictions_cat))
            new_train_data, new_train_labels, remaining_train_data, remaining_train_labels = select_data_by_cluster(heatmaps_df, remaining_train_data, remaining_train_data, k)
        else:
            new_train_data = remaining_train_data
            new_train_labels = remaining_train_labels

        selected_train_data = np.concatenate((new_train_data, selected_train_data), axis=0)  

        selected_train_labels = np.concatenate((new_train_labels, selected_train_labels), axis=0)  

        acc3, acc4 = create_model(np.array(selected_train_data), np.array(selected_train_labels), test_data, test_labels, f"retrained_model_HM__{j}")

        delta_acc = acc3 - acc1
        accuracies.append(delta_acc)

        delta_acc2 = acc4 - acc2
        accuracies2.append(delta_acc2)

        

    np.save("out/retrain_by_heatmap_{RETRAIN_EXPLAINERS[0]}_all", np.array(accuracies))
    np.save("out/retrain_by_heatmap_{RETRAIN_EXPLAINERS[0]}", np.array(accuracies2))

def generate_clusters_by_heatmap(train_data, train_labels):
        # Collect the approaches to use
    print('Collecting the approaches ...')
    # Select the approach from the configurations
    approach = APPROACHES[APPROACH]
    # Select the dimensionality reduction techniques based on the approach
    dimensionality_reduction_techniques = [[]] if approach is OriginalMode else DIMENSIONALITY_REDUCTION_TECHNIQUES
    # If the processing mode is the original one, or there are no best logs -> try all the combinations

    # Collect the approaches
    approaches = [
        approach(
            explainer=explainer(global_values.classifier),
            dimensionality_reduction_techniques=dimensionality_reduction_technique
        )
        for explainer, dimensionality_reduction_technique
        in product(RETRAIN_EXPLAINERS, dimensionality_reduction_techniques)
    ]
    # Collect the data for the approaches
    print('Collecting the data for the approaches ...')
    df_raw = compare_approaches_by_data(
        approaches,
        1,
        train_data,
        train_labels,
        get_info=lambda app: f"perplexity: {get_perplexity(app)}" if get_perplexity(app) != np.nan else "Original Mode"
    )
    return df_raw


def retrain_by_heatmap_provided_explanations():
    mnist_loader = lambda: mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(
        dataset_loader=mnist_loader,
        rgb=USE_RGB,
        verbose=True
    )
    test_data = tf.image.rgb_to_grayscale(test_data).numpy()

    with open("logs/mnist/hm_acc_100.txt", "w") as f:
        # repeat the experiment 5 times
        for rep in range(1, 6):
            f.write(f"Run {rep}: \n")

            classifier = tf.keras.models.load_model(f"out/models/orig_model_{rep}.h5")

            generate_heatmaps_by_classifier(classifier)

            remaining_train_data = np.load(f"logs/mnist/retrain_data/remaining_data_{rep}.npy")
            remaining_train_labels = np.load(f"logs/mnist/retrain_data/remaining_labels_{rep}.npy")
            remaining_train_indices = np.load(f"logs/mnist/retrain_data/remaining_indices_{rep}.npy")

            selected_train_data = np.load(f"logs/mnist/retrain_data/selected_data_{rep}.npy")
            selected_train_labels = np.load(f"logs/mnist/retrain_data/selected_labels_{rep}.npy") 
            
            n = 5
        
            k = 100   
            
            for part in range(1, n+1):  
                new_train_data = []
                new_train_labels = []
                # using HM select k samples from remaining train data
                heatmaps_df = generate_clusters_by_heatmap_indices(remaining_train_indices, classifier)
                new_train_data, new_train_labels, remaining_train_data, remaining_train_labels, remaining_train_indices = select_data_by_cluster(heatmaps_df, remaining_train_data, remaining_train_labels, k)
                selected_train_data = np.concatenate((new_train_data, selected_train_data), axis=0)  
                selected_train_labels = np.concatenate((new_train_labels, selected_train_labels), axis=0) 
            

                acc1, acc2 = create_model(np.array(selected_train_data), np.array(selected_train_labels), test_data, test_labels, f"retrained_hm_{part}")
                
                f.write(f"retrained_hm_{part}: {acc1} , {acc2} \n")


def generate_clusters_by_heatmap_indices(train_indices, classifier):
        # Collect the approaches to use
    print('Collecting the approaches ...')
    # Select the approach from the configurations
    approach = APPROACHES[APPROACH]
    # Select the dimensionality reduction techniques based on the approach
    dimensionality_reduction_techniques = [[]] if approach is OriginalMode else DIMENSIONALITY_REDUCTION_TECHNIQUES
    # If the processing mode is the original one, or there are no best logs -> try all the combinations

    # Collect the approaches
    approaches = [
        approach(
            explainer=explainer(classifier),
            dimensionality_reduction_techniques=dimensionality_reduction_technique
        )
        for explainer, dimensionality_reduction_technique
        in product(RETRAIN_EXPLAINERS, dimensionality_reduction_techniques)
    ]
    # Collect the data for the approaches
    print('Collecting the data for the approaches ...')
    df_raw = compare_approaches_by_index(
        approaches,
        1,
        train_indices,
        get_info=lambda app: f"perplexity: {get_perplexity(app)}" if get_perplexity(app) != np.nan else "Original Mode"
    )
    return df_raw



if __name__ == "__main__":
    retrain_by_heatmap_provided_explanations()