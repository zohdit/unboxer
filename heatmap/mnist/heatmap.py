
from itertools import product
from steps.process_heatmaps import APPROACHES
from tensorflow.keras.datasets import mnist
from config.config_data import EXPECTED_LABEL, USE_RGB
import numpy as np
import tensorflow as tf
from config.config_heatmaps import DIMENSIONALITY_REDUCTION_TECHNIQUES, EXPLAINERS, RETRAIN_EXPLAINERS
from utils import global_values
from tqdm import tqdm, trange
from utils.dataset import get_train_test_data
from keras.utils.np_utils import to_categorical

def generate_heatmaps_by_classifier(classifier):

    mnist_loader = lambda: mnist.load_data()
    (train_data, train_labels), (_, _) = get_train_test_data(
        dataset_loader=mnist_loader,
        rgb=USE_RGB,
        verbose=True
    )

    mask_label = np.array(train_labels == global_values.EXPECTED_LABEL)
    train_data = train_data[mask_label]
    predictions_cat = to_categorical(train_labels, 10)
    predictions_cat = predictions_cat[mask_label]
    train_labels = train_labels[mask_label]

    # original mode
    approach = APPROACHES[0]
    approaches = [
            approach(
                explainer=explainer(classifier),
                dimensionality_reduction_techniques=dimensionality_reduction_technique
            )
            for explainer, dimensionality_reduction_technique
            in product(RETRAIN_EXPLAINERS, DIMENSIONALITY_REDUCTION_TECHNIQUES)
        ]

    for idx, approach in (bar := tqdm(list(enumerate(approaches)))):
        bar.set_description(f'Using the approache ({approach})')
        explainer = approach.get_explainer()
        # Extract some information about the current approach
        contributions = approach.generate_contributions_by_data(train_data, predictions_cat)
        
        print(contributions.shape)
        # save contributions
        np.save(f"logs/mnist/contributions/train_data_only_{EXPECTED_LABEL}_{explainer.__class__.__name__}", contributions)

def generate_heatmaps_train_data():

    mnist_loader = lambda: mnist.load_data()
    (train_data, train_labels), (_, _) = get_train_test_data(
        dataset_loader=mnist_loader,
        rgb=USE_RGB,
        verbose=True
    )

    mask_label = np.array(train_labels == global_values.EXPECTED_LABEL)
    train_data = train_data[mask_label]
    predictions_cat = to_categorical(train_labels, 10)
    predictions_cat = predictions_cat[mask_label]
    train_labels = train_labels[mask_label]

    # original mode
    approach = APPROACHES[0]
    approaches = [
            approach(
                explainer=explainer(global_values.classifier),
                dimensionality_reduction_techniques=dimensionality_reduction_technique
            )
            for explainer, dimensionality_reduction_technique
            in product(RETRAIN_EXPLAINERS, DIMENSIONALITY_REDUCTION_TECHNIQUES)
        ]

    for idx, approach in (bar := tqdm(list(enumerate(approaches)))):
        bar.set_description(f'Using the approache ({approach})')
        explainer = approach.get_explainer()
        # Extract some information about the current approach
        contributions = approach.generate_contributions_by_data(train_data, predictions_cat)
        
        print(contributions.shape)
        # save contributions
        np.save(f"logs/mnist/contributions/train_data_only_{EXPECTED_LABEL}_{explainer.__class__.__name__}", contributions)

def generate_heatmaps_test_data():

    test_data = global_values.test_data
    test_labels = global_values.test_labels

    mask_label = np.array(test_labels == global_values.EXPECTED_LABEL)
    test_data = test_data[mask_label]
    test_labels = test_labels[mask_label]
    predictions_cat = to_categorical(test_labels, 10)
    predictions_cat = predictions_cat[mask_label]

    approaches = [
            approach(
                explainer=explainer(global_values.classifier),
                dimensionality_reduction_techniques=None
            )
            for explainer
            in EXPLAINERS
        ]

    for idx, approach in (bar := tqdm(list(enumerate(approaches)))):
        bar.set_description(f'Using the approache ({approach})')
        explainer = approach.get_explainer()
        # Extract some information about the current approach
        contributions = approach.generate_contributions_by_data(test_data, test_labels)
        
        # save contributions
        np.save(f"logs/mnist/contributions/train_data_only_{EXPECTED_LABEL}_{explainer.__class__.__name__}", contributions[0])





if __name__ == "__main__":
    generate_heatmaps_train_data()






