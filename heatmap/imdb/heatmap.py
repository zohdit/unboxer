
from itertools import product

from tensorflow.keras.datasets import mnist
from config.config_data import EXPECTED_LABEL, USE_RGB
import numpy as np
import tensorflow as tf
from config.config_heatmaps import DIMENSIONALITY_REDUCTION_TECHNIQUES, EXPLAINERS, RETRAIN_EXPLAINERS
from utils import global_values
from tqdm import tqdm, trange
from keras.utils.np_utils import to_categorical

from utils.clusters.Approach import GlobalLatentMode, LocalLatentMode, OriginalMode


from itertools import islice






APPROACHES = [OriginalMode, LocalLatentMode, GlobalLatentMode]

def generate_heatmaps_train_data():

    train_data_padded = np.load(f"in/data/imdb-cached/x_train.npy")
    train_labels = np.load(f"in/data/imdb-cached/y_train.npy")

    mask_label = np.array(train_labels == global_values.EXPECTED_LABEL)
    train_data_padded = train_data_padded[mask_label]
    predictions_cat = to_categorical(train_labels, 2)
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

    for idx, approach in (tqdm(list(enumerate(approaches)))):
        # bar.set_description(f'Using the approache ({approach})')
        explainer = approach.get_explainer()
        # Extract some information about the current approach

        chunks_data = np.array_split(train_data_padded, 100)
        chunks_pred = np.array_split(predictions_cat, 100)

        for i in range(len(chunks_data)):
            if i == 0:
                contributions = approach.generate_contributions_by_data(chunks_data[i], chunks_pred[i])
            else:
                contributions = np.concatenate((contributions, approach.generate_contributions_by_data(chunks_data[i], chunks_pred[i])), axis=0)
        
        print(contributions.shape)
        # save contributions
        np.save(f"logs/imdb/contributions/train_data_only_{EXPECTED_LABEL}", contributions)

def generate_heatmaps_test_data():

    test_data = global_values.test_data_padded
    test_labels = global_values.test_labels

    mask_label = np.array(test_labels == global_values.EXPECTED_LABEL)
    test_data = test_data[mask_label]
    test_labels = test_labels[mask_label]
    predictions_cat = to_categorical(test_labels, 2)
    predictions_cat = predictions_cat[mask_label]

    approaches = [
            approach(
                explainer=explainer(global_values.classifier),
                dimensionality_reduction_techniques=None
            )
            for explainer
            in EXPLAINERS
        ]

    for idx, approach in (tqdm(list(enumerate(approaches)))):
        # bar.set_description(f'Using the approache ({approach})')
        explainer = approach.get_explainer()
        # Extract some information about the current approach
        contributions = approach.generate_contributions_by_data(test_data, test_labels)
        
        # save contributions
        np.save(f"logs/imdb/contributions/train_data_only_{EXPECTED_LABEL}_{approach.explainer}", contributions[0])





if __name__ == "__main__":
    generate_heatmaps_train_data()






