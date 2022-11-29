import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config_data import EXPECTED_LABEL, NUM_INPUTS
from config.config_dirs import FEATUREMAPS_META
from feature_map.imdb.feature_simulator import FeatureSimulator
from feature_map.imdb.sample import Sample
from feature_map.imdb.utils.general import missing
from utils import generate_inputs


def extract_samples_and_stats():
    """
    Iteratively walk in the dataset and process all the json files.
    For each of them compute the statistics.
    """
    # Initialize the stats about the overall features
    stats = {feature_name: [] for feature_name in FeatureSimulator.get_simulators().keys()}

    data_samples = []

    _, _, generated_data, generated_labels, generated_predictions = generate_inputs.load_inputs()

    filtered = list(filter(lambda t: t[1] == EXPECTED_LABEL, zip(
        generated_data,
        generated_labels,
        generated_predictions
    )))
    for text, label, prediction in tqdm(filtered[:NUM_INPUTS], desc='Extracting the samples and the statistics'):
        sample = Sample(text=text, label=label, prediction=prediction)
        data_samples.append(sample)
        # update the stats
        for feature_name, feature_value in sample.features.items():
            stats[feature_name].append(feature_value)

    stats = pd.DataFrame(stats)
    # compute the stats values for each feature
    stats = stats.agg(['min', 'max', missing, 'count'])
    print(stats.transpose())
    stats.to_csv(FEATUREMAPS_META, index=True)

    return data_samples, stats

def extract_samples_and_stats_by_data(filtered):
    """
    Iteratively walk in the dataset by input and process all the json files.
    For each of them compute the statistics.
    """
    # Initialize the stats about the overall features
    stats = {feature_name: [] for feature_name in FeatureSimulator.get_simulators().keys()}

    data_samples = []

    for text, label, prediction in tqdm(filtered, desc='Extracting the samples and the statistics'):
        sample = Sample(text=text, label=label, prediction=prediction)
        data_samples.append(sample)
        # update the stats
        for feature_name, feature_value in sample.features.items():
            stats[feature_name].append(feature_value)

    stats = pd.DataFrame(stats)
    # compute the stats values for each feature
    stats = stats.agg(['min', 'max', missing, 'count'])
    print(stats.transpose())
    # stats.to_csv(FEATUREMAPS_META, index=True)

    return data_samples, stats