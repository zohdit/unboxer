import os
import time
import pandas as pd

from config.config_dirs import FEATUREMAPS_DATA_RAW
from feature_map.mnist.feature import Feature
from feature_map.mnist.utils.feature_map.preprocess import extract_samples_and_stats, extract_samples_and_stats_by_data
from feature_map.mnist.utils.feature_map.visualize import visualize_map


def main():
    # Extract the samples and the stats
    start_time = time.time()
    samples, stats = extract_samples_and_stats()
    # Get the list of features
    features = [
        Feature(feature_name, feature_stats['min'], feature_stats['max'])
        for feature_name, feature_stats in stats.to_dict().items()
    ]
    features_extraction_time = time.time() - start_time
    # Visualize the feature-maps
    data = visualize_map("featuremap", features, samples)
    for features_dict in data:
        features_dict.update({'features_extraction': features_extraction_time})

    # Update the current data or create a new dataframe
    # if os.path.isfile(FEATUREMAPS_DATA_RAW):
    #     old_data = pd.read_pickle(FEATUREMAPS_DATA_RAW)
    #     new_data = pd.DataFrame(data)
    #     features_df = pd.concat([old_data, new_data]).drop_duplicates(subset=['approach', 'map_size'], keep='last')
    #     features_df = features_df.reset_index(drop=True)
    # else:
    features_df = pd.DataFrame(data)

    features_df.to_pickle(FEATUREMAPS_DATA_RAW)

    return features_df, samples


def generate_featuremap_by_data(train_data, train_labels):
    
    filtered = list(zip(
        train_data,
        train_labels,
        train_labels
    ))

    samples, stats = extract_samples_and_stats_by_data(filtered)
    # Get the list of features
    features = [
        Feature(feature_name, feature_stats['min'], feature_stats['max'])
        for feature_name, feature_stats in stats.to_dict().items()
    ]

    # Visualize the feature-maps
    data = visualize_map("retrain_featuremap", features, samples)

    features_df = pd.DataFrame(data)

    features_df.to_pickle(FEATUREMAPS_DATA_RAW)

    return features_df, samples


if __name__ == "__main__":
    main()
    
