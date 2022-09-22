import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from functools import reduce
from itertools import combinations
import time
import warnings
import pandas as pd
import os
from IPython.display import display

from config.config_dirs import FEATUREMAPS_DATA
from config.config_featuremaps import CASE_STUDY, FEATUREMAPS_CLUSTERING_MODE, MAP_DIMENSIONS
from feature_map.mnist.utils.feature_map.preprocess import extract_samples_and_stats

from utils.featuremaps.postprocessing import process_featuremaps_data

BASE_DIR = f'out/featuremaps/{FEATUREMAPS_CLUSTERING_MODE.name}'


def main():
    warnings.filterwarnings('ignore')

    if CASE_STUDY == "MNIST":
        import feature_map.mnist.feature_map as feature_map_generator_mnist
        from feature_map.mnist.feature import Feature
        # Extract the samples and the stats
        start_time = time.time()
        samples, stats = extract_samples_and_stats()
        # Get the list of features
        features = [
            Feature(feature_name, feature_stats['min'], feature_stats['max'])
            for feature_name, feature_stats in stats.to_dict().items()
        ]
        features_extraction_time = time.time() - start_time
        

        # Create one visualization for each pair of self.axes selected in order
        features_combinations = reduce(
            lambda acc, comb: acc + comb,
            [
                list(combinations(features, n_features))
                for n_features in MAP_DIMENSIONS]
        )
        for features_combination in features_combinations:
            data = []
            # Visualize the feature-maps
            print('Generating the featuremaps ...')
            data, samples = feature_map_generator_mnist.main(features_combination, features_extraction_time, samples)

            # Process the feature-maps and get the dataframe
            print('Extracting the clusters data from the feature-maps mnist ...')
            featuremaps_df = process_featuremaps_data(data, samples, features_combination)

            # Update the current data or create a new dataframe
            if os.path.isfile(FEATUREMAPS_DATA):
                old_data = pd.read_pickle(FEATUREMAPS_DATA)
                new_data = featuremaps_df
                features_df = pd.concat([old_data, new_data]).drop_duplicates(subset=['approach'], keep='last')
                features_df = features_df.reset_index(drop=True)
            else:
                features_df = featuremaps_df
            
            features_df.to_pickle(FEATUREMAPS_DATA)

            display(features_df)
    
    elif CASE_STUDY == "IMDB":
        import feature_map.imdb.feature_map as feature_map_generator_imdb
        featuremaps_df, samples = feature_map_generator_imdb.main()

        # Process the feature-maps and get the dataframe
        print('Extracting the clusters data from the feature-maps imdb ...')
        featuremaps_df = process_featuremaps_data(featuremaps_df, samples)
        featuremaps_df.to_pickle(FEATUREMAPS_DATA)

    return featuremaps_df


if __name__ == '__main__':
    main()
