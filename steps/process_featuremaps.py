import warnings

from config.config_dirs import FEATUREMAPS_DATA
from config.config_featuremaps import CASE_STUDY, FEATUREMAPS_CLUSTERING_MODE
from utils.featuremaps.postprocessing import process_featuremaps_data

BASE_DIR = f'out/featuremaps/{FEATUREMAPS_CLUSTERING_MODE.name}'


def main():
    warnings.filterwarnings('ignore')

    if CASE_STUDY == "MNIST":
        import feature_map.mnist.feature_map as feature_map_generator_mnist
        featuremaps_df, samples = feature_map_generator_mnist.main()

        # Process the feature-maps and get the dataframe
        print('Extracting the clusters data from the feature-maps mnist ...')
        featuremaps_df = process_featuremaps_data(featuremaps_df, samples)
        featuremaps_df.to_pickle(FEATUREMAPS_DATA)
    
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
