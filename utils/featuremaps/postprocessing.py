from hashlib import new
import numpy as np
import pandas as pd
from scipy.ndimage import label
from clusim.clustering import Clustering
from IPython.display import display

from config.config_featuremaps import FEATUREMAPS_CLUSTERING_MODE
from utils.featuremaps.FeaturemapsClusteringMode import FeaturemapsClusteringMode
from utils.stats import compute_comparison_matrix
from config.config_general import INDIVIDUAL_SIMILARITY_METRIC
from config.config_heatmaps import CLUSTERING_TECHNIQUE

def process_featuremaps_data(df, samples):
    """
    Process the featuremaps data and prepare it for the next steps
    :param df: The data extracted for the featuremaps
    :return: The processed data for the featuremaps
    """
    # Create the base dataframe with the features and the cells containing the clusters
    original_df = pd.DataFrame.copy(df, deep=True)
    
    # # Compute the merged clusters
    if FEATUREMAPS_CLUSTERING_MODE == FeaturemapsClusteringMode.REDUCED:
        merged_df = pd.DataFrame.copy(original_df, deep=True)
        # Compute the number of items in each cell
        merged_df['cells_size'] = merged_df['clusters'].apply(
            np.vectorize(lambda cell: len(cell) if cell is not None else 0)
        )
        # Get the matrix for the connected components
        shape = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]
        merged_df['connected_components'] = merged_df['cells_size'].apply(lambda matrix: label(matrix, shape)[0])
        # Get the list of merged clusters
        merged_df['clusters_list'] = merged_df.apply(
            lambda row: __merge_clusters(row['clusters'], row['connected_components'])
            , axis=1
        )
        merged_df['mode'] = 'reduced'
        merged_df = merged_df.drop(columns=['connected_components', 'cells_size'])
        complete_df = merged_df 
    
    elif FEATUREMAPS_CLUSTERING_MODE == FeaturemapsClusteringMode.CLUSTERED:
        original_df['mode'] = 'clustered'
        original_df['clusters_list'] = original_df['clusters'].apply(__get_clusters_list)
        complete_df = pd.DataFrame.copy(original_df, deep=True) 
        complete_df['clusters_list'] = complete_df['clusters'].apply(__recluster, args=[samples])

    else:
        original_df['mode'] = 'original'
        original_df['clusters_list'] = original_df['clusters'].apply(__get_clusters_list)
        complete_df = original_df

    complete_df = complete_df.drop(columns='clusters')
    complete_df = complete_df.rename({'clusters_list': 'clusters'}, axis=1)
    complete_df = complete_df.sort_values(['map_size', 'approach', 'mode']).reset_index(drop=True)
    return complete_df

def __recluster(clusters_matrix: np.ndarray, samples: list):
    # Compute the similarity matrix for the contributions
    similarity_matrix = compute_comparison_matrix(
            samples,
            approaches=[],
            metric=INDIVIDUAL_SIMILARITY_METRIC,
            show_progress_bar=True,
            multi_process=False
    )
    # Cluster the individuals using the similarity matrix
    _clusters = CLUSTERING_TECHNIQUE(affinity='precomputed').fit_predict(similarity_matrix)
    cluster_list = Clustering().from_membership_list(_clusters).to_cluster_list()
    print(cluster_list)
    return np.array(cluster_list)


def __get_clusters_list(clusters_matrix: np.ndarray) -> np.ndarray:
    # Flatten the matrix of clusters
    flattened = clusters_matrix.flatten()
    # Filter for the non-empty clusters
    non_empty = flattened[np.vectorize(lambda c: len(c) > 0)(flattened)]
    return non_empty


def __merge_clusters(clusters_matrix: np.ndarray, connected_components: np.ndarray) -> np.ndarray:
    merged_clusters = np.array([])
    for component_id in np.unique(connected_components):
        component_clusters = clusters_matrix[np.where(connected_components == component_id)]
        component_clusters = component_clusters[np.vectorize(lambda c: len(c) > 0)(component_clusters)]
        if len(component_clusters) > 0:
            merged_clusters = np.append(merged_clusters, np.concatenate(component_clusters))

    return merged_clusters
