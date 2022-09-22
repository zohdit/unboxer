import time
from typing import Callable
from config.config_heatmaps import CLUSTERING_TECHNIQUE

import numpy as np
import pandas as pd
from clusim.clustering import Clustering
from tqdm import tqdm, trange

from utils.clusters.Approach import Approach


def compare_approaches(
        approaches: list,
        iterations: int,
        get_info: Callable[[Approach], str] = None
) -> pd.DataFrame:
    """
    Compare a list of approaches and return the collected data
    :param approaches: The list of approaches to compare
    :param iterations: The number of iterations to process each approach
    :param get_info: A function to get the string information about the current approach
    :return: A dataframe with the data collected during the tries
    """
    # Iterate over the approaches in the list
    data = []
    df = pd.DataFrame()
    # bar := 
    for idx, approach in (tqdm(list(enumerate(approaches)))):
        # bar.set_description(f'Comparing the approaches ({approach})')
        print((f'Comparing the approaches ({approach})'))
        # Extract some information about the current approach
        explainer = approach.get_explainer()
        dimensionality_reduction_techniques = approach.get_dimensionality_reduction_techniques()
        for _ in trange(iterations, desc='Iterating', leave=False):
            # Generate the contributions
            contributions_start = time.time()
            contributions = approach.generate_contributions()  
            contributions_time = time.time() - contributions_start
            if len(contributions) == 0:
                print("***** Impossible to generate the contributions ")
                # Impossible to generate the contributions -> skip
                continue
            # Cluster the contributions
            cluster_start = time.time()
            try:
                clusters, projections, score = approach.cluster_contributions(contributions)
            except ValueError:
                # No clusters -> silhouette error
                clusters, score, projections = [], np.nan, []
            cluster_time = time.time() - cluster_start

            if len(clusters) == 0:
                print("***** No cluster found")
                # No cluster found -> skip
                continue

            # Collect the information for the run
            data.append({
                'approach': explainer.__class__.__name__,
                'clustering_mode': approach.__class__.__name__,
                'clustering_technique': CLUSTERING_TECHNIQUE.__qualname__,
                'dimensionality_reduction_techniques': dimensionality_reduction_techniques,
                'score': round(score, 3),
                'contributions': contributions,
                'clusters': Clustering().from_membership_list(clusters).to_cluster_list(),
                'projections': projections,
                'time_contributions': round(contributions_time, 5),
                'time_clustering': round(cluster_time, 5)
            })
    df = pd.DataFrame(data)
    return df




def compare_approaches_by_data(
        approaches: list,
        iterations: int,
        train_data,
        train_labels,
        get_info: Callable[[Approach], str] = None
) -> pd.DataFrame:
    """
    Compare a list of approaches and return the collected data
    :param approaches: The list of approaches to compare
    :param iterations: The number of iterations to process each approach
    :param get_info: A function to get the string information about the current approach
    :return: A dataframe with the data collected during the tries
    """
    # Iterate over the approaches in the list
    data = []
    for idx, approach in (tqdm(list(enumerate(approaches)))):
        # bar.set_description(f'Comparing the approaches ({approach})')
        # Extract some information about the current approach
        explainer = approach.get_explainer()
        clustering_technique = approach.get_clustering_technique()
        dimensionality_reduction_techniques = approach.get_dimensionality_reduction_techniques()

        # for _ in trange(iterations, desc='Iterating', leave=False):
            # Generate the contributions
        contributions_start = time.time()
        
        contributions = approach.generate_contributions_by_data(train_data, train_labels)

        contributions_time = time.time() - contributions_start
        if len(contributions) == 0:
            # Impossible to generate the contributions -> skip
            continue
        # Cluster the contributions
        cluster_start = time.time()
        try:
            clusters, projections, score = approach.cluster_contributions(contributions)
        except ValueError:
            # No clusters -> silhouette error
            clusters, score, projections = [], np.nan, []
        cluster_time = time.time() - cluster_start

        if len(clusters) == 0:
            # No cluster found -> skip
            continue

        # Collect the information for the run
        data.append({
            'approach': explainer.__class__.__name__,
            'clustering_mode': approach.__class__.__name__,
            'clustering_technique': clustering_technique.__class__.__name__,
            'dimensionality_reduction_techniques': dimensionality_reduction_techniques,
            'score': round(score, 3),
            'contributions': contributions,
            'clusters': Clustering().from_membership_list(clusters).to_cluster_list(),
            'projections': projections,
            'time_contributions': round(contributions_time, 5),
            'time_clustering': round(cluster_time, 5)
        })
    df = pd.DataFrame(data)
    return df



def compare_approaches_by_index(
        approaches: list,
        iterations: int,
        train_indices,
        get_info: Callable[[Approach], str] = None
) -> pd.DataFrame:
    """
    Compare a list of approaches and return the collected data
    :param approaches: The list of approaches to compare
    :param iterations: The number of iterations to process each approach
    :param get_info: A function to get the string information about the current approach
    :return: A dataframe with the data collected during the tries
    """
    # Iterate over the approaches in the list
    data = []
    for idx, approach in (tqdm(list(enumerate(approaches)))):
        # bar.set_description(f'Comparing the approaches ({approach})')
        # Extract some information about the current approach
        explainer = approach.get_explainer()
        clustering_technique = approach.get_clustering_technique()
        dimensionality_reduction_techniques = approach.get_dimensionality_reduction_techniques()

        # for _ in trange(iterations, desc='Iterating', leave=False):
            # Generate the contributions
        contributions_start = time.time()

        contributions = []

        print(f"{explainer.__class__.__name__} heatmap loading ...")
        heatmaps = np.load(f"logs/mnist/contributions/train_data_only_5_{explainer.__class__.__name__}.npy")

            

        for index in train_indices:
            contributions.append(heatmaps[index])


        contributions_time = time.time() - contributions_start
        cluster_start = time.time()
        try:
            clusters, projections, score = approach.cluster_contributions(np.array(contributions))
        except ValueError:
            # No clusters -> silhouette error
            clusters, score, projections = [], np.nan, []
        cluster_time = time.time() - cluster_start

        if len(clusters) == 0:
            # No cluster found -> skip
            continue

        # Collect the information for the run
        data.append({
            'approach': explainer.__class__.__name__,
            'clustering_mode': approach.__class__.__name__,
            'clustering_technique': clustering_technique.__class__.__name__,
            'dimensionality_reduction_techniques': dimensionality_reduction_techniques,
            'score': round(score, 3),
            'contributions': contributions,
            'clusters': Clustering().from_membership_list(clusters).to_cluster_list(),
            'projections': projections,
            'time_contributions': round(contributions_time, 5),
            'time_clustering': round(cluster_time, 5)
        })
    df = pd.DataFrame(data)
    return df
