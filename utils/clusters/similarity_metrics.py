from itertools import combinations

import numpy as np
from clusim.clustering import Clustering


def __intra_pairs(clusters_membership: np.ndarray) -> set:
    # Find the labels for the clusters in the configuration
    clusters_labels = np.unique(clusters_membership)
    return set([
        pair
        # Iterate through the cluster labels
        for label in clusters_labels
        # intrapairs = combinations of two elements in the clusters
        for pair in combinations(np.where(clusters_membership == label)[0], 2)
    ])


def intra_pairs_similarity(lhs: Clustering, rhs: Clustering) -> float:
    """
    Compute the intrapairs similarity between two cluster configurations
    :param lhs: The first cluster configuration
    :param rhs: The second cluster configuration
    :return: The fraction of common intrapairs between the two configurations
    """
    # Convert the clusters to membership list
    lhs, rhs = lhs.to_membership_list(), rhs.to_membership_list()
    # Extract the intrapairs from the two configurations
    intra_lhs, intra_rhs = __intra_pairs(lhs), __intra_pairs(rhs)
    # Compute the fraction of common intrapairs
    score = len(intra_lhs.intersection(intra_rhs)) * 2 / (len(intra_lhs) + len(intra_rhs))
    return score


def custom_similarity(lhs: Clustering, rhs: Clustering) -> float:
    """
    Compute a custom similarity between a high level clustering and low level clustering
    The custom similarity is computed based on color distribution in high level clusters
    :param lhs: the low level cluster
    :param rhs: the high level cluster
    :return: The similarity between two clusters
    """

    lhs, rhs = lhs.to_cluster_list(), rhs.to_cluster_list()

    clone_rhs = rhs.copy()

    color = -1
    for cluster in lhs:
        for ind in cluster:
            for cluster in clone_rhs:
                for idx in range(len(cluster)):
                    if cluster[idx] == ind:
                        cluster[idx] = color
        color -= 1

    gini_impurity = 0
    # compute impurity of eacb high level cluster
    for cluster in clone_rhs:
        # if cluster is singletoin imputrity is 0
        if len(cluster) == 1:
            impurity = 0
        # otherwise impurity is sum of manhattan distances 
        else:           
            p_total = list(set(cluster)) 
            sigma_p_color = 0
            for color in p_total:
                p_color = cluster.count(color)/len(cluster)
                sigma_p_color = sigma_p_color + (p_color*p_color)    
            impurity = 1 - sigma_p_color

        gini_impurity = impurity + gini_impurity
    
    return 1 - gini_impurity/len(clone_rhs)    


