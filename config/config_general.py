from typing import Callable

import numpy as np
from clusim.clusimelement import element_sim
from utils.clusters.similarity_metrics import intra_pairs_similarity, custom_similarity
from clusim.clustering import Clustering

from utils.clusters.extractor import get_frac_misses
from utils.images.image_similarity.featuremap_based import manhattan_sim
from utils.images.image_similarity.geometry_based import ssim
from utils.images.image_similarity.intensity_based import euclidean_similarity, mean_squared_similarity
from utils.images.postprocessing import mask_noise

CLUSTERS_SORT_METRIC: Callable[[list], tuple] = lambda cluster: (
    -get_frac_misses(cluster)
    if get_frac_misses(cluster) != 1
    else 0,
    -len(cluster)
)
CLUSTERS_SIMILARITY_METRIC: Callable[[Clustering, Clustering], float] =  custom_similarity #element_sim #custom_similarity #intra_pairs_similarity 


def IMAGES_SIMILARITY_METRIC(lhs, rhs, threshold: float = None, max_activation: float = None, num_bins: int = 2):
    # lhs_processed = lhs
    # rhs_processed = rhs
    # if threshold is not None:
    #     lhs_processed, _ = mask_noise(lhs_processed, normalize=True, threshold=threshold)
    #     rhs_processed, _ = mask_noise(rhs_processed, normalize=True, threshold=threshold)
    # if max_activation is not None:
    #     lhs_processed = np.digitize(lhs_processed, np.linspace(0, max_activation, num_bins))
    #     rhs_processed = np.digitize(rhs_processed, np.linspace(0, max_activation, num_bins))
    return euclidean_similarity(lhs, rhs)

def INDIVIDUAL_SIMILARITY_METRIC(lhs, rhs, max_manhattan):
    return manhattan_sim(lhs, rhs, max_manhattan)

HUMAN_EVALUATION_APPROACHES = [
    'Lime',
    'IntegratedGradients',
    # 'GradCAMPP',
    'moves+orientation+bitmaps(100)_clustered'
]

