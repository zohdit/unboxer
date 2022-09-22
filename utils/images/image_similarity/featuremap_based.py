from config.config_featuremaps import MAP_DIMENSIONS
from feature_map.mnist.sample import Sample


def manhattan_dist(coords_ind1, coords_ind2):
    if MAP_DIMENSIONS[0] == 2:
        return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1])
    else:
        return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1]) + abs(coords_ind1[2] - coords_ind2[2])


def manhattan_sim(lhs: Sample, rhs: Sample, max_manhattan) -> float:
    """
    Compute the manhattan sim between two inds
    :param lhs: The first ind
    :param rhs: The second ind
    :return: The manhattan sim between the two samples
    """
    
    _manhattan = manhattan_dist(lhs.coords, rhs.coords)
    # [0:different, 1:same)
    return 1 - (_manhattan/max_manhattan)