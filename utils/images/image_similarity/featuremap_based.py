from config.config_featuremaps import MAP_DIMENSIONS, NUM_CELLS
from feature_map.mnist.sample import Sample


# maximum possible manhattan distance
if MAP_DIMENSIONS[0] == 2:
    max_manhattan = (NUM_CELLS * NUM_CELLS)
elif MAP_DIMENSIONS[0] == 3:
    max_manhattan = (NUM_CELLS * NUM_CELLS * NUM_CELLS)

def manhattan_dist(coords_ind1, coords_ind2):
    if MAP_DIMENSIONS[0] == 2:
        return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1])
    else:
        return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1]) + abs(coords_ind1[2] - coords_ind2[2])


def manhattan(lhs: Sample, rhs: Sample) -> float:
    """
    Compute the manhattan distance between two inds
    :param lhs: The first ind
    :param rhs: The second ind
    :return: The manhattan distance between the two samples
    """
    _manhattan = 0
    lhs_coords = []
    for lhs_feature_name, lhs_feature_value in lhs.features.items():
        lhs_coords.append(lhs_feature_value)
    
    rhs_coords = []
    for rhs_feature_name, rhs_feature_value in rhs.features.items():
        rhs_coords.append(rhs_feature_value)

    _manhattan = _manhattan + manhattan_dist(lhs_coords, rhs_coords)
    
    return 1 - (_manhattan/max_manhattan)