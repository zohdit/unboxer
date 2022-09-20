from config.config_featuremaps import MAP_DIMENSIONS, NUM_CELLS
from feature_map.mnist.sample import Sample


# maximum possible manhattan distance
if MAP_DIMENSIONS[0] == 2:
    max_manhattan = (NUM_CELLS - 1) + (NUM_CELLS - 1)
elif MAP_DIMENSIONS[0] == 3:
    max_manhattan = (NUM_CELLS-1) + (NUM_CELLS-1) + (NUM_CELLS-1)

def manhattan_dist(coords_ind1, coords_ind2):
    if MAP_DIMENSIONS[0] == 2:
        return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1])
    else:
        return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1]) + abs(coords_ind1[2] - coords_ind2[2])


def manhattan_sim(lhs: Sample, rhs: Sample) -> float:
    """
    Compute the manhattan sim between two inds
    :param lhs: The first ind
    :param rhs: The second ind
    :return: The manhattan sim between the two samples
    """
    
    _manhattan = manhattan_dist(lhs.coords, rhs.coords)
    
    return 1 - (_manhattan/max_manhattan)