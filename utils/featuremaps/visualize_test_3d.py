
from functools import reduce
from itertools import combinations
from feature_map.mnist.feature import Feature
from feature_map.mnist.utils.feature_map.preprocess import extract_samples_and_stats
from feature_map.mnist.utils.feature_map.visualize import unpack_plot_data

import numpy as np
from matplotlib import pyplot as plt

from config.config_data import EXPECTED_LABEL
from config.config_featuremaps import NUM_CELLS, MAP_DIMENSIONS
from utils.featuremaps.postprocessing import process_featuremaps_data
from utils.general import save_figure, scale_values_in_range
import matplotlib.cm as cm

def visualize_colorful_3d_map(coverage_data, misbehavior_data, color_data):
    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    # Set the 3d plot
    ax = fig.add_subplot(projection='3d')
    # Get the coverage and misbehavior data for the plot

    x_coverage, y_coverage, z_coverage, values_coverage = unpack_plot_data(coverage_data)
    x_misbehavior, y_misbehavior, z_misbehavior, values_misbehavior = unpack_plot_data(misbehavior_data)
    x_color, y_color, z_color, values_color = unpack_plot_data(color_data)
    sizes_coverage, sizes_misbehavior = scale_values_in_range([values_coverage, values_misbehavior], 100, 1000)
    # Plot the data
    # ax.scatter(x_coverage, y_coverage, z_coverage, s=sizes_coverage, alpha=.4, label='all')
    ax.scatter(x_misbehavior, y_misbehavior, z_misbehavior, s=sizes_misbehavior, c=values_color, alpha=.8,
               label='misclassified')
    ax.legend(markerscale=.3, frameon=False, bbox_to_anchor=(.3, 1.1))
    return fig, ax



if __name__ == '__main__':

    ll_clusters = [([5.00000000e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00], [0, 7, 41, 18, 85, 24, 26, 61, 63]), 
    ([4.05882353e-01, 1.47301698e-01, 9.97269173e-01, 1.00000000e+00],[1, 4, 10, 16, 20, 86, 25, 90, 95, 42, 43, 44, 46, 110, 55, 127]), 
    ([3.11764706e-01, 2.91389747e-01, 9.89091608e-01, 1.00000000e+00],[64, 2, 45, 28, 15, 48, 81, 112, 117, 22, 58, 123, 60]), 
    ([2.17647059e-01, 4.29120609e-01, 9.75511968e-01, 1.00000000e+00],[65, 3, 39, 70, 38, 94, 31]), 
    ([1.23529412e-01, 5.57489439e-01, 9.56604420e-01, 1.00000000e+00],[5]), 
    ([2.94117647e-02, 6.73695644e-01, 9.32472229e-01, 1.00000000e+00],[17, 84, 6, 56, 11, 12, 125]), 
    ([7.25490196e-02, 7.82927610e-01, 9.00586702e-01, 1.00000000e+00],[96, 66, 103, 8, 111]), 
    ([1.66666667e-01, 8.66025404e-01, 8.66025404e-01, 1.00000000e+00],[98, 67, 68, 101, 9, 59, 77, 14, 50, 51, 52, 21, 91]), 
    ([2.60784314e-01, 9.30229309e-01, 8.26734175e-01, 1.00000000e+00],[105, 34, 69, 73, 13, 30]), 
    ([3.54901961e-01, 9.74138602e-01, 7.82927610e-01, 1.00000000e+00],[97, 129, 99, 131, 102, 107, 108, 79, 80, 113, 19, 23, 87, 29]), 
    ([4.49019608e-01, 9.96795325e-01, 7.34844967e-01, 1.00000000e+00],[27]), 
    ([5.50980392e-01, 9.96795325e-01, 6.78235117e-01, 1.00000000e+00],[32, 36, 54, 71, 75]), 
    ([6.45098039e-01, 9.74138602e-01, 6.22112817e-01, 1.00000000e+00],[72, 33, 74, 35]), 
    ([7.39215686e-01, 9.30229309e-01, 5.62592752e-01, 1.00000000e+00],[76, 37]), 
    ([8.33333333e-01, 8.66025404e-01, 5.00000000e-01, 1.00000000e+00],[121, 83, 100, 40, 89, 62, 47]), 
    ([9.27450980e-01, 7.82927610e-01, 4.34676422e-01, 1.00000000e+00],[49, 115, 116, 119, 124]), 
    ([1.00000000e+00, 6.73695644e-01, 3.61241666e-01, 1.00000000e+00],[57, 53, 92, 93]), 
    ([1.00000000e+00, 5.57489439e-01, 2.91389747e-01, 1.00000000e+00],[88, 104, 78]), 
    ([1.00000000e+00, 4.29120609e-01, 2.19946358e-01, 1.00000000e+00],[82, 109, 106]), 
    ([1.00000000e+00, 2.91389747e-01, 1.47301698e-01, 1.00000000e+00],[120, 114]), 
    ([1.00000000e+00, 1.47301698e-01, 7.38525275e-02, 1.00000000e+00],[128, 118, 126]), 
    ([1.00000000e+00, 1.22464680e-16, 6.12323400e-17, 1.00000000e+00],[122, 130])]


    hl_clusters = [([5.00000000e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00], [0, 5, 108, 13, 115, 116, 119, 87, 124, 95]), 
    ([3.90196078e-01, 1.71625679e-01, 9.96283653e-01, 1.00000000e+00],[1, 66, 67, 4, 37, 68, 7, 39, 9, 44, 76, 18, 50, 52, 55, 27, 28, 61]), 
    ([2.80392157e-01, 3.38158275e-01, 9.85162233e-01, 1.00000000e+00],[2, 130, 12, 15, 17, 23, 29, 30, 33, 34, 35, 40, 41, 43, 47, 57, 59, 63, 69, 70, 72, 73, 74, 117, 121, 122]), 
    ([1.70588235e-01, 4.94655843e-01, 9.66718404e-01, 1.00000000e+00],[64, 65, 3, 10, 22, 58]), 
    ([6.07843137e-02, 6.36474236e-01, 9.41089253e-01, 1.00000000e+00],[6, 38, 105, 42, 11, 14, 16, 19, 20, 21, 51, 56, 26]),
    ([5.68627451e-02, 7.67362681e-01, 9.05873422e-01, 1.00000000e+00],[8, 45, 46, 49, 53, 54, 60, 62]), 
    ([1.66666667e-01, 8.66025404e-01, 8.66025404e-01, 1.00000000e+00],[99, 101, 107, 79, 48, 80, 84, 24]), 
    ([2.76470588e-01, 9.38988361e-01, 8.19740483e-01, 1.00000000e+00],[32, 128, 129, 131, 36, 71, 75, 111, 113, 118, 25, 127, 125, 126, 31]), 
    ([3.86274510e-01, 9.84086337e-01, 7.67362681e-01, 1.00000000e+00],[96, 97, 103, 123, 77, 112, 114, 85, 120, 91, 92, 93, 94]), 
    ([5.03921569e-01, 9.99981027e-01, 7.04925547e-01, 1.00000000e+00],[88, 104, 78]), 
    ([6.13725490e-01, 9.84086337e-01, 6.41213315e-01, 1.00000000e+00],[98, 100, 102, 110, 81, 83, 89, 90]), 
    ([7.23529412e-01, 9.38988361e-01, 5.72735140e-01, 1.00000000e+00],[82, 109, 106]), 
    ([8.33333333e-01, 8.66025404e-01, 5.00000000e-01, 1.00000000e+00],[86])] 
    # ([9.43137255e-01, 7.67362681e-01, 4.23548513e-01, 1.00000000e+00], ), 
    # ( [1.00000000e+00, 6.36474236e-01, 3.38158275e-01, 1.00000000e+00], ), 
    # ([1.00000000e+00, 4.94655843e-01, 2.55842778e-01, 1.00000000e+00], ), 
    # ( [1.00000000e+00, 3.38158275e-01, 1.71625679e-01, 1.00000000e+00], ), 
    # ([1.00000000e+00, 1.71625679e-01, 8.61329395e-02, 1.00000000e+00], ), 
    # ([1.00000000e+00, 1.22464680e-16, 6.12323400e-17, 1.00000000e+00], ),
    # ([8.43137255e-02, 6.07538946e-01, 9.47177357e-01, 1.00000000e+00], )]

    samples, stats = extract_samples_and_stats()
    # Get the list of features
    features = [
        Feature(feature_name, feature_stats['min'], feature_stats['max'])
        for feature_name, feature_stats in stats.to_dict().items()
    ]
    # colors = cm.rainbow(np.linspace(0, 1, len(ll_clusters)+2))

    # Create one visualization for each pair of self.axes selected in order
    data = []
    map_dimensions = [
        min(3, map_dimension) for map_dimension
        in (MAP_DIMENSIONS if type(MAP_DIMENSIONS) is list else [MAP_DIMENSIONS])
    ]
    # Compute all the 2d and 3d feature combinations
    features_combinations = reduce(
        lambda acc, comb: acc + comb,
        [
            list(combinations(features, n_features))
            for n_features in map_dimensions]
    )
    for features_combination in features_combinations:
        features_comb_str = '+'.join([feature.feature_name for feature in features_combination])
        map_size_str = f'{NUM_CELLS}x{NUM_CELLS}x{NUM_CELLS}'
        # Place the values over the map
        
        # Log the information
    feature_comb_str = "+".join([feature.feature_name for feature in features])
    print(f'Using the features {feature_comb_str}')

    # Compute the shape of the map (number of cells of each feature)
    shape = [feature.num_cells for feature in features]
    # Keep track of the samples in each cell, initialize a matrix of empty arrays
    archive_data = np.empty(shape=shape, dtype=list)
    for idx in np.ndindex(*archive_data.shape):
        archive_data[idx] = []

    color_data = np.empty(shape=shape, dtype=list)
    for idx in np.ndindex(*color_data.shape):
        color_data[idx] = None

    color_data_fm = np.empty(shape=shape, dtype=list)
    for idx in np.ndindex(*color_data_fm.shape):
        color_data_fm[idx] = None

    # Count the number of items in each cell
    coverage_data = np.zeros(shape=shape, dtype=int)
    misbehavior_data = np.zeros(shape=shape, dtype=int)

    # Initialize the matrix of clusters to empty lists
    clusters = np.empty(shape=shape, dtype=list)
    for idx in np.ndindex(*clusters.shape):
        clusters[idx] = []


    for idx, sample in enumerate(samples):
        # Coordinates reason in terms of bins 1, 2, 3, while data is 0-indexed
        coords = tuple([feature.get_coordinate_for(sample) - 1 for feature in features])
        # Archive the sample

        archive_data[coords].append(sample)       

        # Increment the coverage
        coverage_data[coords] += 1
        # Increment the misbehaviour
        if sample.is_misbehavior:
            misbehavior_data[coords] += 1
        # Update the clusters
        clusters[coords].append(idx)

        for cluster in hl_clusters:            
            if idx in cluster[1]:
                color_data_fm[coords] = cluster[0]
            

        for cluster in ll_clusters:
            if idx in cluster[1]:
                color = cluster[0]

        if coverage_data[coords] != 0:
            if color_data[coords] == None:
                color_data[coords] = color
            elif color_data[coords] != color:
                color_data[coords] = 'gray' #[1.00000000e+00, 1.22464680e-16, 6.12323400e-17, 1.00000000e+00]


    # Handle the case of 3d maps

    fig, ax = visualize_colorful_3d_map(coverage_data, misbehavior_data, color_data)
    # Handle the case of 2d maps


    # Set the style
    fig.suptitle(f'Feature map: digit {EXPECTED_LABEL}', fontsize=16)
    ax.set_xlabel(features_combination[0].feature_name)
    ax.set_ylabel(features_combination[1].feature_name)
    if len(features_combination) == 3:
        ax.set_zlabel(features_combination[2].feature_name)
    # Export the figure
    save_figure(fig, f'out/featuremaps/test_hm_{EXPECTED_LABEL}_{map_size_str}_{features_comb_str}')

    # Record the data
    data.append({
        'approach': features_comb_str,
        'map_size': NUM_CELLS,
        'clusters': clusters
    })

