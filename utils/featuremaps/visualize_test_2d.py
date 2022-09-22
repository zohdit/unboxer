
from functools import reduce
from itertools import combinations
from feature_map.mnist.feature import Feature
from feature_map.mnist.utils.feature_map.preprocess import extract_samples_and_stats
from feature_map.mnist.utils.feature_map.visualize import unpack_plot_data

import numpy as np
from matplotlib import pyplot as plt

from config.config_data import EXPECTED_LABEL
from config.config_featuremaps import NUM_CELLS
from utils.featuremaps.postprocessing import process_featuremaps_data
from utils.general import save_figure, scale_values_in_range
import matplotlib.cm as cm

def visualize_colorful_2d_map(coverage_data, misbehavior_data, color_data):
    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    # Set the 3d plot
    ax = fig.add_subplot()
    # Get the coverage and misbehavior data for the plot

    x_coverage, y_coverage, values_coverage = unpack_plot_data(coverage_data)
    x_misbehavior, y_misbehavior, values_misbehavior = unpack_plot_data(misbehavior_data)
    x_color, y_color, values_color = unpack_plot_data(color_data)
    sizes_coverage, sizes_misbehavior = scale_values_in_range([values_coverage, values_misbehavior], 100, 1000)
    # Plot the data
    # ax.scatter(x_coverage, y_coverage, z_coverage, s=sizes_coverage, alpha=.4, label='all')
    ax.scatter(x_misbehavior, y_misbehavior, s=sizes_misbehavior, c=values_color, alpha=.8,
               label='misclassified')
    ax.legend(markerscale=.3, frameon=False, bbox_to_anchor=(.3, 1.1))
    return fig, ax



if __name__ == '__main__':

    ll_clusters = [([5.00000000e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00], [0, 1, 133, 44, 48, 112, 188, 154, 92]), 
    ([4.05882353e-01, 1.47301698e-01, 9.97269173e-01, 1.00000000e+00],[2, 135, 111, 113, 150, 87, 183]), 
    ([3.11764706e-01, 2.91389747e-01, 9.89091608e-01, 1.00000000e+00],[3, 196, 102, 134, 74, 171, 76, 173, 174, 81, 51, 20, 180]), 
    ([2.17647059e-01, 4.29120609e-01, 9.75511968e-01, 1.00000000e+00],[4, 10, 178, 157, 190]), 
    ([1.23529412e-01, 5.57489439e-01, 9.56604420e-01, 1.00000000e+00],[164, 5, 108, 83, 85, 89, 59]), 
    ([2.94117647e-02, 6.73695644e-01, 9.32472229e-01, 1.00000000e+00],[161, 98, 130, 193, 197, 6, 106, 11, 45, 109, 144, 149, 118, 57, 27, 191]), 
    ([7.25490196e-02, 7.82927610e-01, 9.00586702e-01, 1.00000000e+00],[99, 68, 100, 165, 39, 7, 72, 73, 103, 12, 13, 105, 166, 145, 116, 187, 125]), 
    ([1.66666667e-01, 8.66025404e-01, 8.66025404e-01, 1.00000000e+00],[192, 38, 167, 8, 198, 170, 199, 77, 17, 50, 179, 121, 122, 156]), 
    ([2.60784314e-01, 9.30229309e-01, 8.26734175e-01, 1.00000000e+00],[9]), 
    ([3.54901961e-01, 9.74138602e-01, 7.82927610e-01, 1.00000000e+00],[97, 169, 203, 14, 186, 63]), 
    ([4.49019608e-01, 9.96795325e-01, 7.34844967e-01, 1.00000000e+00],[64, 128, 67, 131, 107, 79, 15, 19, 147, 91]), 
    ([5.50980392e-01, 9.96795325e-01, 6.78235117e-01, 1.00000000e+00],[37, 71, 16, 124, 31]), 
    ([6.45098039e-01, 9.74138602e-01, 6.22112817e-01, 1.00000000e+00],[200, 18, 110]), 
    ([7.39215686e-01, 9.30229309e-01, 5.62592752e-01, 1.00000000e+00],[159, 46, 143, 115, 21, 54, 182, 24, 25, 127, 158, 95]), 
    ([8.33333333e-01, 8.66025404e-01, 5.00000000e-01, 1.00000000e+00],[32, 96, 146, 22, 123, 28]), 
    ([9.27450980e-01, 7.82927610e-01, 4.34676422e-01, 1.00000000e+00],[163, 202, 172, 141, 78, 23]), 
    ([1.00000000e+00, 6.73695644e-01, 3.61241666e-01, 1.00000000e+00],[80, 82, 93, 53, 56, 185, 26, 61]), 
    ([1.00000000e+00, 5.57489439e-01, 2.91389747e-01, 1.00000000e+00],[43, 29]), 
    ([1.00000000e+00, 4.29120609e-01, 2.19946358e-01, 1.00000000e+00],[65, 101, 138, 140, 175, 49, 117, 181, 119, 120, 90, 30]), 
    ([1.00000000e+00, 2.91389747e-01, 1.47301698e-01, 1.00000000e+00],[33, 42, 142]), 
    ([1.00000000e+00, 1.47301698e-01, 7.38525275e-02, 1.00000000e+00],[41, 34, 35, 47]), 
    ([1.00000000e+00, 1.22464680e-16, 6.12323400e-17, 1.00000000e+00],[36]),
    ([3.90196078e-01, 1.71625679e-01, 9.96283653e-01, 1.00000000e+00],[40, 114, 139]),
    ([2.80392157e-01, 3.38158275e-01, 9.85162233e-01, 1.00000000e+00], [52, 69, 189]),
    ([1.70588235e-01, 4.94655843e-01, 9.66718404e-01, 1.00000000e+00],[55]), 
    ([6.07843137e-02, 6.36474236e-01, 9.41089253e-01, 1.00000000e+00],[168, 137, 75, 177, 58]),
    ([5.68627451e-02, 7.67362681e-01, 9.05873422e-01, 1.00000000e+00],[160, 60, 184, 84]), 
    ([1.66666667e-01, 8.66025404e-01, 8.66025404e-01, 1.00000000e+00],[62]), 
    ([2.76470588e-01, 9.38988361e-01, 8.19740483e-01, 1.00000000e+00],[104, 66]), 
    ([3.86274510e-01, 9.84086337e-01, 7.67362681e-01, 1.00000000e+00],[136, 129, 148, 70]), 
    ([5.03921569e-01, 9.99981027e-01, 7.04925547e-01, 1.00000000e+00],[194, 86]), 
    ([6.13725490e-01, 9.84086337e-01, 6.41213315e-01, 1.00000000e+00],[88, 195, 151]), 
    ([7.23529412e-01, 9.38988361e-01, 5.72735140e-01, 1.00000000e+00],[176, 94]),  
    ([1.00000000e+00, 4.94655843e-01, 2.55842778e-01, 1.00000000e+00], [126]), 
    ( [1.00000000e+00, 3.38158275e-01, 1.71625679e-01, 1.00000000e+00], [155, 132]), 
    ([1.00000000e+00, 1.71625679e-01, 8.61329395e-02, 1.00000000e+00], [152])]



    hl_clusters = [([5.00000000e-01, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00], [0, 66, 35, 104, 77, 23, 154, 191]), 
    ([3.90196078e-01, 1.71625679e-01, 9.96283653e-01, 1.00000000e+00],[160, 1, 36, 5, 200, 112, 179, 84, 85, 86, 60]), 
    ([2.80392157e-01, 3.38158275e-01, 9.85162233e-01, 1.00000000e+00],[2, 103, 74, 11, 43, 109, 20, 117, 181, 87, 152, 26, 187, 156, 29, 30]), 
    ([1.70588235e-01, 4.94655843e-01, 9.66718404e-01, 1.00000000e+00],[162, 3, 131, 163, 7, 72, 41, 140, 141, 82, 21, 150, 123, 28, 157]), 
    ([6.07843137e-02, 6.36474236e-01, 9.41089253e-01, 1.00000000e+00],[34, 4, 164, 169, 107, 172, 78, 142, 24, 27, 158, 63]),
    ([5.68627451e-02, 7.67362681e-01, 9.05873422e-01, 1.00000000e+00],[128, 129, 6, 134, 12, 13, 16, 18, 19, 148, 171, 178, 180, 182, 183, 59, 64, 195, 70, 71, 81, 83, 89, 111, 113, 118, 119, 120, 124]), 
    ([1.66666667e-01, 8.66025404e-01, 8.66025404e-01, 1.00000000e+00],[100, 8, 136, 139, 47, 114, 147, 22, 25, 31], [99, 132, 101, 133, 135, 9, 138, 201, 108, 46, 50, 56, 153, 155, 62, 127]), 
    ([2.76470588e-01, 9.38988361e-01, 8.19740483e-01, 1.00000000e+00],[32, 33, 97, 10, 42, 44, 110, 143, 145, 115, 54, 88, 185, 91, 189]), 
    ([3.86274510e-01, 9.84086337e-01, 7.67362681e-01, 1.00000000e+00],[14, 144, 149, 45, 173, 174, 177, 55, 188, 65, 67, 197, 199, 73, 76, 93, 96, 102, 126]), 
    ([5.03921569e-01, 9.99981027e-01, 7.04925547e-01, 1.00000000e+00],[193, 196, 15, 184, 186]), 
    ([6.13725490e-01, 9.84086337e-01, 6.41213315e-01, 1.00000000e+00],[37, 40, 106, 170, 202, 79, 17, 53, 151, 90, 190]), 
    ([7.23529412e-01, 9.38988361e-01, 5.72735140e-01, 1.00000000e+00],[176, 49, 38, 94]),  
    ([1.00000000e+00, 4.94655843e-01, 2.55842778e-01, 1.00000000e+00], [159, 161, 98, 194, 68, 69, 166, 39, 168, 137, 203, 51, 52, 116, 58, 95]), 
    ( [1.00000000e+00, 3.38158275e-01, 1.71625679e-01, 1.00000000e+00], [192, 198, 167, 105, 48, 80, 125, 121, 122, 92, 61]), 
    ([1.00000000e+00, 1.71625679e-01, 8.61329395e-02, 1.00000000e+00], [130, 165, 75, 175, 146, 57])]

    samples, stats = extract_samples_and_stats()
    # Get the list of features
    features = [
        Feature(feature_name, feature_stats['min'], feature_stats['max'])
        for feature_name, feature_stats in stats.to_dict().items()
    ]
    # colors = cm.rainbow(np.linspace(0, 1, len(ll_clusters)+2))

    # Create one visualization for each pair of self.axes selected in order
    data = []
    map_dimensions = [2]
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
        feature_comb_str = "+".join([feature.feature_name for feature in features_combination])
        print(f'Using the features {feature_comb_str}')

        # Compute the shape of the map (number of cells of each feature)
        shape = [feature.num_cells for feature in features_combination]
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
            coords = tuple([feature.get_coordinate_for(sample) - 1 for feature in features_combination])
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

        fig, ax = visualize_colorful_2d_map(coverage_data, misbehavior_data, color_data_fm)
        # Handle the case of 2d maps


        # Set the style
        fig.suptitle(f'Feature map: digit {EXPECTED_LABEL}', fontsize=16)
        ax.set_xlabel(features_combination[0].feature_name)
        ax.set_ylabel(features_combination[1].feature_name)
        if len(features_combination) == 3:
            ax.set_zlabel(features_combination[2].feature_name)
        # Export the figure
        save_figure(fig, f'out/featuremaps/test_fm_{EXPECTED_LABEL}_{map_size_str}_{features_comb_str}')

    # Record the data
    data.append({
        'approach': features_comb_str,
        'map_size': NUM_CELLS,
        'clusters': clusters
    })

