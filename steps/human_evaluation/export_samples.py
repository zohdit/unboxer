import csv
import math
import os.path
from random import shuffle
import random
import shutil
from xml.etree import cElementTree
from config.config_data import EXPECTED_LABEL

from feature_map.mnist.utils import vectorization_tools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from config.config_general import HUMAN_EVALUATION_APPROACHES, IMAGES_SIMILARITY_METRIC
from config.config_outputs import NUM_IMAGES_PER_CLUSTER, NUM_SEPARABILITY_CLUSTERS
from steps.human_evaluation.helpers import preprocess_data, sample_clusters
from utils import global_values
from utils.clusters.extractor import get_labels_purity, get_central_elements
from utils.clusters.postprocessing import get_misclassified_items
from utils.general import save_figure
from utils.lists.processor import weight_values
from utils.plotter.visualize import visualize_cluster_images, visualize_cluster_texts
from IPython.display import display
from utils.text.processor import random_words, words_from_contribution

__BASE_DIR = 'out/human_evaluation'

def export_clusters_sample_images():
    from feature_map.mnist.sample import Sample
    df = preprocess_data()

    df = df.groupby('approach').first().reset_index()

    # Iterate over the approaches
    df = df.set_index('approach')
    approaches_df = []
    for approach in HUMAN_EVALUATION_APPROACHES:
        try:
            approaches_df.append(df.loc[[approach]])
        except KeyError:
            continue
    if len(approaches_df) < len(HUMAN_EVALUATION_APPROACHES):
        # Sample to reach the same length
        remaining = list(set(df.index) - set(HUMAN_EVALUATION_APPROACHES))
        for approach_df in [
            df.loc[[approach]]
            for approach
            in np.random.choice(
                remaining,
                min(len(HUMAN_EVALUATION_APPROACHES) - len(approaches_df), len(remaining)),
                replace=False
            )
        ]:
            approaches_df.append(approach_df)
    df = pd.concat(approaches_df, axis=0)
    approaches = df.index.values

    with open('out/human_evaluation/human_evaluation_images.csv', mode='w') as f:
        writer = csv.writer(f)
        for approach in tqdm(approaches, desc='Collecting the data for the approaches'):
            # Get the clusters and contributions for the selected approach
            cluster_list, contributions = df.loc[approach][['clusters', 'contributions']]
            # Filter the clusters for the misclassified elements
            # cluster_list = [get_misclassified_items(cluster) for cluster in cluster_list]
            # Keep the clusters with more than one misclassified element
            cluster_list = [cluster for cluster in cluster_list if len(cluster) > 3]
            cluster_list = np.array(cluster_list, dtype=list)
            shuffle(cluster_list)
            cluster_list = cluster_list[:NUM_SEPARABILITY_CLUSTERS]
            # Get the contributions or the images themselves
            label_images = np.array((global_values.generated_data_gs[global_values.generated_labels == global_values.EXPECTED_LABEL]))
            contributions = np.array(contributions)
            # Process the clusters
            for idx, cluster in tqdm(
                    list(enumerate(cluster_list)),
                    desc='Visualizing the central elements for the clusters',
                    leave=False
            ):
                if "moves" in approach:
                    contributions = None
                    cluster = list(cluster)
                # Get the central elements in the cluster
                central_elements = get_central_elements(
                    cluster,
                    cluster_elements=contributions[cluster] if contributions is not None else label_images[cluster],
                    elements_count=NUM_IMAGES_PER_CLUSTER,
                    metric=lambda lhs, rhs: 1 - IMAGES_SIMILARITY_METRIC(lhs, rhs),
                    show_progress_bar=False
                )
                central_elements = np.array(central_elements)
                
                mask_label = (global_values.generated_labels == EXPECTED_LABEL)
                test_data_gs = global_values.generated_data_gs[mask_label]
                test_labels =  global_values.generated_labels[mask_label]
                predictions = global_values.generated_predictions[mask_label]

                features = []
                for element in central_elements:
                    image = test_data_gs[element]
                    label = test_labels[element]
                    prediction = predictions[element]
                    xml_desc = vectorization_tools.vectorize(image)
                    sample = Sample(desc=xml_desc, label=label, prediction=prediction, image=image)
                    features.append(sample.features)


                # Visualize the central elements
                fig, ax = visualize_cluster_images(
                    central_elements,
                    images=label_images,
                    labels='auto'
                )
                plt.close(fig)
                save_figure(fig, os.path.join(__BASE_DIR, f'{approach}_{idx}'))


                # Visualize the central elements
                fig, ax = visualize_cluster_images(
                    central_elements,
                    images=label_images,
                    labels='auto',
                    overlays=contributions

                )
                plt.close(fig)
                save_figure(fig, os.path.join(__BASE_DIR, f'{approach}_{idx}_heatmap'))

                sub_path = f'{__BASE_DIR}/{approach}_{idx}.png'
                writer.writerow([idx, f'{sub_path}', approach, central_elements, features])

def export_clusters_sample_texts():
    from feature_map.imdb.sample import Sample
    df = preprocess_data()

    df = df.groupby('approach').first().reset_index()

    # Iterate over the approaches
    df = df.set_index('approach')
    approaches_df = []
    for approach in HUMAN_EVALUATION_APPROACHES:
        try:
            approaches_df.append(df.loc[[approach]])
        except KeyError:
            continue
    if len(approaches_df) < len(HUMAN_EVALUATION_APPROACHES):
        # Sample to reach the same length
        remaining = list(set(df.index) - set(HUMAN_EVALUATION_APPROACHES))
        for approach_df in [
            df.loc[[approach]]
            for approach
            in np.random.choice(
                remaining,
                min(len(HUMAN_EVALUATION_APPROACHES) - len(approaches_df), len(remaining)),
                replace=False
            )
        ]:
            approaches_df.append(approach_df)
    df = pd.concat(approaches_df, axis=0)
    approaches = df.index.values

    with open('out/human_evaluation/human_evaluation_texts.csv', mode='w') as f:
        writer = csv.writer(f)
        for approach in tqdm(approaches, desc='Collecting the data for the approaches'):
            # Get the clusters and contributions for the selected approach
            cluster_list, contributions = df.loc[approach][['clusters', 'contributions']]
            # Filter the clusters for the misclassified elements
            # cluster_list = [get_misclassified_items(cluster) for cluster in cluster_list]
            # Keep the clusters with more than one misclassified element
            cluster_list = [cluster for cluster in cluster_list if len(cluster) > 0]
            cluster_list = np.array(cluster_list, dtype=list)
            shuffle(cluster_list)
            # cluster_list = cluster_list[:NUM_SEPARABILITY_CLUSTERS]
            # Get the contributions or the images themselves
            label_images = np.array((global_values.generated_data[global_values.generated_labels == global_values.EXPECTED_LABEL]))
            contributions = np.array(contributions)
            # Process the clusters
            for idx, cluster in tqdm(
                    list(enumerate(cluster_list)),
                    desc='Visualizing the central elements for the clusters',
                    leave=False
            ):
                # if "poscount" in approach:
                #     contributions = None
                    # cluster = list(cluster)
                    # central_elements= random.sample(cluster, 1)
                # else:
                #     # Get the central elements in the cluster
                #     central_elements = get_central_elements(
                #         cluster,
                #         cluster_elements=contributions[cluster] if contributions is not None else label_images[cluster],
                #         elements_count=NUM_IMAGES_PER_CLUSTER,
                #         metric=lambda lhs, rhs: 1 - IMAGES_SIMILARITY_METRIC(lhs, rhs),
                #         show_progress_bar=False
                #     )
                central_elements = list(cluster)
                central_elements = np.array(central_elements)
                
                mask_label = (global_values.generated_labels == EXPECTED_LABEL)
                test_data = global_values.generated_data[mask_label]
                test_labels =  global_values.generated_labels[mask_label]
                predictions = global_values.generated_predictions[mask_label]

                features = []
                for element in central_elements:
                    text = test_data[element]
                    label = test_labels[element]
                    prediction = predictions[element]
                    sample = Sample(text=text, label=label, prediction=prediction)
                    features = sample.features

                
                # Visualize the central elements
                # fig, ax = visualize_cluster_texts(
                #     central_elements,
                #     texts=label_images,
                #     labels='auto'
                # )
                # plt.close(fig)
                # save_figure(fig, os.path.join(__BASE_DIR, f'{approach}_{idx}'))


                # # Visualize the central elements
                # fig, ax = visualize_cluster_images(
                #     central_elements,
                #     images=label_images,
                #     labels='auto',
                #     overlays=contributions

                # )
                # plt.close(fig)
                # save_figure(fig, os.path.join(__BASE_DIR, f'{approach}_{idx}_heatmap'))




                    sub_path = f'{__BASE_DIR}/{approach}_{idx}.png'
                    id = element
                    writer.writerow([idx, f'{sub_path}', approach, element, 
                    features, label_images[id], words_from_contribution(id, contributions[id]), random_words()])


if __name__ == "__main__":
    # export_clusters_sample_images()
    export_clusters_sample_texts()