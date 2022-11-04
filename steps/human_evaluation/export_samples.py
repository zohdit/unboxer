import csv
from itertools import product

import os.path
from config.config_featuremaps import VOCAB_SIZE

import numpy as np
import random

from config.config_data import EXPECTED_LABEL
from config.config_heatmaps import EXPLAINERS

from feature_map.mnist.utils import vectorization_tools

from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import global_values
from utils.clusters.Approach import OriginalMode
from utils.general import save_figure
from utils.text.processor import top_ten, words_from_contribution




def export_clusters_sample_images():
    __BASE_DIR = 'out/human_evaluation/mnist'
    from feature_map.mnist.sample import Sample

    mask_label = (global_values.generated_labels == EXPECTED_LABEL)
    test_data_gs = global_values.generated_data_gs[mask_label]
    test_data = global_values.generated_data[mask_label]
    test_labels =  global_values.generated_labels[mask_label]
    predictions = global_values.generated_predictions[mask_label]
    predictions_cat = global_values.generated_predictions_cat[mask_label]

    # select 15 random images 
    sample_indexes = random.sample(range(len(test_data)), 15) #[40, 13, 54, 140, 234, 125, 124, 103, 62, 164, 123, 0, 3, 187, 153] #
    print(sample_indexes)

    sample_data_gs = test_data_gs[sample_indexes]
    sample_data = test_data[sample_indexes]
    sample_labels = test_labels[sample_indexes]
    sample_predictions = predictions[sample_indexes]
    sample_predictions_cat = predictions_cat[sample_indexes]
    
    # Collect the approaches to use
    print('Collecting the approaches ...')
    # Select the approach from the configurations
    approach = OriginalMode

    # Select the dimensionality reduction techniques based on the approach
    dimensionality_reduction_techniques = [[]] 
    # Collect the approaches
    classifier = global_values.classifier
    approaches = [
        approach(
            explainer=explainer(classifier),
            dimensionality_reduction_techniques=dimensionality_reduction_technique
        )
        for explainer, dimensionality_reduction_technique
        in product(EXPLAINERS, dimensionality_reduction_techniques)
    ]
    with open('out/human_evaluation/human_evaluation_images.csv', mode='w') as f:
        writer = csv.writer(f)
        for element_idx in range(len(sample_data)):
            image = sample_data[element_idx]
            image_gs = sample_data_gs[element_idx]
            label = sample_labels[element_idx]
            prediction = sample_predictions[element_idx]
            xml_desc = vectorization_tools.vectorize(image_gs)
            sample = Sample(desc=xml_desc, label=label, prediction=prediction, image=image_gs)
            features = sample.features

            for idx, approach in (bar := tqdm(list(enumerate(approaches)))):
                # Generate the contributions
                contributions = approach.generate_contributions_by_data([image], [sample_predictions_cat[element_idx]])
                explainer = approach.get_explainer()
                fig, ax = plt.subplots(1, 1, figsize=(2 * 1, 2 * 1))
                ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                ax.imshow( 
                    image_gs,
                    cmap='gray_r',
                    extent=(0, image_gs.shape[0], image_gs.shape[1], 0)
                )
                ax.imshow(
                        contributions,
                        cmap='Reds',
                        alpha=.7,
                        extent=(0, image_gs.shape[0], image_gs.shape[1], 0)
                    )
                plt.close(fig)
                save_figure(fig, os.path.join(__BASE_DIR, f'mnist_{explainer.__class__.__name__}_{element_idx}_heatmap'))

            # Visualize the image
            fig, ax = plt.subplots(1, 1, figsize=(2 * 1, 2 * 1))
            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax.imshow( 
                image_gs,
                cmap='gray_r',
                extent=(0, image_gs.shape[0], image_gs.shape[1], 0)
            )
            plt.close()
            save_figure(fig, os.path.join(__BASE_DIR, f'mnist_{element_idx}'))

            writer.writerow([element_idx, features, sample_predictions[element_idx]])


def export_clusters_sample_texts():
    __BASE_DIR = 'out/human_evaluation/imdb'
    from feature_map.imdb.sample import Sample

    mask_label = (global_values.generated_labels == EXPECTED_LABEL)
    test_data = global_values.generated_data[mask_label]
    test_labels =  global_values.generated_labels[mask_label]
    predictions = global_values.generated_predictions[mask_label]

    filtered_idx = []
    for idx in range(len(test_data)):
        if len(test_data[idx]) < 500:
            filtered_idx.append(idx)

    sample_data = test_data[filtered_idx]
    sample_labels =  test_labels[filtered_idx]
    sample_predictions = predictions[filtered_idx]

    # select 15 random text 
    sample_indexes = random.sample(range(len(sample_data)), 15)
    print(sample_indexes)

    sample_data = sample_data[sample_indexes]
    sample_labels = sample_labels[sample_indexes]
    sample_predictions = sample_predictions[sample_indexes]


   
    # Collect the approaches to use
    print('Collecting the approaches ...')
    # Select the approach from the configurations
    approach = OriginalMode

    # Select the dimensionality reduction techniques based on the approach
    dimensionality_reduction_techniques = [[]] 
    # Collect the approaches
    classifier = global_values.classifier
    approaches = [
        approach(
            explainer=explainer(classifier),
            dimensionality_reduction_techniques=dimensionality_reduction_technique
        )
        for explainer, dimensionality_reduction_technique
        in product(EXPLAINERS, dimensionality_reduction_techniques)
    ]
    with open('out/human_evaluation/imdb/human_evaluation_texts.csv', mode='w') as f:
        writer = csv.writer(f)
        for element_idx in range(len(sample_data)):
            text = sample_data[element_idx]
            label = sample_labels[element_idx]
            prediction = sample_predictions[element_idx]
            sample = Sample(text=text, label=label, prediction=prediction)
            features = sample.features

            for idx, approach in (bar := tqdm(list(enumerate(approaches)))):
                # Generate the contributions
                contributions = approach.generate_contributions_by_data([text], [prediction])
                explainer = approach.get_explainer()
                explainer.export_explanation([text], [prediction], os.path.join(__BASE_DIR, f'imdb_{explainer.__class__.__name__}_{element_idx}_heatmap'))

                data = top_ten(text, contributions)
                data = list(reversed(data))
                fig, ax = plt.subplots(1, 1, figsize=(3,1))
                ax.tick_params(right=False, labelbottom=False, bottom=False, labelsize=11)                
                ax.barh(list(zip(*data))[0], list(zip(*data))[1], height=0.3, color="red")
                plt.close(fig)
                save_figure(fig, os.path.join(__BASE_DIR, f'imdb_{explainer.__class__.__name__}_{element_idx}_heatmap'))


                # Visualize the text
                writer.writerow([element_idx, explainer.__class__.__name__, text, features, sample_predictions[element_idx], words_from_contribution(text, contributions)])


  

if __name__ == "__main__":
    # export_clusters_sample_images()
    export_clusters_sample_texts()