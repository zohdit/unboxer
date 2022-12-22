import csv
from itertools import product
from matplotlib import pyplot as plt
from tqdm import tqdm
import os.path
import random
import tensorflow as tf

from config.config_data import EXPECTED_LABEL, classifier
from config.config_heatmaps import EXPLAINERS
from feature_map.mnist.utils import vectorization_tools
from config.config_outputs import NUM_SAMPLES
from utils import generate_inputs
from utils.clusters.Approach import OriginalMode
from utils.general import save_figure


def export_clusters_sample_images():
    __BASE_DIR = 'out/human_evaluation/mnist'
    from feature_map.mnist.sample import Sample
    _, _, test_data, test_labels, predictions = generate_inputs.load_inputs()
    mask_label = (test_labels == EXPECTED_LABEL)
    test_data = test_data[mask_label]
    test_labels = test_labels[mask_label]
    predictions = predictions[mask_label]

    # select 15 random images 
    sample_indexes = range(len(test_data)) #[40, 13, 54, 140, 234, 125, 124, 103, 62, 164, 123, 0, 3, 187, 153] #random.sample(range(len(test_data)), 15) # #
    print(sample_indexes)

    sample_data = test_data[sample_indexes]
    sample_data_gs = tf.image.rgb_to_grayscale(sample_data)
    sample_labels = test_labels[sample_indexes]
    sample_predictions = predictions[sample_indexes]
    sample_predictions_cat = tf.keras.utils.to_categorical(sample_predictions)

    
    # Collect the approaches to use
    print('Collecting the approaches ...')
    # Select the approach from the configurations
    approach = OriginalMode

    # Select the dimensionality reduction techniques based on the approach
    dimensionality_reduction_techniques = [[]] 
    # Collect the approaches
    approaches = [
        approach(
            explainer=explainer(classifier),
            dimensionality_reduction_techniques=dimensionality_reduction_technique
        )
        for explainer, dimensionality_reduction_technique
        in product(EXPLAINERS, dimensionality_reduction_techniques)
    ]
    with open('out/human_evaluation/mnist/human_evaluation_images.csv', mode='w') as f:
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
                cmap='gray',
                extent=(0, image_gs.shape[0], image_gs.shape[1], 0)
            )
            plt.close()
            save_figure(fig, os.path.join(__BASE_DIR, f'mnist_{element_idx}'))

            writer.writerow([element_idx, features, sample_predictions[element_idx]])


def export_clusters_sample_texts():
    __BASE_DIR = 'out/human_evaluation/imdb'
    from feature_map.imdb.sample import Sample

    _, _, test_data, test_labels, predictions = generate_inputs.load_inputs()
    mask_label = (test_labels == EXPECTED_LABEL)
    test_data = test_data[mask_label]
    test_labels = test_labels[mask_label]
    predictions = predictions[mask_label]

    filtered_idx = []
    for idx in range(len(test_data)):
        if len(test_data[idx]) < 500:
            filtered_idx.append(idx)

    sample_data = test_data[filtered_idx]
    sample_labels =  test_labels[filtered_idx]
    sample_predictions = predictions[filtered_idx]

    # select 15 random text 
    sample_indexes = random.sample(range(len(sample_data)), NUM_SAMPLES)
    print(sample_indexes)

    sample_data = test_data[sample_indexes]
    sample_labels = test_labels[sample_indexes]
    sample_predictions = predictions[sample_indexes]
   
    # Collect the approaches to use
    print('Collecting the approaches ...')
    # Select the approach from the configurations
    approach = OriginalMode

    # Select the dimensionality reduction techniques based on the approach
    dimensionality_reduction_techniques = [[]] 
    # Collect the approaches
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
                # contributions = approach.generate_contributions_by_data([text], [prediction])
                explainer = approach.get_explainer()
                data = explainer.export_explanation([text], [label], os.path.join(__BASE_DIR, f'imdb_{explainer.__class__.__name__}_{element_idx}_heatmap'))
 
                data = sorted(data, key=lambda tup: abs(tup[1]), reverse=True)[:10]
                new_data = [[data[0][0], round(float(data[0][1]), 2)]]
                for item in data:
                    if item[0] not in list(zip(*new_data))[0]:
                        new_data.append([item[0],item[1]])
                
                new_data = list(reversed(new_data))
                fig, ax = plt.subplots(1, 1, figsize=(5,2))
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
                ax.tick_params(right=False, labelbottom=False, bottom=False, labelsize=9) 
                color = ['r' if y<0 else 'g' for y in  list(zip(*new_data))[1]]
                ax.barh(list(zip(*new_data))[0], list(zip(*new_data))[1], height=0.4, color=color)
                for bars in ax.containers:
                    ax.bar_label(bars, size=7, fmt='%.2f')
                ax.set_xlim(left=-0.9, right=1) 
                plt.close(fig)
                save_figure(fig, os.path.join(__BASE_DIR, f'imdb_{explainer.__class__.__name__}_{element_idx}_heatmap'))


                # Visualize the text
                writer.writerow([element_idx, explainer.__class__.__name__, text, features, sample_predictions[element_idx]])


  

if __name__ == "__main__":
    export_clusters_sample_images()
    # export_clusters_sample_texts()