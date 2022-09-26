
from itertools import product
from heatmap.mnist.heatmap import generate_heatmaps_by_classifier
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


from steps.process_heatmaps import APPROACHES, get_perplexity
from config.config_data import USE_RGB
from config.config_heatmaps import DIMENSIONALITY_REDUCTION_TECHNIQUES, RETRAIN_EXPLAINERS
from steps.train_model import create_model
from utils.clusters.Approach import OriginalMode
from utils.clusters.compare import compare_approaches_by_index
from utils.clusters.postprocessing import select_data_by_cluster
from utils.dataset import get_train_test_data
from config.config_heatmaps import APPROACH, DIMENSIONALITY_REDUCTION_TECHNIQUES


def retrain_by_heatmap_provided_explanations():
    mnist_loader = lambda: mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(
        dataset_loader=mnist_loader,
        rgb=USE_RGB,
        verbose=True
    )
    test_data = tf.image.rgb_to_grayscale(test_data).numpy()
    part = 0
    classifier = tf.keras.models.load_model(f"out/models/orig_model_{part}.h5")

    generate_heatmaps_by_classifier(classifier)

    accs1 = []
    accs2 = []
    with open("logs/mnist/hm_acc.txt", "w") as f:
        # repeat the experiment n times
        
        for rep in range(0, 30):
            f.write(f"Run {rep}: \n")

            remaining_train_data = np.load(f"logs/mnist/retrain_data/remaining_data_{part}.npy")
            remaining_train_labels = np.load(f"logs/mnist/retrain_data/remaining_labels_{part}.npy")
            remaining_train_indices = np.load(f"logs/mnist/retrain_data/remaining_indices_{part}.npy")

            selected_train_data = np.load(f"logs/mnist/retrain_data/selected_data_{part}.npy")
            selected_train_labels = np.load(f"logs/mnist/retrain_data/selected_labels_{part}.npy") 
        
            k = 100   
            
            new_train_data = []
            new_train_labels = []
            # using HM select k samples from remaining train data
            heatmaps_df = generate_clusters_by_heatmap_indices(remaining_train_indices, classifier)
            new_train_data, new_train_labels, remaining_train_data, remaining_train_labels, remaining_train_indices = select_data_by_cluster(heatmaps_df, remaining_train_data, remaining_train_labels, k)
            selected_train_data = np.concatenate((new_train_data, selected_train_data), axis=0)  
            selected_train_labels = np.concatenate((new_train_labels, selected_train_labels), axis=0) 
        

            acc1, acc2 = create_model(np.array(selected_train_data), np.array(selected_train_labels), test_data, test_labels, f"retrained_hm_{rep}")
            
            f.write(f"retrained_hm_{rep}: {acc1} , {acc2} \n")
            accs1.append(acc1)
            accs2.append(acc2)
    
    np.save("logs/mnist/hm_acc_all", accs1)
    np.save("logs/mnist/hm_acc_5", accs2)

def generate_clusters_by_heatmap_indices(train_indices, classifier):
        # Collect the approaches to use
    print('Collecting the approaches ...')
    # Select the approach from the configurations
    approach = APPROACHES[APPROACH]
    # Select the dimensionality reduction techniques based on the approach
    dimensionality_reduction_techniques = [[]] if approach is OriginalMode else DIMENSIONALITY_REDUCTION_TECHNIQUES
    # If the processing mode is the original one, or there are no best logs -> try all the combinations

    # Collect the approaches
    approaches = [
        approach(
            explainer=explainer(classifier),
            dimensionality_reduction_techniques=dimensionality_reduction_technique
        )
        for explainer, dimensionality_reduction_technique
        in product(RETRAIN_EXPLAINERS, dimensionality_reduction_techniques)
    ]
    # Collect the data for the approaches
    print('Collecting the data for the approaches ...')
    df_raw = compare_approaches_by_index(
        approaches,
        1,
        train_indices,
        get_info=lambda app: f"perplexity: {get_perplexity(app)}" if get_perplexity(app) != np.nan else "Original Mode"
    )
    return df_raw



if __name__ == "__main__":
    retrain_by_heatmap_provided_explanations()