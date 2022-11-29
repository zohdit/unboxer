import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from itertools import product
import numpy as np

from steps.process_heatmaps import APPROACHES
from config.config_data import EXPECTED_LABEL, classifier
from config.config_heatmaps import EXPLAINERS, ITERATIONS
from tensorflow.keras.utils import to_categorical
from utils import generate_inputs
from tqdm import tqdm, trange


def generate_heatmaps_test_data():

    _, _, test_data, test_labels, test_predictions = generate_inputs.load_inputs()

    test_predictions_cat = to_categorical(test_predictions)
    print("data shape:")
    print(test_data.shape)

    approach = APPROACHES[0]
    approaches = [
            approach(
                explainer=explainer(classifier),
                dimensionality_reduction_techniques=[[]] 
            )
            for explainer
            in EXPLAINERS
    ]

    for idx, approach in (bar := tqdm(list(enumerate(approaches)))):
        for iter in trange(ITERATIONS, desc='Iterating', leave=False):
            # bar.set_description(f'Using the approache ({approach})')
            explainer = approach.get_explainer()
            # Extract some information about the current approach

            contributions = approach.generate_contributions_by_data(test_data, test_predictions_cat)                         
                
            # save contributions
            np.save(f"logs/mnist/contributions/test_data_{EXPECTED_LABEL}_{explainer.__class__.__name__}_{iter}", contributions)



if __name__ == "__main__":
    generate_heatmaps_test_data()







