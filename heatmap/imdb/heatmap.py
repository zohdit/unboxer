

import random
from tqdm import tqdm, trange
import numpy as np

from steps.process_heatmaps import APPROACHES
from config.config_data import EXPECTED_LABEL, classifier
from config.config_heatmaps import APPROACH, EXPLAINERS, ITERATIONS
from utils import generate_inputs


def generate_heatmaps_test_data():

    _, _, test_data, test_labels, test_predictions = generate_inputs.load_inputs()

    print("data shape:")
    print(test_data.shape)

    approach = APPROACHES[APPROACH]
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
            try:          
                chunks_data = np.array_split(test_data, 100)
                chunks_pred = np.array_split(test_predictions, 100)
                for i in range(len(chunks_data)):
                    if i == 0:
                        contributions = approach.generate_contributions_by_data(chunks_data[i], chunks_pred[i])
                    else:
                        contributions = np.concatenate((contributions, approach.generate_contributions_by_data(chunks_data[i], chunks_pred[i])), axis=0)
            except ValueError:
                # The explainer doesn't work with texts
                contributions = np.array([])
                
            # save contributions
            np.save(f"logs/imdb/contributions/test_data_{EXPECTED_LABEL}_{explainer.__class__.__name__}_{iter}", contributions)



if __name__ == "__main__":
    generate_heatmaps_test_data()











