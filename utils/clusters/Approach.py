import abc
from cmath import exp
import pickle
from config.config_data import EXPECTED_LABEL, INPUT_MAXLEN, NUM_INPUTS
from config.config_featuremaps import CASE_STUDY

import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from clusim.clustering import Clustering

from config.config_general import IMAGES_SIMILARITY_METRIC
from config.config_heatmaps import CLUSTERING_TECHNIQUE
from utils import generate_inputs
from utils.stats import compute_comparison_matrix


class Approach(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, explainer, dimensionality_reduction_techniques):
        self.__explainer = explainer
        self.__dimensionality_reduction_techniques = dimensionality_reduction_techniques
        self.__clustering_technique = CLUSTERING_TECHNIQUE

    @abc.abstractmethod
    def load_contributions(self) -> np.ndarray:
        """
        load the contributions for the predictions
        :return: The contributions for the predictions
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def cluster_contributions(self, contributions: np.ndarray) -> tuple:
        """
        Cluster the contributions
        :param contributions: The contributions
        :return: The clusters for the contributions as membership list
        """
        raise NotImplementedError()

    def get_clustering_technique(self):
        return self.__clustering_technique

    def get_dimensionality_reduction_techniques(self):
        return self.__dimensionality_reduction_techniques

    def get_explainer(self):
        return self.__explainer

    def _generate_contributions_by_data(
            self,
            data,
            labels,
            only_positive: bool = True
    ) -> np.ndarray:
        # Generate the contributions
        try:
            contributions = self.__explainer.explain(data, labels)
        except:
            # The explainer expects grayscale images
            try:
                if CASE_STUDY == "MNIST":
                    contributions = self.__explainer.explain(
                        tf.image.rgb_to_grayscale(data),
                        labels
                    )
                elif CASE_STUDY == "IMDB":
                        with open('in/models/tokenizer.pickle', 'rb') as handle:
                            tokenizer = pickle.load(handle)
                        x_generated = tokenizer.texts_to_sequences(data)
                        data = tf.keras.preprocessing.sequence.pad_sequences(x_generated, maxlen=INPUT_MAXLEN)
                        contributions = self.__explainer.explain(
                            data,
                            labels
                        )
            except ValueError:
                # The explainer doesn't work with grayscale images
                return np.array([])
        # Convert the contributions to grayscale
        if CASE_STUDY == "MNIST":
            try:
                contributions = np.squeeze(tf.image.rgb_to_grayscale(contributions).numpy())
            except tf.errors.InvalidArgumentError:
                pass
            # Filter for the positive contributions
            if only_positive:
                contributions = np.ma.masked_less(np.squeeze(contributions), 0).filled(0)

        return contributions

    def _load_contributions(self, explainer, iter):
        if CASE_STUDY == "MNIST":
            contributions = np.load(f"logs/mnist/contributions/test_data_{EXPECTED_LABEL}_{explainer.__class__.__name__}_{iter}.npy")
        elif CASE_STUDY == "IMDB":
            contributions = np.load(f"logs/imdb/contributions/test_data_{EXPECTED_LABEL}_{explainer.__class__.__name__}_{iter}.npy")
        return contributions

    def __str__(self):
        params = [(technique.get_params().get('perplexity'), technique.get_params().get('n_components')) for technique in self.__dimensionality_reduction_techniques]
        return f'{self.__explainer.__class__.__name__} - perplexity, dimensions = {params}'

class LocalLatentMode(Approach):

    def __init__(self, explainer, dimensionality_reduction_techniques):
        super(LocalLatentMode, self).__init__(explainer, dimensionality_reduction_techniques)
    
    def generate_contributions_by_data(self, data, labels):
        # Generate the contributions for the filtered data
        return super(LocalLatentMode, self)._generate_contributions_by_data(data, labels)

    def cluster_contributions(self, contributions: np.ndarray) -> tuple:
        _, _, test_data, test_labels, test_predictions = generate_inputs.load_inputs()
        mask_label = np.array(test_labels == EXPECTED_LABEL)
        contributions = contributions[mask_label]
        contributions = contributions[:NUM_INPUTS]
        # Flatten the contributions and project them in the latent space
        contributions_flattened = contributions.reshape(contributions.shape[0], -1)
        projections = np.array([])
        for dim_red_tech in self.get_dimensionality_reduction_techniques():
            projections = dim_red_tech.fit_transform(contributions_flattened)
        # Cluster the projections
        clusters = CLUSTERING_TECHNIQUE().fit_predict(projections)
        # Compute the silhouette for the clusters
        try:
            score = silhouette_score(projections, clusters)
        except ValueError:
            score = np.nan
        return clusters, projections, score

    def load_contributions(self, explainer, iter):
        return super(LocalLatentMode, self)._load_contributions(explainer, iter)

class GlobalLatentMode(Approach):

    def __init__(self, explainer, dimensionality_reduction_techniques):
        super(GlobalLatentMode, self).__init__(explainer, dimensionality_reduction_techniques)

    def generate_contributions_by_data(self, data, labels):
        # Generate the contributions for the filtered data
        return super(GlobalLatentMode, self)._generate_contributions_by_data(data, labels)

    def cluster_contributions(self, contributions: np.ndarray) -> tuple:
        # Flatten the contributions and project them into the latent space
        contributions_flattened = contributions.reshape(contributions.shape[0], -1)
        projections = np.array([])
        for dim_red_tech in self.get_dimensionality_reduction_techniques():
            projections = dim_red_tech.fit_transform(contributions_flattened)
        # Cluster the filtered projections
        _, _, test_data, test_labels, test_predictions = generate_inputs.load_inputs()
        mask_label = np.array(test_labels == EXPECTED_LABEL)
        projections_filtered = projections[mask_label]
        projections_filtered = projections_filtered[:NUM_INPUTS]
        clusters = CLUSTERING_TECHNIQUE().fit_predict(projections_filtered)
        # Compute the silhouette score for the clusters
        try:
            score = silhouette_score(projections_filtered, clusters)
        except ValueError:
            score = np.nan
        return clusters, projections_filtered, score

    def load_contributions(self, explainer, iter):
        return super(GlobalLatentMode, self)._load_contributions(explainer, iter)

class OriginalMode(Approach):

    def __init__(self, explainer, dimensionality_reduction_techniques):
        super(OriginalMode, self).__init__(explainer, dimensionality_reduction_techniques)

    def generate_contributions_by_data(self, data, labels):
        # Generate the contributions for the filtered data
        return super(OriginalMode, self)._generate_contributions_by_data(data, labels)

    @staticmethod
    def multiprocessing_metric(pair):
        return IMAGES_SIMILARITY_METRIC(pair[0], pair[1])

    def cluster_contributions(self, contributions: np.ndarray) -> tuple:
        _, _, test_data, test_labels, test_predictions = generate_inputs.load_inputs()
        mask_label = np.array(test_labels == EXPECTED_LABEL)
        contributions = contributions[mask_label]
        contributions = contributions[:NUM_INPUTS]
        # Compute the similarity matrix for the contributions
        similarity_matrix = compute_comparison_matrix(
            list(contributions),
            approaches=[],
            metric=IMAGES_SIMILARITY_METRIC,
            args=None,
            show_progress_bar=True,
            multi_process=False
        )
        # Cluster the contributions using the similarity matrix

        # sometime affinity gives only one cluster!
        clusters = CLUSTERING_TECHNIQUE(affinity='precomputed').fit_predict(similarity_matrix)
        while len(Clustering().from_membership_list(clusters).to_cluster_list()) == 1:
            clusters = CLUSTERING_TECHNIQUE(affinity='precomputed').fit_predict(similarity_matrix)
        # Compute the silhouette for the clusters
        try:
            distance_matrix = 1 - similarity_matrix
            np.fill_diagonal(distance_matrix, 0)
            score = silhouette_score(distance_matrix, clusters, metric='precomputed')
        except ValueError:
            score = np.nan

        # Flatten the contributions and project them into the latent space
        contributions_flattened = contributions.reshape(contributions.shape[0], -1)
        projections = TSNE(perplexity=1).fit_transform(contributions_flattened)

        return clusters, projections, score

    def load_contributions(self, explainer, iter):
        return super(OriginalMode, self)._load_contributions(explainer, iter)
