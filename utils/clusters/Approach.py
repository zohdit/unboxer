import abc
from config.config_data import EXPECTED_LABEL
from config.config_featuremaps import CASE_STUDY

import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from config.config_general import IMAGES_SIMILARITY_METRIC
from config.config_heatmaps import CLUSTERING_TECHNIQUE
from utils import global_values
from utils.stats import compute_comparison_matrix


class Approach(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, explainer, dimensionality_reduction_techniques):
        self.__explainer = explainer
        self.__dimensionality_reduction_techniques = dimensionality_reduction_techniques
        self.__clustering_technique = CLUSTERING_TECHNIQUE

    @abc.abstractmethod
    def generate_contributions(self) -> np.ndarray:
        """
        Generate the contributions for the predictions
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

    def _generate_contributions(
            self,
            mask: np.ndarray = np.ones(len(global_values.generated_data), dtype=bool),
            only_positive: bool = True
    ) -> np.ndarray:
        # Generate the contributions
        if CASE_STUDY == "MNIST":
            try:            
                contributions = self.__explainer.explain(global_values.generated_data[mask], global_values.generated_predictions_cat[mask])
            except ValueError:
                # The explainer expects grayscale images
                try:
                    contributions = self.__explainer.explain(
                        global_values.generated_data_gs[mask],
                        global_values.generated_predictions_cat[mask]
                    )
                except ValueError:
                    # The explainer doesn't work with grayscale images
                    return np.array([])
            # Convert the contributions to grayscale
            try:
                contributions = np.squeeze(tf.image.rgb_to_grayscale(contributions).numpy())
            except tf.errors.InvalidArgumentError:
                pass
                    # Filter for the positive contributions
            if only_positive:
                contributions = np.ma.masked_less(np.squeeze(contributions), 0).filled(0)
                
        elif CASE_STUDY == "IMDB":
            # explanation = self.__explainer.explain(global_values.test_data_padded[mask], baselines=None, target=global_values.predictions_cat[mask]
            # , attribute_to_layer_inputs=False)
            # attrs = explanation.attributions[0]
            # contributions = attrs.sum(axis=2)
            chunks_data = np.array_split(global_values.test_data_padded[mask], 100)
            chunks_pred = np.array_split(global_values.predictions_cat[mask], 100)

            for i in range(len(chunks_data)):
                if i == 0:
                    contributions = self.generate_contributions_by_data(chunks_data[i], chunks_pred[i])
                else:
                    contributions = np.concatenate((contributions, self.generate_contributions_by_data(chunks_data[i], chunks_pred[i])), axis=0)

        return contributions

    def _generate_contributions_by_data(
            self,
            data,
            labels,
            only_positive: bool = True
    ) -> np.ndarray:
        # Generate the contributions
        if CASE_STUDY == "MNIST":
            try:
                contributions = self.__explainer.explain(data, labels)
            except ValueError:
                # The explainer expects grayscale images
                try:
                    contributions = self.__explainer.explain(
                        tf.image.rgb_to_grayscale(data),
                        labels
                    )
                except ValueError:
                    # The explainer doesn't work with grayscale images
                    return np.array([])
            # Convert the contributions to grayscale
            try:
                contributions = np.squeeze(tf.image.rgb_to_grayscale(contributions).numpy())
            except tf.errors.InvalidArgumentError:
                pass
        elif CASE_STUDY == "IMDB":
            # if self.__explainer == "IntegratedGradients":
            explanation = self.__explainer.explain(data, baselines=None, target=labels
            , attribute_to_layer_inputs=False)
            attrs = explanation.attributions[0]
            contributions = attrs.sum(axis=2)
        # Filter for the positive contributions
        if only_positive:
            contributions = np.ma.masked_less(np.squeeze(contributions), 0).filled(0)
        return contributions


    def __str__(self):
        params = [technique.get_params().get('perplexity') for technique in self.__dimensionality_reduction_techniques]
        return f'{self.__explainer.__class__.__name__} - perplexity = {params}'


class LocalLatentMode(Approach):

    def __init__(self, explainer, dimensionality_reduction_techniques):
        super(LocalLatentMode, self).__init__(explainer, dimensionality_reduction_techniques)

    def generate_contributions(self):
        # Generate the contributions for the filtered data
        mask_label = np.array(global_values.test_labels == global_values.EXPECTED_LABEL)
        return super(LocalLatentMode, self)._generate_contributions(mask=mask_label)

    def cluster_contributions(self, contributions: np.ndarray) -> tuple:
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


class GlobalLatentMode(Approach):

    def __init__(self, explainer, dimensionality_reduction_techniques):
        super(GlobalLatentMode, self).__init__(explainer, dimensionality_reduction_techniques)

    def generate_contributions(self):
        # Generate the contributions for the whole data
        return super(GlobalLatentMode, self)._generate_contributions()

    def cluster_contributions(self, contributions: np.ndarray) -> tuple:
        # Flatten the contributions and project them into the latent space
        contributions_flattened = contributions.reshape(contributions.shape[0], -1)
        projections = np.array([])
        for dim_red_tech in self.get_dimensionality_reduction_techniques():
            projections = dim_red_tech.fit_transform(contributions_flattened)
        # Cluster the filtered projections
        mask_label = np.array(global_values.test_labels == global_values.EXPECTED_LABEL)
        projections_filtered = projections[mask_label]
        clusters = CLUSTERING_TECHNIQUE().fit_predict(projections_filtered)
        # Compute the silhouette score for the clusters
        try:
            score = silhouette_score(projections_filtered, clusters)
        except ValueError:
            score = np.nan
        return clusters, projections_filtered, score


class OriginalMode(Approach):

    def __init__(self, explainer, dimensionality_reduction_techniques):
        super(OriginalMode, self).__init__(explainer, dimensionality_reduction_techniques)

    def generate_contributions(self):
        # Generate the contributions for the filtered data
        mask_label = np.array(global_values.generated_labels == global_values.EXPECTED_LABEL)
        return super(OriginalMode, self)._generate_contributions(mask=mask_label)

    def generate_contributions_by_data(self, data, labels):
        # Generate the contributions for the filtered data
        return super(OriginalMode, self)._generate_contributions_by_data(data, labels)

    @staticmethod
    def multiprocessing_metric(pair):
        return IMAGES_SIMILARITY_METRIC(pair[0], pair[1])

    def cluster_contributions(self, contributions: np.ndarray) -> tuple:
        # Compute the similarity matrix for the contributions
        similarity_matrix = compute_comparison_matrix(
            list(contributions),
            metric=IMAGES_SIMILARITY_METRIC,
            show_progress_bar=True,
            multi_process=False
        )
        # Cluster the contributions using the similarity matrix
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
        projections = TSNE(perplexity=40).fit_transform(contributions_flattened)

        return clusters, projections, score
