from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from xplique.attributions import  Lime, IntegratedGradients

from itertools import product
# from utils.text.IntegratedGradients import IntegratedGradients
# from utils.text.lime import Lime

__batch_size = 64

# Choose between 0: OriginalMode | 1: LocalLatentMode | 2: GlobalLatentMode
APPROACH = 0


DIMENSIONALITY_REDUCTION_TECHNIQUES = [[TSNE(perplexity=1)]]
# DIMENSIONALITY_REDUCTION_TECHNIQUES = [
#     [TSNE(perplexity=perplexity)]
#     for perplexity in list(range(1, 8, 2)) + [10, 20, 30, 40]
# ]

# DIMENSIONALITY_REDUCTION_TECHNIQUES = [
#     [TSNE(n_components=n_components, perplexity=perplexity, method='exact')]
#     for n_components, perplexity in  
#     product([2, 5, 10, 100], [1, 5, 10, 50])
# ]


CLUSTERING_TECHNIQUE = AffinityPropagation
ITERATIONS = 10

#MNIST
EXPLAINERS = [
    # DeconvNet,
    # lambda classifier: Occlusion(classifier, patch_size=10, patch_stride=10, batch_size=__batch_size),
    # Saliency,
    # GuidedBackprop,
    lambda classifier: Lime(classifier, nb_samples=100),
    # lambda classifier: GradCAMPP(classifier, batch_size=__batch_size, output_layer=-1),
    # GradCAM,
    lambda classifier: IntegratedGradients(classifier, steps=50, batch_size=__batch_size),
    # lambda classifier: KernelShap(classifier, nb_samples=100),
    # lambda classifier: SmoothGrad(classifier, nb_samples=100, noise=.3, batch_size=__batch_size),
    # GradCAMPP,
    # lambda classifier: Rise(classifier, nb_samples=1000, preservation_probability=.4, batch_size=__batch_size)
]


# IMDB
# EXPLAINERS = [
#         lambda classifier: IntegratedGradients(classifier),
#         lambda classifier: Lime(classifier)
# ]
