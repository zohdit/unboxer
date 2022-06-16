from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from xplique.attributions import GradCAM, DeconvNet, Occlusion, GuidedBackprop, Saliency, Lime, IntegratedGradients, \
    KernelShap, SmoothGrad, GradCAMPP, Rise

from utils.clusters.ClusteringMode import OriginalMode

__batch_size = 64

HEATMAPS_PROCESS_MODE = OriginalMode
EXPLAINERS = [
    DeconvNet,
    lambda classifier: Occlusion(classifier, patch_size=10, patch_stride=10, batch_size=__batch_size),
    Saliency,
    GuidedBackprop,
    lambda classifier: Lime(classifier, nb_samples=100),
    GradCAM,
    lambda classifier: IntegratedGradients(classifier, steps=50, batch_size=__batch_size),
    lambda classifier: KernelShap(classifier, nb_samples=100),
    lambda classifier: SmoothGrad(classifier, nb_samples=100, noise=.3, batch_size=__batch_size),
    GradCAMPP,
    lambda classifier: Rise(classifier, nb_samples=4000, batch_size=__batch_size)
]
# DIMENSIONALITY_REDUCTION_TECHNIQUES = [[TSNE(perplexity=40)]]
DIMENSIONALITY_REDUCTION_TECHNIQUES = [
    [TSNE(perplexity=perplexity)]
    for perplexity in list(range(1, 8, 2)) + [10, 20, 30, 40]
]
CLUSTERING_TECHNIQUE = AffinityPropagation
ITERATIONS = 3
