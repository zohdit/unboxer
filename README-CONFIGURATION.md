## ⚙️ Configure ⚙️

The tool can be configured using the files:

<dl>
    <code>config/config_dirs.py</code>
    <dd>
This file defines the location of the inputs and the outputs of the tool.
</dd>
    <code>config/config_data.py</code>
    <dd>
Ths file defines configurations for some constants used by the tool.

<code>EXPECTED_LABEL</code>: The label to use to filter the data for the experiment.

<code>NUM_INPUTS</code>: Number of inputs from data set to be consider for comparison.

<code>INPUT_MAXLEN</code>: The size of the texts with paddings in the dataset.

<code>VOCAB_SIZE</code>: The size of dictionary used for text tokenization.

<code>USE_RGB</code>: If use RGB images or grey scaled.

</dd>
    <code>config/config_general.py</code>
    <dd>
Ths file defines configurations common to both the heatmaps and the featuremaps.

<code>CLUSTERS_SORT_METRIC</code>: The preference for the clusters when sampling them to show some of their images. If
no sorting is provided, the tool draw a random sample.

<code>CLUSTER_SIMILARITY_METRIC</code>: The similarity metric to use when comparing different clusters.
</dd>
    <code>config/config_heatmaps.py</code>
    <dd>
This fle defines the configuration for the heatmaps.

<code>APPROACH</code>: The processing mode to use when generating the
heatmaps [`Original`, `LocalLatentMode`, `GlobalLatentMode`].

<code>EXPLAINERS</code>: The list of explainers to use when generating the contributions.

<code>DIMENSIONALITY_REDUCTION_TECHNIQUES</code>: The dimensionality reduction techniques to use to project the
contributions in the two-dimensional latent space. The tool will experiment with the different techniques and choose the
best configuration according to the silhouette score of the corresponding clusters.

<code>CLUSTERING_TECHNIQUE</code>: The clustering technique to use when grouping the contributions.

<code>ITERATIONS</code>: The number of iterations to use when running the experiment.
</dd>
<code>config/config_featuremaps.py</code>
<dd>
This fle defines the configuration for the featuremaps.

<code>CASE_STUDY</code>: MNIST or IMDB.

<code>NUM_CELLS</code>: The size of the featuremaps.

<code>BITMAP_THRESHOLD</code>: The threshold for luminosity metric computation.

<code>ORIENTATION_THRESHOLD</code>: The threshold for orientation metric computation.

<code>FEATUREMAPS_CLUSTERS_MODE</code>: The clustering technique to use on the featuremaps [`ORIGINAL`, `REDUCED`,  `CLUSTERED`].


<code>MAP_DIMENSIONS</code>: The list of dimensions to be considered for generating feature maps.

</dd>
    <code>config/config_outputs.py</code>
    <dd>
Ths file defines configurations for visualisations.

<code>IMG_SIZE</code>: The size of image of an input.

<code>NUM_SAMPLES</code>: The number of samples for human study.

<code>ORIENTATION_THRESHOLD</code>: the threshold for orientation metric computation.


</dd>
</dl>
