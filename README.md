# 🥡 How to run the `unboxer` 🥡

First, you should install the environment and set the configurations based on the case study (MNIST or IMDB) you want to run:

## 📲 [Install](README-INSTALLATION.md) 📲

## ⚙️ [Configure](README-CONFIGURATION.md) ⚙️

##  Generate inputs 

You should run the following command to generate the inputs for corresponding case study.

```commandline
python -m utls.generate_inputs
```

## 🥵 Generate the heatmaps 🥵

You should run the following command to generate the heatmaps.

```commandline
python -m steps.process_heatmaps
```

The tool will experiment with the different explainers, find the best configuration for the dimensionality reduction,
and export the data collected during the experiment.

## 🗺 Generate the featuremaps 🗺

You can run the following command to generate the featuremaps.

```commandline
python -m steps.process_featuremaps
```

The tool will generate the featuremaps, and export the data collected during the experiment.

## 📊 Export the insights 📊

You can run the following command to generate the insights about the data.

```commandline
python -m steps.insights.insights
```

**!!! IMPORTANT !!!**<br>
**Remember to generate the heatmaps and the featuremaps before running this command.**

The tool with prompt a menu with a set of options, and will guide you through the process.

## 🤔 Export the data for the human evaluation 🤔

You can run the following command to export the data for the human evaluation.

```commandline
python -m steps.human_evaluation.export_samples
```

**!!! IMPORTANT !!!**<br>
**Remember to generate the heatmaps and the featuremaps before running this command.**

The tool will generate samples for human study in out/human_evaluation.