from sys import displayhook
from utils.general import save_figure
from utils.plotter.distance_matrix import show_comparison_matrix
from config.config_general import CLUSTERS_SIMILARITY_METRIC
from clusim.clustering import Clustering
from steps.human_evaluation.helpers import preprocess_data
from IPython.display import display
import pandas as pd



def combined_distance_matrix():
    df = preprocess_data()

    # Remove the configurations with only one clusters
    df['num_clusters'] = df['clusters'].apply(len)
    plot_data = df
    plot_data = df[df['num_clusters'] > 1]
    distances_df, fig, ax = show_comparison_matrix(
        values=[Clustering().from_cluster_list(clusters) for clusters in plot_data['clusters']],
        approaches=[approach for approach in plot_data['approach']],
        metric=lambda lhs, rhs: 1 - CLUSTERS_SIMILARITY_METRIC(lhs, rhs),
        index=plot_data['approach'],
        show_progress_bar=True,
        remove_diagonal=False
    )
    display(df[['approach','num_clusters']])
    display(df[['approach','num_clusters']].groupby('approach').mean())
    fig.suptitle('Distance matrix for the low-level and high-level approaches')
    save_figure(fig, f'out/combined/distance_matrix')
