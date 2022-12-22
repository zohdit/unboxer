from config.config_featuremaps import CASE_STUDY
import numpy as np
import pandas as pd
from itertools import combinations

from utils.stats import compare_distributions

DISTANCES_DATA_ORIG_MNIST = "logs/mnist/original/distance_data"
DISTANCES_DATA_LOC_MNIST =  "logs/mnist/local/distance_data"
DISTANCES_DATA_GLOB_MNIST = "logs/mnist/global/distance_data"

DISTANCES_DATA_ORIG_IMDB = "logs/imdb/original/distance_data"
DISTANCES_DATA_LOC_IMDB =  "logs/imdb/local/distance_data"
DISTANCES_DATA_GLOB_IMDB = "logs/imdb/global/distance_data"



def compute_stats(_case_study):
    if _case_study == "MNIST":
        plot_data = pd.read_pickle(DISTANCES_DATA_ORIG_MNIST)
        plot_data_loc = pd.read_pickle(DISTANCES_DATA_LOC_MNIST)
        plot_data_glob = pd.read_pickle(DISTANCES_DATA_GLOB_MNIST)
    elif _case_study == "IMDB":
        plot_data = pd.read_pickle(DISTANCES_DATA_ORIG_IMDB)
        plot_data_loc = pd.read_pickle(DISTANCES_DATA_LOC_IMDB)
        plot_data_glob = pd.read_pickle(DISTANCES_DATA_GLOB_IMDB)


    # each space separately
    i = 0
    for _plot_data in [plot_data, plot_data_loc, plot_data_glob]:   
        if i == 0:
            print("Original space comparison:") 
        elif i == 1:
            print("Local space comprison:")     
        else:
            print("Global space comprison:")   
        approaches=[approach for approach in list(_plot_data.columns)]
        approaches = set(approaches)
        for approach in approaches:
            if approach != "Lime" and approach != "IntegratedGradients":
                lhs = _plot_data.groupby("approach", axis=0)[approach].apply(list).loc["Lime"]
                rhs = _plot_data.groupby("approach", axis=0)[approach].apply(list).loc["IntegratedGradients"]
                p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
                print("Comparing: %s,%s.\n \t p-Value %s \n \t %f - %s " %(
                f"{approach}-Lime", f"{approach}-IntegratedGradients",
                p_value,
                eff_size, eff_size_str))
        
        approaches.remove("Lime")
        approaches.remove("IntegratedGradients")
        pairs = list(combinations(approaches, 2))

        for approach1, approach2 in pairs:
            lhs = _plot_data.groupby("approach", axis=0)[approach1].apply(list).loc["Lime"]
            rhs = _plot_data.groupby("approach", axis=0)[approach2].apply(list).loc["Lime"]
            p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
            print("Comparing: %s,%s.\n \t p-Value %s \n \t %f - %s " %(
            f"{approach1}-Lime", f"{approach2}-Lime",
            p_value,
            eff_size, eff_size_str))

            lhs = _plot_data.groupby("approach", axis=0)[approach1].apply(list).loc["IntegratedGradients"]
            rhs = _plot_data.groupby("approach", axis=0)[approach2].apply(list).loc["IntegratedGradients"]
            p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
            print("Comparing: %s,%s.\n \t p-Value %s \n \t %f - %s " %(
            f"{approach1}-IntegratedGradients", f"{approach2}-IntegratedGradients",
            p_value,
            eff_size, eff_size_str))

        print("\n")
        i+=1

    # combination of spaces
    # original & local
    print("\n")
    print("Original vs Local")
    approaches=[approach for approach in list(plot_data.columns)]
    approaches = set(approaches)
    for approach in approaches:
        if approach != "Lime" and approach != "IntegratedGradients":
            lhs = plot_data.groupby("approach", axis=0)[approach].apply(list).loc["Lime"]
            rhs = plot_data_loc.groupby("approach", axis=0)[approach].apply(list).loc["Lime"]
            p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
            print("Comparing: %s,%s.\n \t p-Value %s \n \t %f - %s " %(
            f"{approach}-Lime-original", f"{approach}-Lime-local",
            p_value,
            eff_size, eff_size_str))

            lhs = plot_data.groupby("approach", axis=0)[approach].apply(list).loc["IntegratedGradients"]
            rhs = plot_data_loc.groupby("approach", axis=0)[approach].apply(list).loc["IntegratedGradients"]
            p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
            print("Comparing: %s,%s.\n \t p-Value %s \n \t %f - %s " %(
            f"{approach}-IntegratedGradients-original", f"{approach}-IntegratedGradients-local",
            p_value,
            eff_size, eff_size_str))            

    # original & local
    print("\n")
    print("Original vs Global")
    approaches=[approach for approach in list(plot_data.columns)]
    approaches = set(approaches)
    for approach in approaches:
        if approach != "Lime" and approach != "IntegratedGradients":
            lhs = plot_data.groupby("approach", axis=0)[approach].apply(list).loc["Lime"]
            rhs = plot_data_glob.groupby("approach", axis=0)[approach].apply(list).loc["Lime"]
            p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
            print("Comparing: %s,%s.\n \t p-Value %s \n \t %f - %s " %(
            f"{approach}-Lime-original", f"{approach}-Lime-global",
            p_value,
            eff_size, eff_size_str))

            lhs = plot_data.groupby("approach", axis=0)[approach].apply(list).loc["IntegratedGradients"]
            rhs = plot_data_glob.groupby("approach", axis=0)[approach].apply(list).loc["IntegratedGradients"]
            p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
            print("Comparing: %s,%s.\n \t p-Value %s \n \t %f - %s " %(
            f"{approach}-IntegratedGradients-original", f"{approach}-IntegratedGradients-global",
            p_value,
            eff_size, eff_size_str))  

    # original & local
    print("\n")
    print("global vs Local")
    approaches=[approach for approach in list(plot_data.columns)]
    approaches = set(approaches)
    for approach in approaches:
        if approach != "Lime" and approach != "IntegratedGradients":
            lhs = plot_data_glob.groupby("approach", axis=0)[approach].apply(list).loc["Lime"]
            rhs = plot_data_loc.groupby("approach", axis=0)[approach].apply(list).loc["Lime"]
            p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
            print("Comparing: %s,%s.\n \t p-Value %s \n \t %f - %s " %(
            f"{approach}-Lime-global", f"{approach}-Lime-local",
            p_value,
            eff_size, eff_size_str))

            lhs = plot_data_glob.groupby("approach", axis=0)[approach].apply(list).loc["IntegratedGradients"]
            rhs = plot_data_loc.groupby("approach", axis=0)[approach].apply(list).loc["IntegratedGradients"]
            p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
            print("Comparing: %s,%s.\n \t p-Value %s \n \t %f - %s " %(
            f"{approach}-IntegratedGradients-global", f"{approach}-IntegratedGradients-local",
            p_value,
            eff_size, eff_size_str))  





def main():
    compute_stats(CASE_STUDY)

if __name__ == '__main__':
    main()

