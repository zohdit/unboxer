from audioop import avg
import os
import scipy.stats as ss
from bisect import bisect_left
import itertools as it
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.power as pw

# https://gist.github.com/jacksonpradolima/f9b19d65b7f16603c837024d5f8c8a65
def VD_A(treatment, control):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param treatment: a numeric list
    :param control: another numeric list
    :returns the value estimate and the magnitude and power and number of runs to have power>0.8
    """
    m = len(treatment)
    n = len(control)

    # if m != n:
    #     raise ValueError("Data must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude#, power, nruns


def _log_raw_statistics(treatment, treatment_name, control, control_name):
    # Compute p : In statistics, the Mann–Whitney U test (also called the Mann–Whitney–Wilcoxon (MWW),
    # Wilcoxon rank-sum test, or Wilcoxon–Mann–Whitney test) is a nonparametric test of the null hypothesis that,
    # for randomly selected values X and Y from two populations, the probability of X being greater than Y is
    # equal to the probability of Y being greater than X.

    statistics, p_value = ss.mannwhitneyu(treatment, control)
    # Compute A12
    estimate, magnitude = VD_A(treatment, control)

    # Print them
    print("Comparing: %s,%s.\n \t p-Value %s - %s \n \t A12 %f - %s " %(
             treatment_name.replace("\n", " "), control_name.replace("\n", " "),
             statistics, p_value,
             estimate, magnitude))

def eff_size_label(eff_size):
    if np.abs(eff_size) < 0.2:
        return 'negligible'
    if np.abs(eff_size) < 0.5:
        return 'small'
    if np.abs(eff_size) < 0.8:
        return 'medium'
    return 'large'

def calculate_effect_size(treatment, treatment_name, control, control_name):

        # boxplot(treatment, control, labels=[treatment_name, control_name])


        (t, p) = stats.wilcoxon(treatment, control)
        eff_size = (np.mean(treatment) - np.mean(control)) / np.sqrt((np.std(treatment) ** 2 + np.std(control) ** 2) / 2.0)                   
        powe = pw.FTestAnovaPower().solve_power(effect_size=eff_size, nobs=len(treatment) + len(control), alpha=0.05)
        nruns = pw.FTestAnovaPower().solve_power(effect_size=eff_size, power=0.8, alpha=0.05)
        print(f"{treatment_name}, {control_name}: Cohen effect size = {eff_size} ({eff_size_label(eff_size)}); Wilcoxon p-value =  {p}; Statistical power = {powe}; number of runs = {nruns}\n")



if __name__ == "__main__":
    dataset_folder = "out/new"
    df = pd.DataFrame()
    tools = []
    accs = []
    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):

        # Consider only the files that match the pattern
        for npy_data_file in [os.path.join(subdir, f) for f in files if f.endswith(".npy")]:
            data = np.load(npy_data_file)

            if "featuremap" in npy_data_file:
                if "5x5" in npy_data_file:
                    tool = "Feature map 5x5"
                elif "10x10" in npy_data_file:
                    tool = "Feature map 10x10"
            if "lime" in npy_data_file:
                tool = "Lime"
            if "gradcampp" in npy_data_file:
                tool = "GradCAM++"
            if "random" in npy_data_file:
                tool = "Random"
            if "both" in npy_data_file:
                tool = "Combined"
            tools.append(tool)
            accs.append(data)
            print(tool)
            print(np.average(data))
        
    list_of_tuples = list(zip(tools, accs))
  

    df = pd.DataFrame(list_of_tuples, columns=['tool', 'data'])        

          
    for treatment_name, control_name in it.combinations(df["tool"].unique(), 2):
            treatment = list(df[df["tool"] == treatment_name]["data"])[0]
            control = list(df[df["tool"] == control_name]["data"])[0]

            # Compute the statistics
            # _log_raw_statistics(treatment, treatment_name, control, control_name)
            calculate_effect_size(treatment, treatment_name, control, control_name)
