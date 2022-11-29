import math
import multiprocessing
import os
from itertools import combinations

import numpy as np
from cliffs_delta import cliffs_delta
from pingouin import compute_effsize
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from tqdm import tqdm


def compare_distributions(lhs: list, rhs: list) -> tuple:
    """
    Compare two distributions by running the appropriate statistical test
    :param lhs: The first distribution
    :param rhs: The second distribution
    :return: [are different, p-value for the difference, effect size of the difference, magnitude of the difference]
    """
    # Check if the two distributions are normal
    _, p_val_lhs = shapiro(lhs)
    _, p_val_rhs = shapiro(rhs)
    if p_val_lhs < .05 and p_val_rhs < .05:
        # Non-normal distributions
        statistic, p_value = mannwhitneyu(lhs, rhs)
        eff_size, eff_size_str = cliffs_delta(lhs, rhs)
    else:
        # Normal distributions
        statistic, p_value = ttest_ind(lhs, rhs)
        eff_size = compute_effsize(lhs, rhs, eftype='cohen')
        if eff_size <= .3:
            eff_size_str = 'small'
        elif eff_size <= .5:
            eff_size_str = 'medium'
        else:
            eff_size_str = 'large'

    return p_value < .05, p_value, eff_size, eff_size_str


def get_effect_size(lhs: list, rhs: list) -> tuple:
    """
    Compare two distributions
    :return: The effect size if the difference is relevant, None otherwise
    """
    is_relevant, p_value, eff_size, eff_size_str = compare_distributions(lhs, rhs)
    eff_size_str_to_val = {
        'small': 1,
        'medium': 2,
        'large': 3
    }
    return abs(eff_size), eff_size_str_to_val[eff_size_str] if is_relevant else None


def weight_value(value: float, weight: float, max_weight: float) -> float:
    """
    Compute the weighted value with a weight between 0, 1
    :param value: The value
    :param weight: The weight
    :param max_weight: The maximum value for the weights
    :return: The weighted value
    """
    return value * math.log((math.e - 1) * weight / max_weight + 1)


def compute_comparison_matrix(        
        values: list,
        approaches: list,
        metric: callable,
        args,
        show_progress_bar: bool = False,
        multi_process: bool = False):

    matrix_ll_hl = compute_comparison_matrix_ll_hl(
        values=values,
        approaches=approaches,
        metric=metric,
        args=None,
        show_progress_bar=show_progress_bar,
        multi_process=multi_process
    )
    matrix_hl_ll = compute_comparison_matrix_hl_ll(
        values=values,
        approaches=approaches,
        metric=metric,
        args=None,
        show_progress_bar=show_progress_bar,
        multi_process=multi_process
    )
    matrix = np.min( np.array([matrix_ll_hl, matrix_hl_ll]), axis=0 )
    return matrix


def compute_comparison_matrix_ll_hl(
        values: list,
        approaches: list,
        metric: callable,
        args,
        show_progress_bar: bool = False,
        multi_process: bool = False
):
    # Compute all the combinations of values

    if len(approaches) > 0: 
        pairs = []
        app_pairs = list(combinations(range(len(approaches)), 2))
        # if we use custom similarity the high level explanation
        # should be passed only as second argument (rhs)
        for idx in range(len(app_pairs)):
            # if the approach is high level
            if "moves" in approaches[app_pairs[idx][0]] or "bitmaps" in approaches[app_pairs[idx][0]] or "orientation" in approaches[app_pairs[idx][0]] or "poscount" in approaches[app_pairs[idx][0]] or "negcount" in approaches[app_pairs[idx][0]] or "verbcount" in approaches[app_pairs[idx][0]]:
                app_pairs[idx] = (app_pairs[idx][1], app_pairs[idx][0])
    
        for idx1, idx2 in app_pairs:
            pairs.append((values[idx1], values[idx2]))
    else:
        pairs = list(combinations(values, 2))

    

    # Show the progress bar
    if show_progress_bar:
        pairs = tqdm(pairs, desc='Computing the comparison matrix', total=len(pairs), leave=False)
    # Create the pool of processes and use it to compute the distances
    if multi_process:
        pool = multiprocessing.Pool(int(os.cpu_count() / 2))
        distances = pool.map(metric, pairs)
    else:
        if args == None:
            distances = [metric(lhs, rhs) for lhs, rhs in pairs]
        else:
            distances = [metric(lhs, rhs, args) for lhs, rhs in pairs]
        
    # Initialize the distance matrix to 0
    matrix = np.zeros(shape=(len(values), len(values)))
    # Set the values of the upper triangular matrix to the distances
    matrix[np.triu_indices_from(matrix, 1)] = distances
    # Complete the matrix by transposing it
    matrix = matrix + np.transpose(matrix)
    return matrix

    
def compute_comparison_matrix_hl_ll(
        values: list,
        approaches: list,
        metric: callable,
        args,
        show_progress_bar: bool = False,
        multi_process: bool = False
):
    # Compute all the combinations of values

    if len(approaches) > 0: 
        pairs = []
        app_pairs = list(combinations(range(len(approaches)), 2))
        # if we use custom similarity the high level explanation
        # should be passed only as second argument (rhs)
        for idx in range(len(app_pairs)):
            # if the approach is high level
            if "moves" in approaches[app_pairs[idx][1]] or "bitmaps" in approaches[app_pairs[idx][1]] or "orientation" in approaches[app_pairs[idx][1]] or "poscount" in approaches[app_pairs[idx][1]] or "negcount" in approaches[app_pairs[idx][1]] or "verbcount" in approaches[app_pairs[idx][1]]:
                app_pairs[idx] = (app_pairs[idx][1], app_pairs[idx][0])
    
        for idx1, idx2 in app_pairs:
            pairs.append((values[idx1], values[idx2]))
    else:
        pairs = list(combinations(values, 2))

    
    # Show the progress bar
    if show_progress_bar:
        pairs = tqdm(pairs, desc='Computing the comparison matrix', total=len(pairs), leave=False)
    # Create the pool of processes and use it to compute the distances
    if multi_process:
        pool = multiprocessing.Pool(int(os.cpu_count() / 2))
        distances = pool.map(metric, pairs)
    else:
        if args == None:
            distances = [metric(lhs, rhs) for lhs, rhs in pairs]
        else:
            distances = [metric(lhs, rhs, args) for lhs, rhs in pairs]
        
    # Initialize the distance matrix to 0
    matrix = np.zeros(shape=(len(values), len(values)))
    # Set the values of the upper triangular matrix to the distances
    matrix[np.triu_indices_from(matrix, 1)] = distances
    # Complete the matrix by transposing it
    matrix = matrix + np.transpose(matrix)
    return matrix