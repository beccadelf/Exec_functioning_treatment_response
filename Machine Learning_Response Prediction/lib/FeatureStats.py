# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:11:07 2024

@author: Charlotte Meinke, Kevin Hilbert
"""

#%% Import functions

import numpy as np
import pandas as pd
from collections import Counter

#%% Define functions

# 1. Function to calculate summary statistics (selection frequency, mean importance) of each feature

def summarize_features(outcomes, key_feat_names, key_feat_weights):
    """
    Summarize selected features and feature importances across iterations.

    Parameters:
    - outcomes (list): List with one entry per iteration of k-fold cross-validation. 
      Each entry is a dictionary with all information saved per iteration. (results_single_iter)
      This dictionary needs to contain selected features per iteration and their feature weight.
    - key_feat_names (str): Key in the results_single_iter dictionary containing the names of selected features.
    - key_feat_weights (str): Key in the results_single_iter dictionary containing the weights of selected features.

    Returns:
    pd.DataFrame: DataFrame summarizing, for each feature, the selection frequency and mean coefficient across iterations.
    The DataFrame is sorted by selection frequency and mean coefficient in descending order.
    """
    # Create np_array of features that were at least once part of the final model
    feat_names_all = []
    for itera in outcomes:
        feat_names = itera[key_feat_names]
        feat_names_all.append(feat_names)
    feat_names_all = np.concatenate(feat_names_all)
    unique_feat_names = np.unique(feat_names_all)
    
    # Count selection frequency based on feat_names_all
    feat_counts = Counter(feat_names_all)

    # Create empty df with feature names as index
    feat_all_data = []
    empty_df = pd.DataFrame(index=unique_feat_names)
    feat_all_data.append(empty_df)

    # Collect feature weights for all iterations
    for itera in outcomes:
        feature_names = itera[key_feat_names]
        feature_weights = itera[key_feat_weights]
        features_coef_df = pd.DataFrame(feature_weights, index=feature_names)
        feat_all_data.append(features_coef_df)
    # Concatenate the collected DataFrames into a single DataFrame
    feat_all_df = pd.concat(feat_all_data, axis=1, keys=[
                            f'itera_{i}' for i in range(len(outcomes))])

    # Calculate mean feature weights and selection frequencies across iterations
    mean = feat_all_df.mean(axis=1)
    sel_freq = pd.Series(feat_counts, index=unique_feat_names)

    # Save mean feature weights and selection frequenies in DataFrame
    feat_sum_df = pd.DataFrame({
        "selection frequency": sel_freq,
        "mean coefficient": mean},
        index=feat_all_df.index)

    # Sort DataFrame
    feat_sum_df.sort_values(
        by=["selection frequency", "mean coefficient"], key=abs, ascending=False, inplace=True)

    return feat_sum_df