# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:02:24 2024

@author: Acer
"""

#%% Import functions

import numpy as np
import pandas as pd
import sklearn.metrics as metrics

#%% Define functions

# 1. Functions to evaluate the model performance (i.e. calculate metrics)

def calc_eval_metrics_regression(y_true, y_pred):
    """
    Calculate evaluation metrics for regression.
    
    Arguments:
        y_true: array-like of shape (n_samples,), true target values (i.e. y_test).
        y_pred: array-like of shape (n_samples,), predicted target values.
    
    Returns:
        pd.DataFrame, containing the following regression metrics:
            - r2: R-squared, the coefficient of determination.
            - MSE: Mean squared error.
            - RMSE: Root mean squared error.
            - MAE: Mean absolute error.
            - Correlation: Correlation coefficient between true and predicted values.
    """
    # mean squared error
    MSE = metrics.mean_squared_error(y_true, y_pred)
    # Root mean squared error
    RMSE = metrics.mean_squared_error(y_true, y_pred, squared = False)
    # Mean absolute error
    MAE = metrics.mean_absolute_error(y_true, y_pred)
    # Correlation between predicted and true values
    corr = np.corrcoef(y_true, y_pred)[0,1]
    # R-Squared
    Rsquared = metrics.r2_score(y_true,y_pred)
    # Save metrics in a dataframe
    ev_metrics = pd.DataFrame({"r2": Rsquared, "MSE": MSE, "RMSE": RMSE,
                               "MAE": MAE, "Correlation": corr}, index = [0])
    return ev_metrics

def calc_eval_metrics_classification(y_true, y_pred):
    """ 
    Calculate evaluation metrics for classification.

    Arguments:
        y_true: array-like of shape (n_samples,), true target labels (i.e. y_test).
        y_pred: array-like of shape (n_samples,), predicted target labels.

    Returns:
        pd.DataFrame, containing the following classification metrics:
            - Accuracy: Proportion of correct predictions.
            - Balanced accuracy: Average of recall obtained on each class.
            - Specificity: True negative rate.
            - Sensitivity: True positive rate.
            - F1 Score: Harmonic mean of precision and recall.
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    specificity = metrics.recall_score(y_true, y_pred, pos_label = 0)
    sensitivity = metrics.recall_score(y_true, y_pred, pos_label = 1)
    f1_score = metrics.f1_score(y_true, y_pred)
    # Save metrics in dataframe
    ev_metrics = pd.DataFrame({"accuracy": accuracy, "balanced_accuracy": bal_acc,
                               "sensitivity": sensitivity, "specificity": specificity, 
                               "f1_score": f1_score}, index = [0])    
    return ev_metrics

def get_performance_metrics_across_iters(outcomes, key_metrics):
    """
    Compile evaluation metrics from multiple iterations into a single DataFrame.

    Arguments:
        - outcomes: list of dicts, each containing the results of a single iteration.
        - key_metrics: str, key to the metric-dataframe of each iteration.
    
    Returns:
        pd.DataFrame, with evaluation metrics per fold.
    """    
    # Turn all ev_metrics dictionaries into dataframes and concatenate them
    dataframes = [inner_df[key_metrics] for inner_df in outcomes]
    performance_metrics_across_iters_df = pd.concat(dataframes)
    
    return performance_metrics_across_iters_df

# 2. Function to summarize model performance

def summarize_performance_metrics_across_iters(outcomes, key_metrics):
    """
    Summarize model performance metrics across iterations.

    Parameters:
        - outcomes (list): List with one entry per iteration
          Each entry is a dictionary with all information saved per iteration (results_single_iter).
        - key_modelperformance_metrics (str): Key in the results_single_iter dictionary 
          containing the model performance metrics.

    Returns:
        pd.DataFrame, with summary statistics for model performance metrics.

    """
    sum_stats_performance_metrics = pd.DataFrame()
    count_iters = len(outcomes)

    for metric in outcomes[0][key_metrics]:
        
        # Initialize an empty list to store values for the current variable
        list_values = []
        
        # Concatenate values of all iterations to the list
        for itera in outcomes:
            list_values.append(itera[key_metrics][metric])
        
        list_values = pd.concat(list_values, axis=0)
     
        # Calculate summary statistics
        if count_iters > 1:
            min_val = min(list_values)
            max_val = max(list_values)
            mean_val = np.mean(list_values)
            std_val = np.std(list_values)
        elif count_iters == 1:
            min_val = "NA"
            max_val = "NA"
            mean_val = list_values[0]
            std_val = "NA"
        # Add summary statistics to the intialized DataFrame
        sum_stats_performance_metrics["Min_" + metric] = [min_val]
        sum_stats_performance_metrics["Max_" + metric] = [max_val]
        sum_stats_performance_metrics["Mean_" + metric] = [mean_val]
        sum_stats_performance_metrics["Std_" + metric] = [std_val]

    return sum_stats_performance_metrics