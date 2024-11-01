# -*- coding: utf-8 -*-
"""
This script implements a machine learning regressor (XXX) to predict the reduction 
in the "Fear of Spyders Questionnaire" in percent.

@author: Rebecca Delfendahl
"""

#%% Import libraries and functions

import os
import numpy as np
import pandas as pd
import time
import sklearn
from sklearn.model_selection import train_test_split

# Ensure cwd is script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import ... functions
from Preprocessing_Classes import FeatureSelector
from Pipeline_Functions import load_data, impute_data, z_scale_data, select_features, fit_random_forest_regressor, fit_ridge_regressor
from ModelPerformance_Functions import calc_eval_metrics_regression, get_performance_metrics_across_folds, summarize_performance_metrics_across_iters
from FeatureStats import summarize_features

#%% Set global variables 
# TODO: adapt for command-line arguments

OPTIONS = {}
OPTIONS['number_iterations'] = 10 # planned number: 100
OPTIONS['Analysis'] = "all_features" # choose between "all_features" and "clin_features"
OPTIONS['Regressor'] = 'random_forest_regressor' # choose between "random_forest_regressor" and "ridge_regressor"
PATH_INPUT_DATA = "Z:\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\Feature_Label_Dataframes"

#%% Define functions

# Procedure for one single iteration
# TEST
# num_iter = 0

def procedure_per_iter(num_iter):
    X, y, feature_names = load_data(X_path, y_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                        random_state = num_iter) # TODO: stratify = y rausnehmen?
    # By changing the random_state with each iteration, we always get a different split
    
    # TODO: instead of oversampling, XXX outcome distribution to check whether 
    # certain ranges are underrepresented
    
    X_train_imp, X_test_imp = impute_data(X_train, X_test)
    
    # Exclude features using FeatureSelector (no variance, correlation too high!)
    selector = FeatureSelector()
    selector.fit(X_train_imp)
    X_train_imp_clean = selector.transform(X_train_imp)
    X_test_imp_clean = selector.transform(X_test_imp)
    feature_names_clean = feature_names[selector.is_feat_excluded == 0]
    feature_names_excl = feature_names[selector.is_feat_excluded == 1]
    
    X_train_imp_clean_scaled, X_test_imp_clean_scaled = z_scale_data(X_train_imp_clean, X_test_imp_clean)
    
    X_train_imp_clean_scaled_sel, X_test_imp_clean_scaled_sel, features_selected = select_features(X_train_imp_clean_scaled, X_test_imp_clean_scaled, y_train, feature_names_clean)
    
    # Fit classifier
    if OPTIONS["Regressor"] == "random_forest_regressor":
        clf, feature_weights = fit_random_forest_regressor(
            X_train_imp_clean_scaled_sel, y_train)
    elif OPTIONS["Regressor"] == "ridge_regressor":
        clf, feature_weights = fit_ridge_regressor(
            X_train_imp_clean_scaled_sel, y_train)
        
    # Calculate model performance metrics
    y_pred_test = clf.predict(X_test_imp_clean_scaled_sel)
    ev_metrics = calc_eval_metrics_regression(
        y_true=y_test, y_pred=y_pred_test)
    
    # Save relevant information for each iteration in a dictionary
    results_single_iter = {
        "ev_metrics": ev_metrics,
        "sel_features_names": list(features_selected),
        "sel_features_coef": list(feature_weights),
        "excluded_feat": feature_names_excl
    }
    
    return results_single_iter

#%% Final run

if __name__ == '__main__':
    start_time = time.time()
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    
    if OPTIONS['Analysis'] == "all_features":
        X_path = os.path.join(PATH_INPUT_DATA,"Features.csv")
        RESULTS_PATH = "C:\\Users\\Acer\\Documents\\Studentischer Hilfsjob\\FOR5187 Precision Psychotherapy\\TEST_ML_Results"
    elif OPTIONS['Analysis'] == "clin_features":
        X_path = os.path.join(PATH_INPUT_DATA,"Clinical_Features.csv")
        RESULTS_PATH = "C:\\Users\\Acer\\Documents\\Studentischer Hilfsjob\\FOR5187 Precision Psychotherapy\\TEST_ML_Results"    
    y_path = os.path.join(PATH_INPUT_DATA,"Outcome.csv")
    
    runs_list = []
    outcomes = []
    for i in range (OPTIONS['number_iterations']):
        runs_list.append(i)
    # Run procedure n times, saving results of each iteration
    outcomes[:] = map(procedure_per_iter, runs_list) # We are using map to enable parallel-processing
    
    performance_metrics_across_iters = get_performance_metrics_across_folds(outcomes, key_metrics = "ev_metrics")
    performance_metrics_summarized = summarize_performance_metrics_across_iters(outcomes, key_metrics = "ev_metrics")
    features_summarized = summarize_features(outcomes=outcomes, key_feat_names="sel_features_names", key_feat_weights="sel_features_coef")
    
    # Save summaries as csv
    performance_metrics_across_iters.to_csv(os.path.join(
        RESULTS_PATH, "performance_across_iters.txt"), sep="\t", na_rep="NA")
    performance_metrics_summarized.to_csv(os.path.join(
        RESULTS_PATH, "performance_summary.txt"), sep="\t")
    features_summarized.to_csv(os.path.join(
        RESULTS_PATH, "features_summary.txt"), sep="\t", na_rep="NA")
    
    elapsed_time = time.time() - start_time
    print('\nThe time for running was {}.'.format(elapsed_time))
    print('Results were saved at {}.'.format(RESULTS_PATH))