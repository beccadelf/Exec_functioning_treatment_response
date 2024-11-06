# -*- coding: utf-8 -*-
"""
This script implements a machine learning classifier (XXX) to predict the response
to an exposure therapy based on the "Fear of Spyders Questionnaire".

@author: Rebecca Delfendahl
"""

#%% Import libraries and functions

import os
import numpy as np
import pandas as pd
import argparse
import time
from functools import partial
import sklearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Ensure cwd is script's directory
script_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_wd)

# Import self-defined functions
from lib.Preprocessing_Classes import FeatureSelector
from lib.Pipeline import load_data, impute_data, z_scale_data, select_features
from lib.Models import fit_random_forest_classifier
from lib.ModelPerformance import calc_eval_metrics_classification, get_performance_metrics_across_folds, summarize_performance_metrics_across_iters
from lib.FeatureStats import summarize_features

#%% OLD: Set global variables 

OPTIONS = {}
OPTIONS['number_iterations'] = 5 # planned number: 100
OPTIONS['Analysis'] = "all_features" # choose between "all_features" and "clin_features"
OPTIONS['Regressor'] = 'random_forest_regressor' # choose between "random_forest_regressor" and "ridge_regressor"
PATH_INPUT_DATA = "Z:\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\Feature_Label_Dataframes"

#%% 

def set_options_and_paths():
    """ Set options and paths based on command-line or inline arguments depending on the use of command line or IDE.

    Returns:
    - args: An object containing parsed command-line arguments.
    - PATH_RESULTS: Path to save results.
    """

    def generate_and_create_results_path(args):
        model_name = f"{args.NAME_RESULTS_FOLDER}"
        PATH_RESULTS = os.path.join(args.PATH_RESULTS_BASE, model_name)
        os.makedirs(PATH_RESULTS, exist_ok=True)
        PATH_RESULTS_PLOTS = os.path.join(PATH_RESULTS, "plots")
        os.makedirs(PATH_RESULTS_PLOTS, exist_ok=True)
        PATHS = {
            "RESULT": PATH_RESULTS,
            "RESULT_PLOTS": PATH_RESULTS_PLOTS
        }

        return PATHS
    
    # Argparser
    parser = argparse.ArgumentParser(
        description='Script to predict treatment outcome')
    parser.add_argument('--PATH_INPUT_DATA', type=str,
                        help='Path to input data')
    parser.add_argument('--PATH_RESULTS_BASE', type=str,
                        help='Path to save results')
    parser.add_argument('--NAME_RESULTS_FOLDER', type=str,
                        help='Name result folder')
    parser.add_argument('--ANALYSIS', type=str,
                        help='Features to include, set all_features or clinical_features_only')
    parser.add_argument('--CLASSIFIER', type=str,
                        help='Classifier to use, set random_forest_classifier or svm')
    parser.add_argument('--OVERSAMPLING', type=str, default="Yes",
                        help='Should training and testset be oversampled to represent distribution in sample?')
    parser.add_argument('--NUMBER_REPETITIONS', type=int, default=100,
                        help='Number of repetitions of the cross-validation')


    args = parser.parse_args()
    
    try:
        PATHS = generate_and_create_results_path(args)
        print("Using arguments given via terminal")
    except:
        print("Using arguments given in the script")
        args = parser.parse_args([
            '--PATH_INPUT_DATA', "Z:\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\Feature_Label_Dataframes",
            '--PATH_RESULTS_BASE', script_wd,
            '--NAME_RESULTS_FOLDER', "Results_Classifier",
            '--ANALYSIS', "all_features",
            '--CLASSIFIER', 'random_forest_classifier',
            '--OVERSAMPLING', 'Yes',
            '--NUMBER_REPETITIONS', "5"
        ])
        PATHS = generate_and_create_results_path(args)
        
    return args, PATHS

#%% Procedure for one single iteration

# TEST
# num_iter = 0

def procedure_per_iter(num_iter, args):
    
    # Load data
    X_import_path = os.path.join(args.PATH_INPUT_DATA, args.ANALYSIS + ".csv")
    y_import_path = os.path.join(args.PATH_INPUT_DATA, "labels.csv")
    X, y, feature_names = load_data(X_import_path, y_import_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                        random_state = num_iter) # TODO: stratify = y rausnehmen?
    # By changing the random_state with each iteration, we always get a different split
    
    # Oversample XXX
    oversample = RandomOverSampler(sampling_strategy = 'minority')
    X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
    
    # Impute missing values
    X_train_imp, X_test_imp = impute_data(X_train_over, X_test)
    
    # Exclude features using FeatureSelector (no variance, correlation too high!)
    selector = FeatureSelector()
    selector.fit(X_train_imp)
    X_train_imp_clean = selector.transform(X_train_imp)
    X_test_imp_clean = selector.transform(X_test_imp)
    feature_names_clean = feature_names[selector.is_feat_excluded == 0]
    feature_names_excl = feature_names[selector.is_feat_excluded == 1]
    
    # X-scale data
    X_train_imp_clean_scaled, X_test_imp_clean_scaled = z_scale_data(X_train_imp_clean, X_test_imp_clean)
    
    # Select features
    X_train_imp_clean_scaled_sel, X_test_imp_clean_scaled_sel, features_selected = select_features(X_train_imp_clean_scaled, X_test_imp_clean_scaled, y_train, feature_names_clean)
    
    # Fit classifier
    if args.CLASSIFIER == "random_forest_classifier":
        clf, feature_importances = fit_random_forest_classifier(
            X_train_imp_clean_scaled_sel, y_train)
    elif args.CLASSIFIER == "svm":
        clf, feature_importances = # TODO: adjust for svm
        
    # Calculate model performance metrics
    y_pred_test = clf.predict(X_test_imp_clean_scaled_sel)
    ev_metrics = calc_eval_metrics_classification(
        y_true=y_test, y_pred=y_pred_test)
    
    # Save relevant information for each iteration in a dictionary
    results_single_iter = {
        "ev_metrics": ev_metrics,
        "sel_features_names": list(features_selected),
        "sel_features_imp": list(feature_importances),
        "excluded_feat": feature_names_excl
    }
    
    return results_single_iter

#%% Final run

if __name__ == '__main__':
    start_time = time.time()
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    
    args, PATHS = set_options_and_paths()
    
    # Run procedure per iterations
    outcomes = []
    runs_list = []
    for i in range (args.NUMBER_REPETITIONS):
        runs_list.append(i)
    procedure_per_iter_spec = partial(procedure_per_iter,
                                      args=args)
    # Run procedure n times, saving results of each iteration
    outcomes[:] = map(procedure_per_iter_spec, runs_list) # We are using map to enable parallel-processing
    
    # Summarize metric and feature results across iterations
    performance_metrics_across_iters = get_performance_metrics_across_folds(outcomes, key_metrics = "ev_metrics")
    performance_metrics_summarized = summarize_performance_metrics_across_iters(outcomes, key_metrics = "ev_metrics")
    features_summarized = summarize_features(outcomes=outcomes, key_feat_names="sel_features_names", key_feat_weights="sel_features_imp")
    
    # Save summaries as csv
    performance_metrics_across_iters.to_csv(os.path.join(
        PATHS["RESULT"], "performance_across_iters.txt"), sep="\t", na_rep="NA")
    performance_metrics_summarized.to_csv(os.path.join(
        PATHS["RESULT"], "performance_summary.txt"), sep="\t")
    features_summarized.to_csv(os.path.join(
        PATHS["RESULT"], "features_summary.txt"), sep="\t", na_rep="NA")
    
    elapsed_time = time.time() - start_time
    print('\nThe time for running was {}.'.format(elapsed_time))
    print('Results were saved at {}.'.format(PATHS["RESULT"]))