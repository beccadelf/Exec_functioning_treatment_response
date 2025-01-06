# -*- coding: utf-8 -*-
"""
This script implements a machine learning classifier (XXX) to predict the response
to an exposure therapy based on the "Fear of Spyders Questionnaire".

@author: Rebecca Delfendahl
"""

#%% Import libraries and functions

import os
import argparse
import time
from functools import partial
import sklearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# Ensure cwd is script's directory
script_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_wd)

# Import self-defined functions
from lib.Preprocessing_Classes import FeatureSelector
from lib.Pipeline import load_data, impute_data, z_scale_data, select_features_classification
from lib.Models import fit_random_forest_classifier, fit_svm_classifier
from lib.ModelPerformance import calc_eval_metrics_classification, get_performance_metrics_across_iters, summarize_performance_metrics_across_iters
from lib.FeatureStats import summarize_features

#%% 

def set_options_and_paths():
    """ Set options and paths based on command-line or inline arguments depending on the use of command line or IDE.

    Returns:
    - args: An object containing parsed command-line arguments.
    - PATH_RESULTS: Path to save results.
    """

    def generate_and_create_results_path(args):
        model_name = f"{args.NAME_RESULTS_FOLDER}_new4"
        path_results_base = args.PATH_INPUT_DATA.replace( "Feature_Label_Dataframes","Results")
        PATH_RESULTS = os.path.join(path_results_base, model_name)
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
                        help='Classifier to use, set random_forest_classifier or svm_classifier')
    parser.add_argument('--OVERSAMPLING', type=str, default="yes_simple",
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
            '--PATH_INPUT_DATA', "Y:\\PsyThera\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\2_Machine_learning\\Feature_Label_Dataframes\\RT_trimmed_RT_wrong_removed_outliers-removed",
            '--PATH_RESULTS_BASE', "Y:\\PsyThera\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\2_Machine_learning\\Results",
            '--ANALYSIS', "all_features",
            '--CLASSIFIER', 'random_forest_classifier',
            '--OVERSAMPLING', 'yes_smote',
            '--NUMBER_REPETITIONS', "100"
        ])
        args.NAME_RESULTS_FOLDER = f"{args.ANALYSIS}_{args.CLASSIFIER}_{args.OVERSAMPLING}" + "_oversampling"
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                        stratify=y,
                                                        random_state = num_iter)
    # By changing the random_state with each iteration, we always get a different split
    
    # Oversample XXX
    if args.OVERSAMPLING == "yes_simple":
        oversample = RandomOverSampler(sampling_strategy = 'minority')
        X_train_final, y_train_final = oversample.fit_resample(X_train, y_train)
    elif args.OVERSAMPLING == "yes_smote":
        sm = SMOTE(random_state=42)
        X_train_final, y_train_final = sm.fit_resample(X_train, y_train)
    elif args.OVERSAMPLING == "no":
        X_train_final, y_train_final = X_train, y_train
    
    # Impute missing values
    X_train_imp, X_test_imp = impute_data(X_train_final, X_test)
    
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
    X_train_imp_clean_scaled_sel, X_test_imp_clean_scaled_sel, features_selected = select_features_classification(X_train_imp_clean_scaled, X_test_imp_clean_scaled, y_train_final, feature_names_clean)
    
    # Fit classifier
    if args.CLASSIFIER == "random_forest_classifier":
        clf, feature_weights = fit_random_forest_classifier(
            X_train_imp_clean_scaled_sel, y_train_final)
    # elif args.CLASSIFIER == "random_forest_classifier_0.8":
    #     clf, feature_weights = fit_random_forest_classifier(
    #         X_train_imp_clean_scaled_sel, y_train_final, max_features = 0.8)
    # elif args.CLASSIFIER == "random_forest_classifier_0.9":
    #     clf, feature_weights = fit_random_forest_classifier(
    #         X_train_imp_clean_scaled_sel, y_train_final, max_features = 0.9)
    elif args.CLASSIFIER == "svm_classifier_C1":
        clf, feature_weights = fit_svm_classifier(
            X_train_imp_clean_scaled_sel, y_train_final, C=1)
    # elif args.CLASSIFIER == "svm_classifier_C0.01":
    #     clf, feature_weights = fit_svm_classifier(
    #         X_train_imp_clean_scaled_sel, y_train_final, C=1)
    # elif args.CLASSIFIER == "svm_classifier_C1":
    #     clf, feature_weights = fit_svm_classifier(
    #         X_train_imp_clean_scaled_sel, y_train_final, kernel = "rbf")
    
    # Calculate model performance metrics
    y_pred_test = clf.predict(X_test_imp_clean_scaled_sel)
    ev_metrics_test = calc_eval_metrics_classification(
        y_true=y_test, y_pred=y_pred_test)
    
    # Calculate training metrics
    y_pred_train = clf.predict(X_train_imp_clean_scaled_sel)
    ev_metrics_train = calc_eval_metrics_classification(
        y_true=y_train_final, y_pred=y_pred_train)
    
    # Save relevant information for each iteration in a dictionary
    results_single_iter = {
        "ev_metrics_test": ev_metrics_test,
        "ev_metrics_train": ev_metrics_train,
        "sel_features_names": list(features_selected),
        "sel_features_imp": list(feature_weights),
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
    ## Test metrics
    performance_metrics_across_iters = get_performance_metrics_across_iters(outcomes, key_metrics = "ev_metrics_test")
    performance_metrics_summarized = summarize_performance_metrics_across_iters(outcomes, key_metrics = "ev_metrics_test")
    features_summarized = summarize_features(outcomes=outcomes, key_feat_names="sel_features_names", key_feat_weights="sel_features_imp")
    ## Training metrics
    train_performance_summarized = summarize_performance_metrics_across_iters(outcomes, key_metrics = "ev_metrics_train")   
    
    # Save summaries as csv
    performance_metrics_across_iters.to_csv(os.path.join(
        PATHS["RESULT"], "performance_across_iters.txt"), sep="\t", na_rep="NA")
    performance_metrics_summarized.to_csv(os.path.join(
        PATHS["RESULT"], "performance_summary.txt"), sep="\t")
    features_summarized.to_csv(os.path.join(
        PATHS["RESULT"], "features_summary.txt"), sep="\t", na_rep="NA")
    train_performance_summarized.to_csv(os.path.join(
        PATHS["RESULT"], "training_performance_summary.txt"), sep="\t")
    
    elapsed_time = time.time() - start_time
    print('\nThe time for running was {}.'.format(elapsed_time))
    print('Results were saved at {}.'.format(PATHS["RESULT"]))