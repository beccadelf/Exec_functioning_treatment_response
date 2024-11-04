# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:46:40 2024

@author: Acer
"""

#%% Import functions

from itertools import product
#import os
#%% 

# Define the parameters
regressors = ["random_forest_regressor", "ridge_regressor"]
classifiers = ["random_forest", "svm"]
analysis = ["All_Features", "Clin_Features"]
oversampling = ["Yes", "No"]

path_input_data = "Z:\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\Feature_Label_Dataframes"
path_results_base = "C:\\Users\\Acer\\Documents\\GitHub\\Exec_functioning_treatment_response\\Machine Learning Analysis"

# generate argument_sets without preprocessing
def generate_argument_sets(analysis, regressors, classifiers, oversampling):
    # Regression
    argument_sets_regression = []
    all_combinations_regression = product(regressors, analysis)
    for combo in all_combinations_regression:
        argument_sets_regression.append({
            'PATH_INPUT_DATA': path_input_data,
            'PATH_RESULTS_BASE': path_results_base,
            'NAME_RESULTS_FOLDER': "_".join(combo),
            'REGRESSOR': combo[0],
            'ANALYSIS': combo[1],
            'NUMBER_REPETITIONS': 5
        })
    # Classification
    argument_sets_classification = []
    all_combinations_classification = product(classifiers, analysis, oversampling)
    for combo in all_combinations_classification:
        argument_sets_classification.append({
            'PATH_INPUT_DATA': path_input_data,
            'PATH_RESULTS_BASE': path_results_base,
            'NAME_RESULTS_FOLDER': "_".join(combo),
            'CLASSIFIER': combo[0],
            'ANALYSIS': combo[1],
            'OVERSAMPLING': combo[2],
            'NUMBER_REPETITIONS': 5
        })
    return argument_sets_regression, argument_sets_classification

# Paths to the Python scripts
script_path_regr = "Regressor_treatment_outcome.py"
script_path_class = "RaFo_manuell_cv_struc.py"