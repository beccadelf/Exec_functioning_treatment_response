# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:46:40 2024

@author: Acer
"""

#%% Import functions

from itertools import product
import os
#%% 

# Define the parameters
regressors = ["random_forest_regressor", "ridge_regressor"]
classifiers = ["random_forest", "svm"]
analysis = ["All_Features", "Clin_Features"]
oversampling = ["Yes", "No"]

# generate argument_sets without preprocessing
def generate_argument_sets(analysis, regressors, classifiers, oversampling):
    # Regression
    argument_sets_regression = []
    all_combinations_regression = product(regressors, analysis)
    for combo in all_combinations_regression:
        input_data_parentfolder = f"predict_treatmentoutcome_{combo[0]}"
        final_path = os.path.join(input_data_basis_path, input_data_parentfolder, f"{combo[1]}")
        argument_sets_regression.append({
            'PATH_INPUT_DATA': final_path,
            'PATH_RESULTS_BASE': path_results_base,
            'NAME_RESULTS_FOLDER': "_".join(combo),
            'REGRESSOR': combo[0],
            'ANALYSIS': combo[1],
            'NUMBER_REPETITIONS': 5
        })
    # Classification
    argument_sets_classification = []
    all_combinations_classification = product(classifiers, analysis, oversampling)
    return argument_sets_regression, argument_sets_classification
