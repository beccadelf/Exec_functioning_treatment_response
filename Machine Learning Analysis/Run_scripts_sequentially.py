# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:46:40 2024

@author: Acer
"""

#%% Import functions

from itertools import product
import subprocess
#import os
#%% 

# Define the parameters
regressors = ["random_forest_regressor", "ridge_regressor"]
classifiers = ["random_forest_classifier", "svm"]
analysis = ["All_Features", "Clin_Features"]
oversampling = ["Yes", "No"]

path_input_data = "Z:\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\Feature_Label_Dataframes"
path_results_base = "C:\\Users\\Acer\\Documents\\GitHub\\Exec_functioning_treatment_response\\Machine Learning Analysis"

# Generate argument_sets
# Regression
argument_sets_regression = []
all_combinations_regression = product(analysis, regressors)
for combo in all_combinations_regression:
    argument_sets_regression.append({
        'PATH_INPUT_DATA': path_input_data,
        'PATH_RESULTS_BASE': path_results_base,
        'NAME_RESULTS_FOLDER': "_".join(combo),
        'ANALYSIS': combo[0],
        'REGRESSOR': combo[1],
        'NUMBER_REPETITIONS': 5
    })
# Classification
argument_sets_classification = []
all_combinations_classification = product(analysis, classifiers, oversampling)
for combo in all_combinations_classification:
    argument_sets_classification.append({
        'PATH_INPUT_DATA': path_input_data,
        'PATH_RESULTS_BASE': path_results_base,
        'NAME_RESULTS_FOLDER': "_".join(combo), # TODO: somehow add oversampling
        'ANALYSIS': combo[0],
        'CLASSIFIER': combo[1],
        'OVERSAMPLING': combo[2], 
        'NUMBER_REPETITIONS': 5
    })


# Paths to the Python scripts
script_paths = {
    "regression": "Regressor_treatment_outcome.py",
    "classification": "RaFo_manuell_cv_struc.py"
}

# Helper function to run the script sequentially with different argument sets
def run_script(script_path, argument_sets):
    for arguments in argument_sets:
        command = ["python", script_paths] + [
            f"--{key}" for key in arguments.keys() for value in arguments[key] # TODO: adjust
            ]
        try:
            subprocess.check_output(command, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running script with arguments {arguments}: {e}")

# Run the scripts with generated argument sets
run_script(script_paths["regression"], argument_sets_regression)
run_script(script_paths["classification"], argument_sets_classification)