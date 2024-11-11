# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:46:40 2024

@author: Acer
"""

#%% Import functions

from itertools import product
import subprocess
import os

#%% 

# Ensure cwd is script's directory
script_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_wd)

# Define the parameters
regressors = ["random_forest_regressor", "ridge_regressor"]
classifiers = ["random_forest_classifier", "svm"]
analysis = ["all_features", "clinical_features_only"]
oversampling = ["yes", "no"]

path_input_data = "Z:\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\2_Machine_learning\\Feature_Label_Dataframes"
path_results_base = "C:\\Users\\Acer\\Documents\\GitHub\\Exec_functioning_treatment_response\\Machine Learning Analysis\\Results"

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
        'NAME_RESULTS_FOLDER': "_".join(combo) + "_oversampling",
        'ANALYSIS': combo[0],
        'CLASSIFIER': combo[1],
        'OVERSAMPLING': combo[2], 
        'NUMBER_REPETITIONS': 5
    })


# Paths to the Python scripts
script_paths = {
    "regression": "Regressor_treatment_outcome.py",
    "classification": "Classifier_treatment_response.py"
}

#%% Run scripts sequentially

# Helper function to run the script sequentially with different argument sets
def run_script(script_path, argument_sets):
    """
    Execute Python script with multiple sets of arguments, running it separately 
    for each set.
    
    Arguments:
        script_path: str, path to the Python script to execute.
        argument_sets: list of dict, each containing pairs of argument names and values
                       to pass to the script.
    
    Returns:
        None. Outputs from the script execution are captured and printed if there is an error.
    
    """
    for arguments in argument_sets:
        command = ["python", script_path]
        
        # Add each key-value pair from the argument set to the command
        for key, value in arguments.items():
            command.extend([f"--{key}", str(value)])
            
        try:
            subprocess.check_output(command, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running script with arguments {arguments}: {e}")

# Run the scripts with generated argument sets
run_script(script_paths["regression"], argument_sets_regression)
run_script(script_paths["classification"], argument_sets_classification)