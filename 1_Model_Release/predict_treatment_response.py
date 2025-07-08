# -*- coding: utf-8 -*-
"""
This script implements a machine learning classifier to predict the response
to an exposure therapy based on the "Fear of Spyders Questionnaire".

@author: Rebecca Delfendahl, Charlotte Meinke
"""

#%% Import libraries and functions

import os
import sys
import argparse
import time
import sklearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import joblib

# Ensure cwd is script's directory
script_wd = os.path.dirname(os.path.abspath(__file__))
os.chdir("C:\\Users\\Acer\\Documents\\GitHub\\Exec_functioning_treatment_response\\Machine Learning_Response Prediction")

# Import self-defined functions
from lib.Pipeline import load_data, impute_data, z_scale_data
from lib.Models import fit_random_forest_classifier
from lib.ModelPerformance import calc_eval_metrics_classification

#%% 
def set_options_and_paths():
    """ Set options and paths based on command-line or inline arguments depending on the use of command line or IDE.

    Returns:
    - args: An object containing parsed command-line arguments.
    """
    
    # Argparser
    parser = argparse.ArgumentParser(
        description='Script to predict treatment outcome')
    parser.add_argument('--PATH_INPUT_DATA', type=str, required=True,
                        help='Path to input data')
    parser.add_argument('--OUTPUT_PATH', type=str, required=True,
                        help='Path where trained model and scalor should be saved')
    parser.add_argument('--MODE', required=True,
                        choices=["train_only", "inference_only"],
                        help="Select the operation mode: 'train_only' to train and save a model, 'inference_only' to use a pre-trained model.")
    parser.add_argument('--X_FILE', type=str, required=True,
                        help='Filename for the feature matrix (e.g., clinical_features_only.csv or simulated_data.csv)')
    parser.add_argument('--Y_FILE', type=str, required=True,
                        help='Filename for the label vector (e.g., labels.csv or simulated_labels.csv)')
    parser.add_argument('--OVERSAMPLING', type=str, default="yes_simple",
                        help='Should training and testset be oversampled to represent distribution in sample?')
    
    # Try to get command-line args
    if len(sys.argv) > 1:
        # Running with CLI args
        args = parser.parse_args()            
        print("Using arguments given via terminal")
    else:
        # Running interactively
        print("Using arguments given in the script")
        args = parser.parse_args([
            '--PATH_INPUT_DATA', "Y:\\PsyThera\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\Make_model_available\\Feature_Label_Dataframes\\response_FSQ", # "Y:\\PsyThera\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\Make_model_available", 
            '--OUTPUT_PATH', "Y:\\PsyThera\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\Make_model_available", # "Y:\\PsyThera\\Projekte_Meinke\\Old_projects\\Labrotation_Rebecca\\Make_model_available\\Test_model_evaluation",
            '--MODE', "train_only",
            '--X_FILE', "clinical_features_only.csv", # 'simulated_data.csv'
            '--Y_FILE', "labels.csv", # 'simulated_labels.csv'
            '--OVERSAMPLING', "no", # "yes_smote" # 'no'
        ])
        
    return args

#%% Final run

if __name__ == '__main__':
    start_time = time.time()
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    
    args = set_options_and_paths()
    
    # Warning
    if args.MODE == "train_only":
        print("⚠️ WARNING: 'train_only' mode requires access to the original data for model training.")
    
    # Load data
    X_import_path = os.path.join(args.PATH_INPUT_DATA, args.X_FILE)
    y_import_path = os.path.join(args.PATH_INPUT_DATA, args.Y_FILE)
    X, y, feature_names = load_data(X_import_path, y_import_path)
    
    # Oversampling
    if args.OVERSAMPLING == "yes_simple":
        oversample = RandomOverSampler(sampling_strategy = 'minority')
        X_final, y_final = oversample.fit_resample(X, y)
    elif args.OVERSAMPLING == "yes_smote":
        sm = SMOTE(random_state=42)
        X_final, y_final = sm.fit_resample(X, y)
    elif args.OVERSAMPLING == "no":
        X_final, y_final = X, y
    
    # Impute missing values
    X_imp = impute_data(X_final)
    
    if args.MODE == "train_only":
        print("Training mode: model is only trained and saved.")
        # X-scale data
        X_imp_scaled, scaler = z_scale_data(X_imp, return_scaler=True)
        
        # Fit classifier
        clf, feature_weights = fit_random_forest_classifier(
            X_imp_scaled, y_final)
        
        # Save scaler + model
        joblib.dump({'scaler': scaler, 'model': clf, 'feature_weights': feature_weights}, os.path.join(args.OUTPUT_PATH, "training_output.joblib"))
        
        # # Calculate training metrics
        # y_pred = clf.predict(X_imp_scaled)
        # y_prob = clf.predict_proba(X_imp_scaled)
        # ev_metrics_train = calc_eval_metrics_classification(
        #     y_true=y_final, y_pred=y_pred, y_prob=y_prob)  
    
    elif args.MODE == "inference_only":
        print("Inference mode: a pre-trained model is evaluated on new data.")
        # Load model components
        training_output = joblib.load(os.path.join(args.PATH_INPUT_DATA, "training_output.joblib"))
        scaler = training_output["scaler"]
        
        # Skip feature exclusion and selection, use scaler directly
        X_imp_scaled = scaler.transform(X_imp)
        
        # Define pretrained model
        clf = training_output["model"]
        feature_weights = training_output.get("feature_weights", None)
        
        # Calculate model performance metrics
        y_pred = clf.predict(X_imp_scaled)
        y_prob = clf.predict_proba(X_imp_scaled) 
        ev_metrics = calc_eval_metrics_classification(
            y_true=y_final, y_pred=y_pred, y_prob=y_prob)
    
        # Save ev_metrics
        ev_metrics.to_csv(os.path.join(args.OUTPUT_PATH, "model_performance_evaluation.txt"),
                          sep="\t", na_rep="NA")
    
    elapsed_time = time.time() - start_time
    print('\nThe time for running was {}.'.format(elapsed_time))
    print('Results were saved at {}.'.format(args.OUTPUT_PATH))