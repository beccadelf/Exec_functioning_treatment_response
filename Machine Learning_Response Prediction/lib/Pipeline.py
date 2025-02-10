# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:36:56 2024

@author: Rebecca Delfendahl, Charlotte Meinke
"""

#%% Import functions

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer # must be imported to use IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, LogisticRegression
from sklearn.feature_selection import SelectFromModel

from lib.Preprocessing_Classes import ZScalerDimVars
#%% Define functions

def load_data(X_path, y_path): 
    """
    Load feature and outcome data, convert them to arrays and get feature names.

    Arguments:
        - X_path: str, file path to the CSV file containing feature data.
        - y_path: str, file path to the CSV file containing outcome data.

    Returns:
        - X: numpy array (n_samples, n_features), containing feature data.
        - y: numpy array (n_samples,), containing outcome data.
        - feature_names: Index or array (n_features,), containing names of the features.
    """    
    X_df = pd.read_csv(X_path) # features
    y = pd.read_csv(y_path) # outcome
    # Get feature_names 
    feature_names = X_df.columns
    # Turn into numpy arrays for higher speed
    X = X_df.to_numpy()
    y = np.ravel(y) 
    
    return X, y, feature_names


def impute_data(X_train, X_test):
    """
    Impute missing data using different strategies for binary and continuous variables.

    Arguments:
        - X_train: array-like or DataFrame of shape (n_samples_train, n_features), feature set for training.
        - X_test: array-like or DataFrame of shape (n_samples_test, n_features), feature set for testing.

    Returns:
        - X_train_imputed: numpy array (n_samples_train, n_features), imputed training data.
        - X_test_imputed: numpy array (n_samples_test, n_features), imputed testing data.

    Imputation Process:
        - Binary variables: Missing values are imputed using the mode (most frequent value).
          Assumes missing values in binary variables are marked with `77777`.
        - Continuous variables: Missing values are imputed using Bayesian Ridge Regression 
          with multiple iterations, assuming missing values are marked with `99999`.
    """
    # Impute binary variables by using the mode
    imp_mode = SimpleImputer(missing_values=77777, strategy='most_frequent')
    imp_mode.fit(X_train)
    X_train_imputed = imp_mode.transform(X_train)
    X_test_imputed = imp_mode.transform(X_test)
    # Impute dimensional variabels by using Bayesian Ridge Regression
    imp_mice = IterativeImputer(
        estimator=BayesianRidge(),
        missing_values=99999,
        sample_posterior=True, 
        max_iter=10, 
        initial_strategy="mean", 
        random_state=0)
    imp_mice.fit(X_train_imputed)
    X_train_imputed = imp_mice.transform(X_train_imputed)
    X_test_imputed = imp_mice.transform(X_test_imputed)
    
    return X_train_imputed, X_test_imputed


def z_scale_data(X_train, X_test):
    """
    Apply Z-score scaling to non-binary data.

    Arguments:
        X_train: array or DataFrame (n_samples_train, n_features), feature set for training.
        X_test: array or DataFrame (n_samples_test, n_features), feature set for testing.

    Returns:
        - X_train_scaled: numpy array (n_samples_train, n_features), Z-score scaled training data.
        - X_test_scaled: numpy array (n_samples_test, n_features), Z-score scaled testing data.
    """
    # z-scale only non-binary data!
    scaler = ZScalerDimVars()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


def select_features_classification(X_train, X_test, y_train, feature_names):
    """
    Perform feature selection using ElasticNet regularization, retaining 
    features based on a threshold applied to model coefficients.

    Arguments:
        - X_train: array or DataFrame (n_samples_train, n_features), feature set for training.
        - X_test: array or DataFrame (n_samples_test, n_features), feature set for testing.
        - y_train: array (n_samples_train,), target values for training.
        - feature_names: array (n_features,), names of all features initially provided.

    Returns:
        - X_train_selected: numpy array (n_samples_train, n_selected_features), 
                            training data with selected features.
        - X_test_selected: numpy array (n_samples_test, n_selected_features), 
                           testing data with selected features.
        - feature_names_selected: array of shape (n_selected_features,), names of the selected features.

    Notes:
        This function uses an ElasticNet model with specified regularization parameters for feature 
        selection. Features are selected if their importance meets or exceeds the mean coefficient 
        magnitude across features, as determined by the model.
    """
    clf_elastic = LogisticRegression(penalty = "elasticnet", solver="saga", C = 1,
                                     l1_ratio= 0.5,
                             max_iter=1000, tol=0.0001, 
                             random_state=0)
    sfm = SelectFromModel(clf_elastic, threshold="mean")
    sfm.fit(X_train, y_train)
    X_train_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)
    # Get feature names of selected features
    is_selected = sfm.get_support()
    feature_names_selected = feature_names[is_selected]
    
    return X_train_selected, X_test_selected, feature_names_selected


def select_features_regression(X_train, X_test, y_train, feature_names):
    """
    Perform feature selection using ElasticNet regularization, retaining 
    features based on a threshold applied to model coefficients.

    Arguments:
        - X_train: array or DataFrame (n_samples_train, n_features), feature set for training.
        - X_test: array or DataFrame (n_samples_test, n_features), feature set for testing.
        - y_train: array (n_samples_train,), target values for training.
        - feature_names: array (n_features,), names of all features initially provided.

    Returns:
        - X_train_selected: numpy array (n_samples_train, n_selected_features), 
                            training data with selected features.
        - X_test_selected: numpy array (n_samples_test, n_selected_features), 
                           testing data with selected features.
        - feature_names_selected: array of shape (n_selected_features,), names of the selected features.

    Notes:
        This function uses an ElasticNet model with specified regularization parameters for feature 
        selection. Features are selected if their importance meets or exceeds the mean coefficient 
        magnitude across features, as determined by the model.
    """
    clf_elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                             max_iter=1000, tol=0.0001, 
                             random_state=0, selection='cyclic')
    sfm = SelectFromModel(clf_elastic, threshold="mean")
    sfm.fit(X_train, y_train)
    X_train_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)
    # Get feature names of selected features
    is_selected = sfm.get_support()
    feature_names_selected = feature_names[is_selected]
    
    return X_train_selected, X_test_selected, feature_names_selected