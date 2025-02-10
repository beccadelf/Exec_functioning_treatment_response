# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:43:52 2024

@author: Rebecca Delfendahl, Charlotte Meinke
"""
#%% Import functions

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn import svm

#%% Define functions

def fit_random_forest_classifier(X, y, max_features = "sqrt"):
    """
    Function to train a Random Forest classifier and get the feature importances.

    Arguments:
        X: array or DataFrame (n_samples, n_features), feature set for training.
        y: array (n_samples,), target labels for training.
    
    Returns:
        - clf: Fitted RandomForestClassifier model.
        - feature_weights: array (n_features,), importance scores for each feature.
    """
    # These are the default settings in scikit-learn (max_features = "sqrt")
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                 min_samples_split=2, max_features=max_features, 
                                 bootstrap=True, oob_score=False, 
                                 random_state=0, max_samples=None)
    clf = clf.fit(X, y)
    feature_weights = clf.feature_importances_
    
    return clf, feature_weights


def fit_svm_classifier(X, y, C = 0.1, kernel = "rbf"):
    """
    Function to train a Support Vector Machine (SVM) classifier with a linear kernel 
    and get the feature weights.
    
    Arguments:
        X: array or DataFrame (n_samples, n_features), feature set for training.
        y: array (n_samples,), target labels for training.
    
    Returns:
        - clf: Fitted SVM model with a linear kernel.
        - feature_weights: array (1, n_features), coefficients representing the
                           importance of each feature in the linear decision boundary.
    """
    clf = svm.SVC(C=C, kernel= kernel, random_state=0)
    clf = clf.fit(X, y)
    
    if kernel == "linear":
        feature_weights = clf.coef_
    else: 
        feature_weights = np.full(X.shape[1], np.nan) 
    
    return clf, feature_weights


def fit_random_forest_regressor(X, y, max_features = None):
    """
    Function to train a Random Forest regressor and get the feature importances.
    
    Arguments:
        X: array or DataFrame (n_samples, n_features), feature set for training.
        y: array (n_samples,), target values for training.
        max_features is set to None as recommended it scikit-learn and due to the relatively small number of features.
        It can also bet set to n_features//3 in the main script (as recommended in Probst)
    
    Returns:
        - clf: Fitted RandomForestRegressor model.
        - feature_weights: array (n_features,), importance scores for each feature.
    """
    # Default settings of scikit, modified based on recommendations in Probst et al. (2019, e.g., min_samples_split = 5)
    clf = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                max_depth=None, min_samples_split=5, min_samples_leaf = 1,
                                max_features= max_features, bootstrap=True,
                                oob_score=False, random_state=0,
                                max_samples=None)
    clf.fit(X, y)
    feature_weights = clf.feature_importances_
    
    return clf, feature_weights


def fit_ridge_regressor(X, y):
    """
    Function to train a Ridge regression model and get the feature weights.

    Arguments:
        X: array or DataFrame (n_samples, n_features), feature set for training.
        y: array (n_samples,), target values for training.

    Returns:
        - clf: Fitted Ridge regression model.
        - feature_weights: array (n_features,), coefficients of each feature in the regression model.
    """
    clf = Ridge(fit_intercept=True, random_state=0)
    clf.fit(X, y)
    feature_weights = clf.coef_
    
    return clf, feature_weights