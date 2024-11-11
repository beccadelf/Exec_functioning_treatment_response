# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:43:52 2024

@author: Acer
"""
#%% Import functions

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn import svm

#%% Define functions

def fit_random_forest_classifier(X, y):
    """
    Function to train a Random Forest classifier and get the feature importances.

    Arguments:
        X: array or DataFrame (n_samples, n_features), feature set for training.
        y: array (n_samples,), target labels for training.
    
    Returns:
        - clf: Fitted RandomForestClassifier model.
        - feature_weights: array (n_features,), importance scores for each feature.
    """
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                 min_samples_split=5, max_features=1.0, 
                                 bootstrap=True, oob_score=False, 
                                 random_state=0, max_samples=None)
    clf = clf.fit(X, y)
    feature_weights = clf.feature_importances_
    
    return clf, feature_weights


def fit_svm_classifier(X, y):
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
    clf = svm.SVC(C=1.0, kernel='linear', random_state=0)
    clf = clf.fit(X, y)
    feature_weights = clf.coef_
    
    return clf, feature_weights


def fit_random_forest_regressor(X, y):
    """
    Function to train a Random Forest regressor and get the feature importances.
    
    Arguments:
        X: array or DataFrame (n_samples, n_features), feature set for training.
        y: array (n_samples,), target values for training.
    
    Returns:
        - clf: Fitted RandomForestRegressor model.
        - feature_weights: array (n_features,), importance scores for each feature.
    """
    clf = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                max_depth=None, min_samples_split=5,
                                max_features=1.0, bootstrap=True,
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
    clf = Ridge(fit_intercept=False, random_state=0)
    clf.fit(X, y)
    feature_weights = clf.coef_
    
    return clf, feature_weights