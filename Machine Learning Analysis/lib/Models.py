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
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                 min_samples_split=5, max_features=1.0, 
                                 bootstrap=True, oob_score=False, 
                                 random_state=0, max_samples=None)
    clf = clf.fit(X, y)
    feature_weights = clf.feature_importances_
    return clf, feature_weights

def fit_svm_classifier(X, y):
    clf = svm.SVC(C=1.0, kernel='linear', random_state=0)
    clf = clf.fit(X, y)
    feature_weights = clf.coef_
    return clf, feature_weights


def fit_random_forest_regressor(X, y):
    clf = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                max_depth=None, min_samples_split=5,
                                max_features=1.0, bootstrap=True,
                                oob_score=False, random_state=0,
                                max_samples=None)
    clf.fit(X, y)
    feature_weights = clf.feature_importances_
    return clf, feature_weights


def fit_ridge_regressor(X, y):
    clf = Ridge(fit_intercept=False, random_state=0)
    clf.fit(X, y)
    feature_weights = clf.coef_
    return clf, feature_weights