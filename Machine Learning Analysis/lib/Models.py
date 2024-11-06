# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:43:52 2024

@author: Acer
"""
#%% Import functions

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

#%% Define functions

def fit_random_forest_classifier(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                 min_samples_split=5, max_features=1.0, 
                                 bootstrap=True, oob_score=False, 
                                 random_state=0, max_samples=None)
    clf = clf.fit(X_train, y_train)
    feat_imp = clf.feature_importances_
    return clf, feat_imp


def fit_random_forest_regressor(X_train, y_train):
    clf = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                max_depth=None, min_samples_split=5,
                                max_features=1.0, bootstrap=True,
                                oob_score=False, random_state=0,
                                max_samples=None)
    clf.fit(X_train, y_train)
    feature_weights = clf.feature_importances_
    return clf, feature_weights


def fit_ridge_regressor(X_train, y_train):
    clf = Ridge(fit_intercept=False, random_state=0)
    clf.fit(X_train, y_train)
    feature_weights = clf.coef_
    return clf, feature_weights