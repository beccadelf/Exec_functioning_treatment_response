# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:36:56 2024

@author: Acer
"""

#%% Import functions

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer # must be imported to use IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

from lib.Preprocessing_Classes import ZScalerDimVars
#%% Define functions

def load_data(X_path, y_path):    
    X_df = pd.read_csv(X_path) # features
    y = pd.read_csv(y_path) # outcome
    # Get feature_names 
    feature_names = X_df.columns
    # Turn into numpy arrays for higher speed
    X = X_df.to_numpy()
    y = np.ravel(y) 
    return X, y, feature_names

def impute_data(X_train, X_test):
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
    # z-scale only non-binary data!
    scaler = ZScalerDimVars()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def select_features(X_train, X_test, y_train, feature_names):
    clf_elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False,
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