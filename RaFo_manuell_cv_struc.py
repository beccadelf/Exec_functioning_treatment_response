# -*- coding: utf-8 -*-
"""
Basis Script for Machine Learning (Random Forest)
"""

# %% Import libraries/functions

import os
import pandas as pd
import numpy as np
import re
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer # must be imported to use IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, ElasticNet
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

# %% Set global variables

OPTIONS_OVERALL = {}
OPTIONS_OVERALL['number_iterations'] = 100 # planned number: 100
OPTIONS_OVERALL['Analysis'] = "clin_features" # choose between "all_features" and "clin_features"
input_path = "Z:\\PsyThera\\Projekte_Meinke\\Labrotation_Rebecca\\Feature_Label_Dataframes"

if OPTIONS_OVERALL['Analysis'] == "all_features":
    X_path = os.path.join(input_path,"Features.csv")
    results_path = "Z:\\PsyThera\\Projekte_Meinke\\Labrotation_Rebecca\\Results_ML\\Mit_oversampling\\All_features"
elif OPTIONS_OVERALL['Analysis'] == "clin_features":
    X_path = os.path.join(input_path,"Clinical_Features.csv")
    results_path = "Z:\\PsyThera\\Projekte_Meinke\\Labrotation_Rebecca\\Results_ML\\Mit_oversampling\\Demo_clin_features"
y_path = os.path.join(input_path,"Labels.csv")
random_state_seed = 42

# %% Define classes

# From here: https://stackoverflow.com/questions/72572232/how-to-preserve-column-order-after-applying-sklearn-compose-columntransformer-on?noredirect=1&lq=1
class ReorderColumnTransformer(BaseEstimator, TransformerMixin):
    index_pattern = re.compile(r'\d+$')
    
    def __init__(self, column_transformer):
        self.column_transformer = column_transformer
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        order_after_column_transform = [int( self.index_pattern.search(col).group()) for col in self.column_transformer.get_feature_names_out()]
        order_inverse = np.zeros(len(order_after_column_transform), dtype=int)
        order_inverse[order_after_column_transform] = np.arange(len(order_after_column_transform))
        return X[:, order_inverse]


class ZScalerDimVars(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = None
        self.dim_features= None

    def fit(self, X, y = None):
        unique_counts = (np.sort(X, axis=0)[:-1] != np.sort(X, axis=0)[1:]).sum(axis=0) + 1
        self.dim_features = unique_counts > 2
        column_trans = ColumnTransformer(transformers=[('standard', StandardScaler(), self.dim_features)],
                                        remainder='passthrough')
        self.scaler = make_pipeline(column_trans, ReorderColumnTransformer(column_transformer=column_trans))
        self.scaler.fit(X)

    def transform(self, X):
        if self.scaler is None or self.dim_features is None:
            raise ValueError("Scaler not fitted. Call fit method before transform.")
        return self.scaler.transform(X)
    
    def fit_transform(self, X, y=None):
        if self.scaler is None or self.dim_features is None:
            self.fit(X)
        return self.transform(X)

# %% Define functions

# 1. Functions for use within each iteration of crossvalidation

def load_data(X_path, y_path):
    X_df = pd.read_csv(X_path) # features
    y = pd.read_csv(y_path) # labels
    # Get feature_names 
    feature_names = X_df.columns
    # Turn into numpy arrays for higher speed
    X = np.array(X_df)
    y = np.array(y)
    
    return X, y, feature_names

def split_data(X, y, random_state):
    # Initialize split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                        stratify = y,
                                                        random_state = random_state)
    
    return X_train, X_test, y_train, y_test

def oversample_data(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy = 'minority')
    X_over, y_over = oversample.fit_resample(X_train, y_train)
    return X_over, y_over

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
        random_state=random_state_seed)
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
    clf_elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, fit_intercept=False,
                             max_iter=1000, tol=0.0001, 
                             random_state=random_state_seed, selection='cyclic')
    sfm = SelectFromModel(clf_elastic, threshold="mean")
    sfm.fit(X_train, y_train)
    X_train_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)
    # Get feature names of selected features
    is_selected = sfm.get_support()
    feature_names_selected = feature_names[is_selected]
    return X_train_selected, X_test_selected, feature_names_selected

def fit_and_apply_model(X_train, y_train, X_test):
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, 
                                 min_samples_split=5, max_features=1.0, 
                                 bootstrap=True, oob_score=False, 
                                 random_state=random_state_seed, max_samples=None)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    feat_imp = clf.feature_importances_
    
    return y_pred, feat_imp

# 2. Function to evaluate the model (i.e. calculate metrics)

def evaluate_randomforest_model(y_true, y_pred):
    """ Calculate evaluation metrics for random forest model
    Args:
        y_true: true labels (i.e. y_test).
        y_pred: labels predicted by a classifier.
    Returns:
        a dictionary with evaluation metrics
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    bal_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
    specificity = metrics.recall_score(y_true, y_pred, pos_label = 0)
    sensitivity = metrics.recall_score(y_true, y_pred, pos_label = 1)
    f1_score = metrics.f1_score(y_true, y_pred)
    # Save metrics in dataframe
    ev_metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": bal_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score}
    
    return ev_metrics

def get_ev_metrics_across_folds(outcomes, key_ev_metrics):
    """ Returns dataframe with evaluation metrics per fold
    """
    # Turn all ev_metrics dictionaries into dataframes and concatenate them
    dataframes = [pd.DataFrame([inner_dict[key_ev_metrics]]) for inner_dict in outcomes]
    ev_metrics_across_cvs_df = pd.concat(dataframes, ignore_index=True)
    
    return ev_metrics_across_cvs_df

# 3. Function to calculate summary statistics (Min, Max, Mean, Std) for each metric in ev_metrics

def calc_sum_statistic_for_ev_metrics(outcomes, key_ev_metrics):
    """
    Input: 
        outcomes: list with results-dictionaries (results_single_iter) of each iteration
        key_ev_metrics: name of dictionary with evaluation metrics
    Output: dataframe with summary statistics for ev_metrics 
    """
    
    sum_stat_ev_metrics = pd.DataFrame()
    count_iters = len(outcomes)
    
    # Loop over each metric within key_ev_metrics
    for var in outcomes[0][key_ev_metrics]: # TODO: Frage: wieso ist [0] hier notwendig, wenn lediglich auf metrics zugegriffen werden soll?
    # Combine values of all iterations to a list
        list_var = [itera[key_ev_metrics][var] for itera in outcomes]
        
        # Calculate summary statistics
        if count_iters > 1:
            min_val = min(list_var)
            max_val = max(list_var)
            mean_val = np.mean(list_var)
            std_val = np.std(list_var)
        elif count_iters == 1:
            min_val = "NA"
            max_val = "NA"
            mean_val = list_var[0]
            std_val = "NA"
        
        # Assign summary statistics to the DataFrame
        sum_stat_ev_metrics["Min_" + var] = [min_val]
        sum_stat_ev_metrics["Max_" + var] = [max_val]
        sum_stat_ev_metrics["Mean_" + var] = [mean_val]
        sum_stat_ev_metrics["Std_" + var] = [std_val]
    
    return sum_stat_ev_metrics

# 4. Function to calculate summary statistis (selection frequency, mean importance) of each feature

def summarize_features(outcomes, key_feat_names, key_feat_imps, feature_names_all):
    """
    Input: 
        outcomes: list with results-dictionaries (results_single_iter) of each iteration
        key_feat_names: selected features of respective iteration
        key_feat_imps: feature importances of respective iteration
        feature_names_all: all feature names
    Output: dataframe 
    """
    # Create empty df with all feature names as index
    feat_all_data = []
    empty_df = pd.DataFrame(index = feature_names_all)
    feat_all_data.append(empty_df)
    
    # Loop over results-dictionary of each iteration
    for itera in outcomes:
        feature_names = itera[key_feat_names]
        feature_imps = itera[key_feat_imps].T
        features_imps_df = pd.DataFrame(feature_imps, index=feature_names)
        feat_all_data.append(features_imps_df)
        
    # Concatenate the collected DataFrames into a single DataFrame
    feat_all_df = pd.concat(feat_all_data, axis=1)
    
    # Get mean
    mean = feat_all_df.mean(axis = 1) # mean importances per feature
    count_na = feat_all_df.isna().sum(axis=1) 
    sel_freq = feat_all_df.shape[1] - count_na # number of selections
                   
    feat_sum_df = pd.DataFrame({
        "selection frequency": sel_freq, 
        "mean feature importance": mean},
        index = feat_all_df.index)
    
    feat_sum_df.sort_values(by= ["selection frequency","mean feature importance"], key = abs, ascending = False, inplace = True)

    return feat_sum_df

# %% Procedure for one single iteration

# TEST
# num_iter = 0

def pipe_single_iter(num_iter):
    
    # We could also load the data outside the function, but it is better for parallel processing, if it is loaded in the function
    X, y, feature_names = load_data(X_path, y_path)
    
    X_train, X_test, y_train, y_test = split_data(X, y, random_state = num_iter) 
    # By changing the random_state with each iteration, we always get a different split
    
    X_train_over, y_train_over = oversample_data(X_train, y_train)
    
    X_train_imp, X_test_imp = impute_data(X_train_over, X_test)
    
    X_train_imp_scaled, X_test_imp_scaled = z_scale_data(X_train_imp,X_test_imp)
    
    X_train_imp_scaled_sel, X_test_imp_scaled_sel, features_selected = select_features(X_train_imp_scaled, X_test_imp_scaled, y_train_over,feature_names)
    
    y_pred_test, feat_imps = fit_and_apply_model(X_train_imp_scaled_sel, y_train_over, X_test_imp_scaled_sel)
    
    ev_metrics = evaluate_randomforest_model(y_true = y_test, y_pred = y_pred_test)
    
    # Create one dictionary with results to save
    results_single_iter = {"ev_metrics": ev_metrics,
                           "feature_names_sel": features_selected,
                           "feature_importances": feat_imps}
    
    return results_single_iter

# %% Final run
if __name__ == '__main__':
    # reminder()
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    # Load data once outside the loop to get the feature names
    X, y, feature_names = load_data(X_path, y_path)
    #pool = Pool(54)
    runs_list = []
    outcomes = []
    for i in range (OPTIONS_OVERALL['number_iterations']):
        runs_list.append(i)
    #outcomes[:] = pool.map(just_do_it,runs_list)
    # We are using map to enable parallel-processing
    outcomes[:] = map(pipe_single_iter,runs_list)
    ev_metrics_across_cvs = get_ev_metrics_across_folds(outcomes, key_ev_metrics = "ev_metrics")
    ev_metrics_sum = calc_sum_statistic_for_ev_metrics(outcomes, key_ev_metrics = "ev_metrics")
    feat_sum = summarize_features(outcomes = outcomes, 
                                  key_feat_names = "feature_names_sel", 
                                  key_feat_imps = "feature_importances", 
                                  feature_names_all = feature_names)
    
    ev_metrics_across_cvs.to_csv(os.path.join(results_path,"ev_metrics_across_folds.txt"),sep="\t", na_rep ="NA")
    feat_sum.to_csv(os.path.join(results_path,"features_sum.txt"),sep="\t", na_rep ="NA")
    ev_metrics_sum.T.to_csv(os.path.join(results_path,"ev_metrics_sum.txt"),sep="\t")