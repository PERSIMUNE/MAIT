
# # MAIT 1.0.0 template

# %%
# Here we load dataset and required libraries 
# Remove Future Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from numpy.random import uniform, randint
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn import metrics
from sklearn.metrics import make_scorer, matthews_corrcoef, average_precision_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score, brier_score_loss, precision_recall_curve, roc_curve, auc, matthews_corrcoef, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler, LabelEncoder, MinMaxScaler
import shap
import matplotlib.pyplot as plt
from sklearn import ensemble
import seaborn as sns
from scipy.stats import pointbiserialr, wilcoxon, mannwhitneyu, chi2_contingency, norm, iqr, kruskal, spearmanr, ttest_rel, linregress
from joblib import Parallel, delayed
from mrmr import mrmr_classif # for feature selection (optional)
from imblearn.over_sampling import RandomOverSampler # for oversampling (optional)
# loading models (libraries or packages for base models)
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LogisticRegression
import feyn
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import AgglomerativeClustering
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc, concordance_index_censored
from sklearn.ensemble import IsolationForest, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import joblib
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.semi_supervised import LabelPropagation
from survshap import SurvivalModelExplainer, PredictSurvSHAP, ModelSurvSHAP
from sksurv.compare import compare_survival
from sklearn.dummy import DummyClassifier
from scipy.integrate import trapezoid
from matplotlib import cm
from matplotlib.table import table
from sympy import init_printing
import itertools
from io import StringIO
from IPython import get_ipython
from IPython.display import display, Javascript
import sys
import re
import random
import os
import time
import platform
import subprocess
import psutil

# Initialize MinMaxScaler for normalization
minmax_scaler = MinMaxScaler()

# %% [markdown]
# ### Most important settings
# 
# Below you can find parameters and configurations to set and complete like an entry form that are critical and important, so please make sure that you have an understanding on the parameters by reading the comments and the documentation of the pipeline (the MANUAL page on GitHub).

# %%
# settings for categorical variables
# specify the names of categorical features (variables) - if no categorical feature leave it empty as []
cat_features = []
merged_rare_categories = True # merge rare categories and unify missing categories (recommended)
rarity_threshold = 0.05  # Define a threshold for rarity (e.g., 0.05 means 5%) this is used to merge rare categories in categorical features (optional)

###################################################################################
# specify columns that must be removed
columns_to_drop = [] # specify column names that must be dropped 

###################################################################################
# import data
mydata = pd.read_csv("combined_data_Azithromycin.csv")
external_val = False # True if you have a dataset that can be used as an external validation data
# load any external data
# extval_data = pd.read_excel('external_validation_data.xlsx')
ext_val_demo = False # only used to run a demo for external validation - this creates external validation dataset for simulation 

###################################################################################
# random data split
data_split = True # True to apply stratified data split by outcome variable (e.g., 80% training or development data and 20% test data) 
# if data_split = False, all the dataset will be used for cross validation (can be used when there is no enough data to set aside for the test set)
train_size_perc = 0.8 # percentage of the samples to be used for training (e.g. 0.8 means 80% of samples for training)
# see the following conditions to check if you need to do any custom data split based on your data
# below can be used in the case where multiple samples (instances) are available from same patients
data_split_by_patients = False # True to apply data split by patient ID (the column name that contains patient ID should then be specified)
if data_split_by_patients:
    patient_id_col = "patient_ID" # the column name that contains patient ID should then be specified (if not patients, it could be any individual identification number for example)
data_split_multi_strats = False # True if you need to use more than one variable for stratification
if data_split_multi_strats: # the names of the columns used for multiple stratification should be specified by user
    strat_var1 = "stratification variable 1"
already_split = False # indicate if the data is already split to train and test sets
if already_split: # specify the names of the train (development) and test sets
    # Splitting based on values
    testset = mydata[mydata['subset'] == 'Test']
    mydata = mydata[mydata['subset'] == 'Train']
# so data_split = True is used for MAIT Discovery and Prediciton whereas data_split = False is used for MAIT Discovery pipeline (only cross validation is done)
###################################################################################

# available binary classification models in the pipeline to use (7 in total) - you can delete the name of any algorithms/models from the list to exclude them from the pipeline
models_to_include = ["QLattice_mdl", "NaiveBayes_mdl", "RandomForest_mdl", "LightGBM_mdl", "CatBoost_mdl", "LogisticRegression_mdl", "HistGBC_mdl"]

# outcome variable (e.g., class 1 indicated as "0" and class 2 as "1")
outcome_var = "azm_sr" # specify the name of the column for the binary outcome variable (note: it should not contain any missingness)

###################################################################################
# set a directory to save the results
main_folder_name = 'results_Azithromycin'
# Define class labels for display
class_labels_display = ['non-resistant', 'resistant']   # Specify the labels for the two classes to display in figures

# Specify the class labels
class_0 = class_labels_display[0]
class_1 = class_labels_display[1]

# Create a mapping dictionary for class labels
class_label_dict = {0.0:class_0, 1.0:class_1}  # this has to be set by user based on class labels in the outcome variable

###################################################################################
# feature selection
feat_sel = True # feature selection based on minimum Redundancy - Maximum Relevance (mRMR) algorithm
num_features_sel = 30 # number of features to select using mRMR algorithm within each fold (common selected features are then used for machine learning). 
# If there was no common selected features, increase num_features_sel.
top_n_f = 20 # number of top features (most impactful features) based on SHAP values to be displayed for SHAP plots

###################################################################################
# survival analysis
# Two models are included: random survival forest (RSF) as main model and Cox proportional hazard (CPH) model as a baseline model to compare against the RSF
survival_analysis = False # True to conduct survival analyses. To do this you should provide a backup data that contains a column for time-to-event
if survival_analysis:
    survival_demo = False # only used to create a column for time to event just to showcase how the results would look like if survival models are used
    time_to_event_column = "" # use the column name for time-to-event in your data
    if survival_demo: 
        # Adding a new column with random integers between 90 to 365 (only for demonstration purpose - not to be used when the data is available)
        mydata[time_to_event_column] = np.random.randint(90, 366, size=len(mydata))
        
    mydata_copy_survival = mydata.copy() # get a copy of your data as back up for the time-to-event column
    # mydata.drop(columns = [time_to_event_column], inplace = True) # remove the time-to-event column for the data that's going to be used for binary classification

###################################################################################
# regression analysis
# Two models are included: random forest regressor (RFR) as main model and linear regression model as a baseline model to be compared against the RFR
regression_analysis = False
if regression_analysis:
    regression_outcome = "regression_outcome_var"
    demo_regression_analysis = False # only used for demonstration (simulation) purpose when the data is not available - not to be used otherwise
    if demo_regression_analysis:
        mydata_copy_regression = mydata.copy()
        # Generate random features
        X = np.random.randn(mydata_copy_regression.shape[0], mydata_copy_regression.shape[1])
        # Define coefficients for each feature
        true_calculate = np.random.randn(mydata_copy_regression.shape[1])
        # Generate outcome variable (target) based on features and calculate
        # Adding some noise for randomness
        noise = np.random.randn(mydata_copy_regression.shape[0]) * 0.5  # Adjust the magnitude of noise
        mydata_copy_regression[regression_outcome] = np.dot(X, true_calculate) + noise

###################################################################################
# settings for processing resouces
GPU_avail = True # True if GPU is available in your machine otherwise set to False
hp_tuning = True # True if you want to conduct hyperparameter tuning otherwise set to False
n_cpu_for_tuning = 20 # number of CPUs to be available for hyperparameter tuning
n_cpu_model_training = 20 # number of CPUs to be available for model training
n_rep_feature_permutation = 100 # number of repetitions for feature permutation
n_iter_hptuning = 10 # number of iterations in repeated cross validation for hyperparameter tuning
SEED = 123 # arbitrarily chosen, this modifies computer randomization, if there are significant differences between train and test sets due to the random data split, this can be modified for example

###################################################################################

###################################################################################
cv_folds = 5 # number of folds for the outer loop in cross validation
cv_folds_hptuning = 5 # number of folds for hyperparameter tuning (inner loop - nested cross validation)
use_default_threshold = True # use default threshold of 0.5 bor binary classification, otherwise it optimize the threshold based on the development set
test_only_best_cvmodel = True # True to test only the best performing model from cross validation, this option speeds up the process
###################################################################################

###################################################################################
# handle missingness
exclude_highly_missing_columns = True # True to exclude features with high missingness
exclude_highly_missing_rows = True # True to exclude rows (samples) with high missingness
column_threshold = 0.99  # Threshold for variables - columns (e.g., 99% missingness)
row_threshold = 0.90     # Threshold for samples - rows (e.g., 90% missingness)

###################################################################################
remove_outliers = False # True to enable outlier detection and removal using Isolation Forest algorithm
###################################################################################
# Specify the filename of this Jupyter notebook so that it can be saved after execution
JupyterNotebook_filename = "MAIT_Tutorial_Azithromycin_pub.ipynb"

# %% [markdown]
# ### Less important settings
# 
# Here you have configurations that can be set but they are usually fine to be set as it is (default settings).

# %%
###################################################################################
# continuous variables
specify_continuous_variables = False  # optional but recommended in case there are continuous variables that may have entries that could be recognized as categorical variables
continuous_features = []

###################################################################################
export_missclassified = False
if export_missclassified:
    mydata_backup = mydata.copy()
    mydata_backup['ID'] = mydata["patid"]
    if already_split:
        testset_backup = testset.copy()
        testset_backup['ID'] = testset["patid"]
        mydata_backup = pd.concat([mydata_backup, testset_backup])
    # Generate random IDs in the format "ID_randomnumber"
    # mydata_backup['ID'] = mydata["ID"]

###################################################################################
# data manipulation settings
oversampling = False # apply oversampling using random oversampling only on train set to increase the number of samples of the minority class
scale_data = False # data scale using robust scaling (see scikit-learn)
semi_supervised = False # if True it applies a method to impute missingness in the outcome variable using label propagation method otherwise they are excluded

###################################################################################
# supplementary analyses
model_uncertainty_reduction = False # True to use model uncertainty reduction (MUR) approach to build more robust models (filtering samples based on SHAP values and predicted probabilities being close to the chance level)
do_decision_curve_analysis = True # True to conduct decision curve analysis
calibration_and_conformal_predictions = True # True to conduct model calibration and conformal predictions for the best performing binary classification model (recommended only if a large dataset is available)
find_interacting_feature_permutation = False # True to enable the analysis based on feature permutation on the final model to inspect potential interactions among pairs of features

###################################################################################
# settings for displaying feature names
shorten_feature_names = True # True to shorten feature names (for visualization purposes if your data has features with too long names)
if shorten_feature_names:
    fname_max_length = 30 # Specify max length (number of characters for the feature names to appear)
data_dictionary = {
}
# optional, using the data dictionary the name of the variables are displayed in figures and tables to facilitate reporting.
###################################################################################

###################################################################################
demo_configs = True # True only if using the Breast Cancer dataset to add missingness and a categorical feature to the data for demonstration - not to be used otherwise

###################################################################################
# following is the default custom metric (Mean of ROCAUC and PRAUC) used for hyperparameter tuning
def combined_metric(y_true, y_pred_proba):
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    return (roc_auc + pr_auc) / 2  # Mean of ROCAUC and PRAUC

custom_scorer = make_scorer(combined_metric, needs_proba=True)
use_single_metric = False # in case you want to use a single metric for training and hyperparameter tuninig
if use_single_metric:
    single_score = "ROCAUC" # choose between ROCAUC and PRAUC
    if single_score == "ROCAUC":
        custom_scorer = make_scorer(roc_auc_score, needs_proba=True)
    elif single_score == "PRAUC":
        custom_scorer = make_scorer(average_precision_score, needs_proba=True)

###################################################################################
# Check if the main folder exists, and create it if not
if not os.path.exists(main_folder_name):
    os.makedirs(main_folder_name)

# Change the current working directory to the main folder
os.chdir(main_folder_name)

# reporting file formats
fig_file_format = "tif" # specify desired file format for figures (e.g. tif, svg)

# limit the number of lines to display
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

skip_block = False # should not be changed
###################################################################################
# Record start time
start_time = time.time()

# %% [markdown]
# #### Data preparation
# 
# The data may require some preparation before feeding them to machine learning models (algorithms).
# The most important things to check are:
# data types (feature types: categorical, numerical, etc.)
# missingness
# if there are more than one dataset, all variables must have the same definition and type across datasets 
# redundant features

# %%
# Ensure the outcome variable exists in the dataset
if outcome_var in mydata.columns:
    # Map the class labels in the outcome variable
    mydata[outcome_var] = mydata[outcome_var].map(class_label_dict)
    if already_split: 
        testset[outcome_var] = testset[outcome_var].map(class_label_dict)
    if external_val:
        extval_data[outcome_var] = extval_data[outcome_var].map(class_label_dict)
else:
    raise ValueError(f"'{outcome_var}' column not found in the dataset")

# %%
# add missingness
# empty entries replaced by NaN
mydata.replace(" ", np.nan, inplace=True)

# If there are lost to follow ups (missing endpoint/event) in the data, one option is to use semi-supervised learning methods to impute missing target labels (outcome variable)
# LabelPropagation assigns the labels based on similarity of samples (instances)
if semi_supervised:
    # Replace NaN values in the outcome variable with -1
    mydata[outcome_var] = mydata[outcome_var].replace(np.nan, -1)

    # Create a LabelPropagation model
    label_prop_model = LabelPropagation()

    # Separate the features (X) and labels (y)
    X = mydata.drop(columns=[outcome_var])
    y = mydata[outcome_var]

    # Fit the model on the dataset (X: features, y: labels including -1 for missing labels)
    label_prop_model.fit(X, y)

    # Get the predicted labels (transduction_ includes predictions for both labeled and unlabeled)
    predicted_labels = label_prop_model.transduction_

    # Update the original DataFrame with predicted labels where labels were missing (-1)
    mydata.loc[mydata[outcome_var] == -1, outcome_var] = predicted_labels[mydata[outcome_var] == -1]

else:
    # Remove rows where outcome_var column contains NaN (missing) values
    mydata = mydata.dropna(subset=[outcome_var])

if demo_configs: # used only for the Breast Cancer dataset
    # Randomly selecting some indices to set as missing values
    np.random.seed(SEED)
    rows_to_change = np.random.choice(mydata.index, size=5, replace=False)
    np.random.seed(SEED)
    cols_to_change = np.random.choice(mydata.columns, size=5, replace=False)
    outcome_var_backup = mydata[outcome_var]
    # Setting these selected entries to NaN
    mydata.loc[rows_to_change, cols_to_change] = np.nan
    mydata[outcome_var] = outcome_var_backup

    # Adding a column for race with three categories
    races = ['White', 'Black', 'Asian']
    np.random.seed(SEED)
    mydata['Race'] = np.random.choice(races, size=mydata.shape[0])

mydata.drop(columns=columns_to_drop, inplace=True)
    
mydata[cat_features] = mydata[cat_features].astype('category')

# Convert categories to strings for each categorical column
for col in cat_features:
    mydata[col] = mydata[col].astype(str).astype('category')

if ext_val_demo:
    # Randomly select a few samples from the dataframe
    num_samples = 100  # Change this number to select different number of samples
    extval_data = mydata.sample(n=num_samples).reset_index(drop=True)
    
if external_val:
    columns_present = [col for col in columns_to_drop if col in extval_data.columns]
    if columns_present:
        extval_data.drop(columns=columns_present, inplace=True)

    extval_data[cat_features] = extval_data[cat_features].astype('category')

    # Convert categories to strings for each categorical column
    for col in cat_features:
        extval_data[col] = extval_data[col].astype(str).astype('category')
    extval_data.replace(" ", np.nan, inplace=True)

if already_split:
    columns_present = [col for col in columns_to_drop if col in testset.columns]
    if columns_present:
        testset.drop(columns=columns_present, inplace=True)

    testset[cat_features] = testset[cat_features].astype('category')

    # Convert categories to strings for each categorical column
    for col in cat_features:
        testset[col] = testset[col].astype(str).astype('category')
    testset.replace(" ", np.nan, inplace=True)

# %% [markdown]
# #### Specify data types for numerical features (optional)

# %%
if specify_continuous_variables:

    # Replace non-numeric values (including empty strings) with NaN
    mydata[continuous_features] = mydata[continuous_features].apply(pd.to_numeric, errors='coerce')

    # Convert to float64 after replacing non-numeric values with NaN
    mydata[continuous_features] = mydata[continuous_features].astype("float64")

# %% [markdown]
# #### Defined missingness
# 
# make sure all missing values are defined. For categorical features, missing values should be encoded as "missing"

# %%
# empty entries replaced by NaN (if NaN occurs after the previous code chunk)
mydata.replace(" ", np.nan, inplace=True)

# %% [markdown]
# #### Rare categories in categorical variables and data harmonization for missing values
# 
# If there are rare categories in categorical features, they can have negativae effect on learning process and thus the following code can handle merging such rare categories. It is however more favorable to be done by the data engineer or the researcher who knows the context of the data to merge rare categories in a meaningful way rather than automated merging.
# The code also ensures that all values similar to "missing" are treated as "missing" for unification.

# %%
# sometimes there are rare categories that can be merged in the data
if merged_rare_categories:
    if 'mydata_backup' not in locals():
        mydata_backup = mydata.copy()
    if external_val:
        mydata = pd.concat([mydata,extval_data])
    categorical_columns = mydata.select_dtypes(include=['category']).columns
    category_frequencies = {}
    for col in categorical_columns:
        category_counts = mydata[col].value_counts()
        category_frequencies[col] = category_counts

    print(category_frequencies)

    # Categorical columns
    categorical_columns = mydata.select_dtypes(include=['category']).columns

    # Dictionary to store category frequencies
    category_frequencies = {}

    # Loop through categorical columns
    for col in categorical_columns:
        # Calculate category frequencies
        category_counts = mydata[col].value_counts()
        
        # Identify rare categories
        rare_categories = category_counts[category_counts / len(mydata) < rarity_threshold].index
        
        # Group rare categories into a single category and eliminate individual rare categories
        grouped_category_name = ""
        for cat in rare_categories:
            grouped_category_name += f" or {cat}"
            # Replace individual rare category with an empty string
            mydata[col] = mydata[col].replace({cat: ""})
        
        # Replace the empty strings with the grouped category name
        mydata[col] = mydata[col].replace({"": grouped_category_name.lstrip(" or ")})
        
        # Store updated category frequencies
        category_frequencies[col] = mydata[col].value_counts()

        # Create a new categorical Series with only used categories
        used_categories = mydata[col].cat.categories
        mydata[col] = pd.Categorical(mydata[col], categories=used_categories)

    # Print updated categories with the original category index
    for col in categorical_columns:
        updated_categories = mydata[col].cat.categories
        print(f"Categories for {col}: {updated_categories}")

    #########################################################################
        # Define a list of values similar to "missing" that should be unified
        missing_values = ["missing", "unknown", "not available", "na", "n/a", "none", "Missing"]

        # Iterate through each categorical column
        for col in categorical_columns:
            # Check if "missing" is not already in the categories
            if "missing" not in mydata[col].cat.categories:
                # Add "missing" as a category
                mydata[col] = mydata[col].cat.add_categories("missing")
            
            # Replace values in the column (case-insensitive)
            mydata[col] = mydata[col].apply(lambda x: str(x).lower() if isinstance(x, str) else x)

            # Replace any "missing" related values (now lowercase) with "missing"
            mydata[col] = mydata[col].replace([val.lower() for val in missing_values], "missing").fillna("missing")
    #########################################################################


    # Find features with mixed category data types
    mixed_category_features = []
    for col in mydata.columns:
        if mydata[col].dtype == 'category':
            unique_categories = mydata[col].cat.categories
            unique_dtypes = set(type(cat) for cat in unique_categories)
            if len(unique_dtypes) > 1:
                mixed_category_features.append(col)

    # Convert categories to strings for mixed category features
    for feature in mixed_category_features:
        mydata[feature] = mydata[feature].astype('str').astype('category')


    # Identify categorical columns
    categorical_columns = mydata.select_dtypes(include=['category']).columns

    # Convert categories to strings for each categorical column
    for col in categorical_columns:
        mydata[col] = mydata[col].astype(str).astype('category')

    category_frequencies = {}
    for col in categorical_columns:
        category_counts = mydata[col].value_counts()
        category_frequencies[col] = category_counts

    print(category_frequencies)
    
    # Restore original index for mydata
    mydata = mydata.loc[mydata_backup.index]
    if external_val:
        # Restore original index for extval_data
        extval_data = extval_data.loc[extval_data.index]

# %%
mydata.dtypes

# %% [markdown]
# #### Shorten the name of features (optional)
# If feature names are longer than a specific number of characters specified by user, it cuts it down to that so when the feature names appear on the plots they're shortened for visualization purposes.

# %%
if shorten_feature_names:
    def shorten_column_names(df, max_length):
        """
        Shortens column names in a pandas DataFrame to fit within a maximum length.

        Parameters:
            df (pandas.DataFrame): The input DataFrame containing the columns to be shortened.
            max_length (int): The maximum allowed length for each column name.

        ## Returns
            list: A list of shortened column names.
        """
        new_columns = []
        used_names = set()  # Keep track of used names to avoid duplicates
        for col in df.columns:
            # Check if column name is longer than max_length
            if len(col) > max_length:
                # Shorten column name by keeping only the beginning part and adding ...
                new_col = col[:max_length] + '...'
            else:
                new_col = col
            
            # If the shortened name already exists, add a numeric suffix
            suffix = 1
            base_name = new_col
            while new_col in used_names:
                new_col = f"{base_name}_{suffix}"
                suffix += 1
            
            used_names.add(new_col)
            new_columns.append(new_col)
        return new_columns

    # Shorten column names
    mydata.columns = shorten_column_names(mydata, fname_max_length)

    if external_val:
        extval_data.columns = shorten_column_names(extval_data, fname_max_length)

    def shorten_data_dictionary(data_dict, max_length):
        """
        Shortens values in a dictionary to fit within a specified maximum length.

        This function replaces long values with shorter versions by truncating them and adding an ellipsis ('...') if necessary.
        If two values have the same length after truncation, it appends a numeric suffix (e.g., 'value_1', 'value_2') to make them unique.

        ## Parameters:
            data_dict (dict): The input dictionary containing the keys and values to be shortened.
            max_length (int): The maximum allowed length for each value in the dictionary.

        ## Returns
            dict: A new dictionary with shortened values.
        """
        new_data_dict = {}
        used_values = set()  # Keep track of used values to avoid duplicates
        for key, value in data_dict.items():
            # Check if column name is longer than max_length
            if len(value) > max_length:
                # Shorten column name by keeping only the beginning part and adding ...
                new_value = value[:max_length] + '...'
            else:
                new_value = value
            
            # If the shortened value already exists, add a numeric suffix
            suffix = 1
            base_value = new_value
            while new_value in used_values:
                new_value = f"{base_value}_{suffix}"
                suffix += 1
            
            used_values.add(new_value)
            new_data_dict[key] = new_value
        return new_data_dict

    # Shorten data dictionary
    data_dictionary = shorten_data_dictionary(data_dictionary, fname_max_length)


# %% [markdown]
# The code chunk below applies outlier (anomaly) detection and removal based on isolation forest algorithm. It's optional and is done if chosen by the user (remove_outliers = True). It follows these steps:
# 
# 1. Data Preparation:
# 
# Separates the input features (X) and the target variable (y) from the original dataset (mydata).
# Encodes categorical features using one-hot encoding to convert them into numerical format, avoiding multicollinearity by dropping the first category.
# 
# 2. Handling Missing Values:
# 
# Imputes missing values in the combined dataset (X_combined), which includes both numerical and encoded categorical features, using the K-Nearest Neighbors (KNN) imputation method. The number of neighbors used for imputation is calculated based on the size of the dataset.
# 
# 3. Outlier Detection:
# 
# Initializes an IsolationForest model to detect outliers.
# Fits the model to the data and predicts outliers, labeling them as -1.
# 
# 4. Filtering Outliers:
# 
# Filters out rows marked as outliers from both the features (X) and the target variable (y).
# Combines the cleaned features and target variable back into a single DataFrame (mydata).
# 
# 5. Final Cleanup:
# 
# Removes the 'outlier' column from the final DataFrame.

# %%
# to remove outliers automatically detected using isolation forest
if remove_outliers:
    # Separate features and outcome
    X = mydata.drop(columns=[outcome_var])
    y = mydata[outcome_var]
    
    # One-Hot Encode categorical features
    encoder = OneHotEncoder(drop='first', sparse=False)  # drop='first' to avoid multicollinearity, sparse=False to get a dense output
    X_encoded = encoder.fit_transform(X[cat_features])

    # Convert encoded features into a DataFrame with preserved index
    encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(cat_features), index=X.index)

    # Combine encoded features with numerical features
    X_combined = pd.concat([X.drop(columns=cat_features), encoded_df], axis=1)

    # Impute missing values in continuous features of X_train using KNN
    nn = int(np.sqrt(X_combined.shape[0])) # from Devroye, L., Györfi, L., & Lugosi, G. (1996). A Probabilistic Theory of Pattern Recognition. Springer. https://doi.org/10.1007/978-1-4612-0711-5.
    cont_imputer = KNNImputer(n_neighbors=nn, weights = 'distance', keep_empty_features = True)  
    
    X_combined = pd.DataFrame(cont_imputer.fit_transform(X_combined), index=X_combined.index)
    
    # Step 1: Initialize the IsolationForest model
    iso_forest = IsolationForest(contamination='auto', random_state=SEED)

    # Step 2: Fit the model to the DataFrame
    iso_forest.fit(X_combined)

    # Step 3: Predict the outliers (-1 indicates an outlier)
    X['outlier'] = iso_forest.predict(X_combined)

    # Step 4: Filter out the outliers using the original index
    mask = X['outlier'] != -1
    X = X[mask]
    y = y[mask]  # Filter y using the same mask
    
    mydata = pd.concat([X, y], axis=1)

    # Drop the 'outlier' column if you don't need it anymore
    mydata = mydata.drop(columns=['outlier'])
    
    if already_split:
        # Separate features and outcome
        X = testset.drop(columns=[outcome_var])
        y = testset[outcome_var]
        
        # One-Hot Encode categorical features
        encoder = OneHotEncoder(drop='first', sparse=False)  # drop='first' to avoid multicollinearity, sparse=False to get a dense output
        X_encoded = encoder.fit_transform(X[cat_features])

        # Convert encoded features into a DataFrame with preserved index
        encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(cat_features), index=X.index)

        # Combine encoded features with numerical features
        X_combined = pd.concat([X.drop(columns=cat_features), encoded_df], axis=1)

        # Impute missing values in continuous features of X_train using KNN
        nn = int(np.sqrt(X_combined.shape[0])) # from Devroye, L., Györfi, L., & Lugosi, G. (1996). A Probabilistic Theory of Pattern Recognition. Springer. https://doi.org/10.1007/978-1-4612-0711-5.
        cont_imputer = KNNImputer(n_neighbors=nn, weights = 'distance', keep_empty_features = True)  
        
        X_combined = pd.DataFrame(cont_imputer.fit_transform(X_combined), index=X_combined.index)
        
        # Step 1: Initialize the IsolationForest model
        iso_forest = IsolationForest(contamination='auto', random_state=SEED)

        # Step 2: Fit the model to the DataFrame
        iso_forest.fit(X_combined)

        # Step 3: Predict the outliers (-1 indicates an outlier)
        X['outlier'] = iso_forest.predict(X_combined)

        # Step 4: Filter out the outliers using the original index
        mask = X['outlier'] != -1
        X = X[mask]
        y = y[mask]  # Filter y using the same mask
        
        testset = pd.concat([X, y], axis=1)

        # Drop the 'outlier' column if you don't need it anymore
        testset = testset.drop(columns=['outlier'])

# %%
mydata.shape

# %% [markdown]
# ### Data split (prediction vs. discovery)
# 
# This section covers performing a random data split. It's important to note that the default data split may not always be the optimal approach. In certain cases, a custom data split, such as one stratified by multiple variables (data_split_multi_strats) or by patient ID (or other individual identifiers), may be more appropriate. The following code can be adjusted to accommodate these specific data splitting needs.
# 
# Additionally, if the primary aim of a study is to explore relationships between independent variables (features) and the outcome variable using cross-validation of machine learning models, it is recommended to use the entire dataset for cross-validation. This approach is particularly useful when the dataset is too small to be divided into a training (development) set and a test set.
# 
# However, if the study's goal is both to investigate associations and to develop a model for prognostic or diagnostic purposes, then data splitting becomes relevant. The first approach is considered a discovery phase, while the second approach is aimed at prediction.

# %%
mydata.shape

# %%
mydata[outcome_var].value_counts(dropna=False)

# %%
mydata[outcome_var].value_counts()

# %%

test_size_perc = 1 - train_size_perc
if data_split:
    if data_split_by_patients: 
        # If there are more than one sample per patient, this split is suggested
        # to avoid having the same patient data in both training and test sets.
        # In this case, the column that specifies patient ID is also used in the data split.
        unique_patients = mydata[patient_id_col].unique()
        train_patients, test_patients = train_test_split(unique_patients, test_size=test_size_perc, random_state=SEED)
        trainset = mydata[mydata[patient_id_col].isin(train_patients)]
        testset = mydata[mydata[patient_id_col].isin(test_patients)]

    elif data_split_multi_strats:
        # If the data split must be stratified by more than one variable.
        # In this case, if there are two variables specified by the user,
        # they should be combined to create a combined variable.
        # Then the combined variable is used for the stratification so that
        # the same portion of categories exist in both train and test sets.
        # this is defined for two variables, if you need more then you should modify the following code
        combined_strats = mydata[strat_var1].astype(str) + '_' + mydata[outcome_var].astype(str)
        mydata['combined_strats'] = combined_strats
        mydata, testset = train_test_split(mydata, test_size=test_size_perc, random_state=SEED, stratify=mydata['combined_strats'])
        mydata.drop(columns=['combined_strats',strat_var1], inplace=True)
        testset.drop(columns=['combined_strats',strat_var1], inplace=True)
        # mydata.drop(columns = ['combined_strats',strat_var1], inplace=True)
        if external_val:
            if not extval_data.empty and strat_var1 in extval_data.columns:
                extval_data.drop(columns=strat_var1, inplace=True)
    elif already_split:
        # If you have done data split already using a different approach,
        # here just check if trainset and testset are comparable
        # (have the same variables and datatype).
        if set(mydata.columns) != set(testset.columns):
            raise ValueError("Trainset and testset have different columns.")
        if not all(mydata.dtypes == testset.dtypes):
            raise ValueError("Trainset and testset have different data types for columns.")
        # Optionally, you can check for other comparability metrics as needed.
    else:
        # stratified data split based on outcome variable (see also the other conditions if they may be relevant for your dataset)
        mydata, testset = train_test_split(mydata, test_size= test_size_perc, random_state=SEED, stratify= mydata[outcome_var])
else:
    _, testset = train_test_split(mydata, test_size=test_size_perc, random_state=SEED, stratify= mydata[outcome_var]) # this way we keep a dummy testset just to avoid including too many conditions in the pipeline


# %% [markdown]
# #### Checking the availability of all categories
# make sure both train and test sets have all categorical levels

# %%
if data_split:
    # Get list of categorical variable names
    categorical_vars = mydata.select_dtypes(include=['category']).columns.tolist()

    for var in categorical_vars:
        unique_categories = set(mydata[var]).union(set(testset[var]))
        print(var)
        print(unique_categories)
        
        # Exclude old categories before adding new categories to the train set
        new_categories_train = unique_categories.difference(mydata[var].cat.categories)
        mydata[var] = mydata[var].cat.add_categories(new_categories_train)
        
        # Exclude old categories before adding new categories to the test set
        new_categories_test = unique_categories.difference(testset[var].cat.categories)
        testset[var] = testset[var].cat.add_categories(new_categories_test)
        
        if external_val:
            # Exclude old categories before adding new categories to the extval_data set
            new_categories_test = unique_categories.difference(extval_data[var].cat.categories)
            extval_data[var] = extval_data[var].cat.add_categories(new_categories_test)



# %%
mydata.dtypes

# %% [markdown]
# #### Filter highly missing data
# 
# Missing data can significantly impact model performance and introduce bias, making consistent preprocessing crucial. 
# 
# To address this, the following steps are undertaken:
# 
# 1. **Filter Columns in `mydata`:** Identify and retain columns in `mydata` where the proportion of missing values is below a specified threshold. This step removes columns with excessive missing data that could skew analysis or model training.
# 
# 2. **Apply Identified Columns to Other Datasets:** Ensure that `testset` and `extval_data` are aligned with `mydata` by selecting only the columns present in the filtered `mydata`. This maintains consistency across datasets, which is essential for reliable model evaluation and comparison.
# 
# 3. **Filter Rows in All Datasets:** After aligning columns, filter out rows from all datasets where the proportion of missing values exceeds the threshold. This step ensures that all datasets have comparable completeness, supporting fair and accurate modeling.
# 
# By following this approach, all datasets are harmonized with respect to both columns and rows, ensuring consistency and reducing potential bias from missing data.

# %%
def filter_columns(df, threshold=0.90):
    """
    Filter out columns with missingness greater than the threshold.
    
    The `filter_columns` function filters out columns in a DataFrame based on the presence of missing values. It calculates the mean missingness ratio for each column, sets a threshold to determine which columns to keep, and returns a new DataFrame with only the specified columns.

    1. Calculate the mean number of missing values per column.
    2. Identify columns where the mean missingness is less than or equal to the specified threshold (default=0.90).
    3. Return a new DataFrame containing only the selected columns, effectively removing columns with high missingness rates.

    ## Parameters

    * `df`: input DataFrame
    * `threshold` (optional): maximum allowed missingness ratio (default=0.90)

    ## Returns

    * A new DataFrame with filtered columns.
    * The number of remaining columns is less than or equal to the input DataFrame's original column count by at least 1.

    Note: This function does not modify the original DataFrame, but instead returns a new one with the filtered columns.
    """
    # Calculate the missingness ratio for each column
    column_missingness = df.isnull().mean(axis=0)
    # Identify columns to keep
    columns_to_keep = column_missingness[column_missingness <= threshold].index
    return df[columns_to_keep]

def filter_rows(df, threshold=0.90):
    """
    Filter out rows with missingness greater than the threshold.

    The `filter_rows` function filters out rows in a DataFrame based on the presence of missing values. It calculates the mean number of missing values per row, sets a threshold to determine which rows to keep, and returns a new DataFrame with only the specified rows.

    1. Calculate the mean number of missing values per row.
    2. Identify rows where the mean missingness is less than or equal to the specified threshold (default=0.90).
    3. Return a new DataFrame containing only the selected rows, effectively removing rows with high missingness rates.

    ## Parameters

    * `df`: input DataFrame
    * `threshold` (optional): maximum allowed missingness ratio (default=0.90)

    ## Returns

    * A new DataFrame with filtered rows.
    * The number of remaining rows is less than or equal to the original DataFrame's row count by at least 1.

    Note: This function does not modify the original DataFrame, but instead returns a new one with the filtered rows.
    """
    # Calculate the missingness ratio for each row
    row_missingness = df.isnull().mean(axis=1)
    # Keep rows with missingness less than or equal to the threshold
    return df[row_missingness <= threshold]

# Apply the filtering to `mydata`
if exclude_highly_missing_columns:
    mydata = filter_columns(mydata, threshold=column_threshold)

# Apply column filtering to `testset` and `extval_data` using columns identified from `mydata`
if exclude_highly_missing_columns:
    testset = testset[mydata.columns]
    if external_val:
        extval_data = extval_data[mydata.columns]

# Apply row filtering to all datasets
if exclude_highly_missing_rows:
    mydata = filter_rows(mydata, threshold=row_threshold)
    testset = filter_rows(testset, threshold=row_threshold)
    if external_val:
        extval_data = filter_rows(extval_data, threshold=row_threshold)


# %% [markdown]
# #### Feature selection (optional but recommended if the dataset is high dimensional, e.g >100 features)
# 
# Minimum Redundancy Maximum Relevance (mRMR) is one of the most popular algorithms for feature selection. For more information on its implementation see https://github.com/smazzanti/mrmr.
# 
# Reference: 
# F. Long, H. Peng and C. Ding, "Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy" in IEEE Transactions on Pattern Analysis & Machine Intelligence, vol. 27, no. 08, pp. 1226-1238, 2005.
# doi: 10.1109/TPAMI.2005.159

# %%
if feat_sel: # feature selection
    # Separate features and outcome variable
    X = mydata.drop(columns=[outcome_var])
    y = mydata[outcome_var]
    
    numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = X.select_dtypes(include=['category']).columns
    
    skf = StratifiedKFold(n_splits=cv_folds, random_state=SEED, shuffle=True)
    # Initialize list to store selected features for each fold
    selected_features_per_fold = []
    selected_features_fold = []
    # all_selected_features = set()
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        # for details see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
        if scale_data:
            robust_scaler = RobustScaler().fit(X_train_fold[numerical_columns]) 
            
            # Use the RobustScaler to scale the numerical features
            X_train_fold[numerical_columns] = robust_scaler.fit_transform(X_train_fold[numerical_columns])

        # Impute missing values in continuous features of X_train using KNN
        nn = int(np.sqrt(X_train_fold.shape[0])) # from Devroye, L., Györfi, L., & Lugosi, G. (1996). A Probabilistic Theory of Pattern Recognition. Springer. https://doi.org/10.1007/978-1-4612-0711-5.
        cont_imputer = KNNImputer(n_neighbors=nn, weights = 'distance', keep_empty_features = True)  
        
        X_train_fold_filled = pd.DataFrame(cont_imputer.fit_transform(X_train_fold[numerical_columns]), columns=numerical_columns, index=X_train_fold.index)
        
        # Combine the categorical features with the normalized continuous features for the training set and the test set
        X_train_fold = pd.concat([X_train_fold[categorical_columns], X_train_fold_filled], axis=1)
        
        # Replace categorical features with integers using LabelEncoder
        label_encoders = {}
        for col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            X_train_fold[col] = label_encoders[col].fit_transform(X_train_fold[col])

        # Select top num_features_sel features using mRMR
        selected_features_fold = mrmr_classif(X=X_train_fold, y=y_train_fold, K=num_features_sel)

        # Append selected features for this fold to the list
        selected_features_per_fold.append(selected_features_fold)

    # Find features common across all folds
    selected_features = set(selected_features_per_fold[0]).intersection(*selected_features_per_fold[1:])
    
    # # Convert the set to a list if needed
    selected_features = list(selected_features)
    print(f'Final union of selected features across all folds: {selected_features}')

    # Remove categorical features that are not selected
    # categorical_vars = [feat_name for feat_name in categorical_vars if feat_name in selected_features]
    cat_features = [feat_name for feat_name in cat_features if feat_name in selected_features]
    
    print("Selected Features:")
    print(selected_features)

    mydata = mydata[selected_features + [outcome_var]]
    testset = testset[selected_features + [outcome_var]]
    if external_val:
        extval_data = extval_data[selected_features + [outcome_var]]
    
    # Sort columns alphabetically (column order can affect the reproducibility of some estimators)
    mydata = mydata.reindex(sorted(mydata.columns), axis=1)
    testset = testset.reindex(sorted(testset.columns), axis=1)
    if external_val:
        extval_data = extval_data.reindex(sorted(extval_data.columns), axis=1)

# %% [markdown]
# #### Cross correlation of variables
# 
# Here the correlation coefficients between every pair of variables are calculated and presented as a heatmap. It also includes the outcome variable.
# method: Spearman rank-order correlation

# %%
# Replace NaN with median for each class
mydata_filled = mydata.copy()
# One-hot encode categorical features
mydata_filled = pd.get_dummies(mydata_filled, drop_first= True)
for column in mydata.columns:
    if mydata[column].isna().any():
        mydata_filled[column] = mydata[column].fillna(mydata.groupby(outcome_var)[column].transform('median'))

# Now, impute any remaining NaN values with the median of the entire column
mydata_filled.fillna(mydata_filled.median(), inplace=True)

# Calculate Spearman rank-order correlation
correlation_matrix = mydata_filled.corr(method='spearman')

# Find pairs of features with NaN values
nan_pairs = []
for col in correlation_matrix.columns:
    nan_in_col = correlation_matrix[col].isna()
    nan_pairs.extend([(col, row) for row, val in nan_in_col.items() if val])

# Replace NaN values with 0
correlation_matrix.fillna(0, inplace=True)

# Print names of pairs with NaN values
if nan_pairs:
    print("Pairs of features with undefined correlation values:")
    for pair in nan_pairs:
        print(pair)
else:
    print("No pairs of features had undefined correlation values.")


# Calculate figure size based on the number of features
num_features = correlation_matrix.shape[0]

height = round(np.max([10, np.log(num_features)])) 
# Ensure height does not exceed the maximum allowed dimension
max_height = 65535 / 72  # Convert pixels to inches
if height > max_height:
    height = max_height

# Create the clustermap
g = sns.clustermap(correlation_matrix,
                   cmap="viridis",
                   figsize=(height, height),
                   cbar_pos=(0.05, 0.95, 0.9, 0.05),
                   method="ward"
                   )

# Create a mask to hide the upper triangle
mask = np.triu(np.ones_like(correlation_matrix))

# Apply the mask to the heatmap
values = g.ax_heatmap.collections[0].get_array().reshape(correlation_matrix.shape)
new_values = np.ma.array(values, mask=mask)
g.ax_heatmap.collections[0].set_array(new_values)

# Adjust x-axis and y-axis ticks
g.ax_heatmap.set_xticks(np.arange(correlation_matrix.shape[0]) + 0.5, minor=False)
g.ax_heatmap.set_yticks(np.arange(correlation_matrix.shape[1]) + 0.5, minor=False)
g.ax_heatmap.set_xticklabels(correlation_matrix.index, fontsize=8)
g.ax_heatmap.set_yticklabels(correlation_matrix.index, fontsize=8)

g.ax_heatmap.set_facecolor('white')

# display grid lines
g.ax_heatmap.grid(which='both', color = "grey")

# modify grid lines
g.ax_heatmap.grid(which='minor', alpha=0.1)
g.ax_heatmap.grid(which='major', alpha=0.2)

# Hide the x-axis dendrogram
g.ax_col_dendrogram.set_visible(False)
g.savefig("feature_cor_clustermap.tif", bbox_inches='tight')
plt.show()


# %% [markdown]
# ### Sample size assessment 
# 
# The following script provides an analysis of dataset characteristics, including class imbalance, dataset size, and sample distribution per class, to determine the suitability of hyperparameter tuning. 
# 

# %%
def is_hyperparameter_tuning_suitable(data, outcome_var, train_size=train_size_perc, n_splits_outer=cv_folds, n_splits_inner=cv_folds_hptuning):
    """
    Checks whether hyperparameter tuning is suitable for a given dataset.

    ## Parameters:
        data (pandas DataFrame): The input dataset.
        outcome_var (str): The name of the column containing the target variable.
        train_size (float or int, optional): The proportion of samples to use for training. Defaults to 0.7.
        n_splits_outer (int, optional): The number of folds for outer cross-validation. Defaults to 5.
        n_splits_inner (int, optional): The number of folds for inner cross-validation. Defaults to 5.

    ## Returns
        None
    """
    # Extract features and target variable
    X = data.drop(columns=[outcome_var])
    y = data[outcome_var]
    y = y.replace({class_1: 1, class_0: 0})
        
    # Calculate number of samples used in model training
    n_train_samples = int(len(X) * train_size)
    
    # Print the number of samples used in model training
    print(f"Number of samples used in model training: {n_train_samples}")
  
    # Define cross-validation strategy for outer loop
    cv_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=SEED)
    
    # Initialize arrays to store number of samples per class per fold for outer CV
    n_samples_per_class_outer = np.zeros((n_splits_outer, 2))
    n_samples_per_class_inner = []
    
    # Iterate over outer folds
    for i, (train_index, _) in enumerate(cv_outer.split(X, y)):
        # Get class distribution in the training fold for outer CV
        y_train_fold_outer = y.iloc[train_index]
        class_counts_outer = np.bincount(y_train_fold_outer)
        n_samples_per_class_outer[i, :] = class_counts_outer
        
        # Define cross-validation strategy for inner loop
        cv_inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=SEED)
        
        # Initialize array to store number of samples per class per fold for inner CV
        n_samples_per_class_inner_fold = np.zeros((n_splits_inner, 2))
        
        # Iterate over inner folds
        for j, (train_index_inner, _) in enumerate(cv_inner.split(X.iloc[train_index], y.iloc[train_index])):
            # Get class distribution in the testing fold for inner CV
            y_train_fold_inner = y.iloc[train_index].iloc[train_index_inner]
            class_counts_inner = np.bincount(y_train_fold_inner)
            n_samples_per_class_inner_fold[j, :] = class_counts_inner
        
        # Append the array for inner CV to the list
        n_samples_per_class_inner.append(n_samples_per_class_inner_fold)
    
    # Convert the list of arrays to a numpy array for inner CV
    n_samples_per_class_inner = np.array(n_samples_per_class_inner)
            
    # Print mean and standard deviation of samples per class per fold for outer CV
    print("Samples per class per fold for outer CV:")
    print(n_samples_per_class_outer)
    
    # Print mean and standard deviation of samples per class per fold for inner CV
    print("Samples per class per fold for inner CV:")
    print(n_samples_per_class_inner)
    
    # Check if the number of samples per class in the inner CV is for example less than 10 (10 is chosen arbitrarily here)
    if np.any(n_samples_per_class_inner < 10):
        print("Warning: Number of samples per class in the inner cross-validation is less than 10. Hyperparameter tuning may not be suitable.")
    
    # Combine outer and inner CV samples per class matrices
    combined_samples_per_class = np.concatenate([n_samples_per_class_outer[:, np.newaxis, :], n_samples_per_class_inner], axis=1)

    # Plot heatmap of samples per class per fold
    plot_samples_per_class_per_fold(combined_samples_per_class)

def plot_samples_per_class_per_fold(samples_per_class_per_fold):
    """
    Plots a heatmap of the samples per class per fold.

    ## Parameters:
        samples_per_class_per_fold (numpy array): A 3D array where each element represents the number of samples for a particular class and fold combination.

    ## Returns
        None
    """
    fig, axes = plt.subplots(1, samples_per_class_per_fold.shape[0], figsize=(3 * samples_per_class_per_fold.shape[0], 6), sharey=True)
    for i in range(samples_per_class_per_fold.shape[0]):
        ax = axes[i]
        im = ax.imshow(samples_per_class_per_fold[i], cmap='coolwarm', interpolation='nearest')

        # Add color bar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Number of samples')

        # Set axis labels and title
        ax.set_title(f'Fold {i+1}')
        ax.set_xlabel('Classes')
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(['Class 0', 'Class 1'])
        ax.set_ylabel('Cross validation folds')
        ax.set_yticks(np.arange(6))
        ax.set_yticklabels(['Outer CV'] + [f'Inner CV {j+1}' for j in range(5)])
        # We change the fontsize of minor ticks label 
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        # Loop over data dimensions and create text annotations
        for j in range(samples_per_class_per_fold.shape[1]):
            for k in range(samples_per_class_per_fold.shape[2]):
                text = ax.text(k, j, f'{samples_per_class_per_fold[i, j, k]:.0f}',
                               ha="center", va="center", color="black")

    fig.suptitle('Samples per class per fold', fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # it shows how many samples are expected to be available for the cross validation and hyperparameter tuning
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 8
    fig.savefig("Samples_CV_map.tif", bbox_inches='tight') 
    plt.show()

is_hyperparameter_tuning_suitable(mydata, outcome_var)


# %%
# in case of applying oversampling
if oversampling:
    # Define your features and outcome variable
    X_train = mydata.drop(outcome_var, axis=1)
    y_train = mydata[outcome_var]

    # Initialize RandomOverSampler to oversample the minority class
    random_oversampler = RandomOverSampler(sampling_strategy='auto', random_state=SEED)

    # Oversample the minority class in the training set
    X_train_resampled, y_train_resampled = random_oversampler.fit_resample(X_train, y_train)
    # Concatenate resampled features and outcome variable back into a dataframe
    mydata = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train.columns), 
                            pd.Series(y_train_resampled, name=outcome_var)], axis=1)


# %%
mydata.shape

# %% [markdown]
# ##### Statistical comparision of the training and test sets

# %%
if data_split:

    # Define a function to check statistical difference for numerical variables
    def check_numerical_difference(train_data, test_data):
        '''
        check_numerical_difference compares statistical differences between numerical variables in training and testing datasets.

        This function takes two dataframes (train_data and test_data) as input, identifies the numerical columns, and computes the Mann-Whitney U-statistic to determine if there are significant differences between the distributions of these numerical variables in the training and testing datasets. It returns a dictionary containing the results for each identified variable, including the statistic and p-value.

        ## Parameters
        train_data: The original training data.
        test_data: The test set obtained from train_test_split.
        
        ## Returns
        A dictionary containing the statistical difference results for each numerical variable.
        '''
        numerical_vars = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        results = {}
        for var in numerical_vars:
            train_values = train_data[var].dropna()  # Drop missing values
            test_values = test_data[var].dropna()    # Drop missing values
            if len(train_values) > 0 and len(test_values) > 0:
                stat, p = mannwhitneyu(train_values, test_values)
                results[var] = {'Statistic': stat, 'p-value': p}
        return results

    # Check statistical difference for numerical variables
    numerical_results = check_numerical_difference(mydata, testset)
    
    print("Statistical Difference for Numerical Variables:")
    for var, result in numerical_results.items():
        print(f"{var}: Statistic = {result['Statistic']}, p-value = {result['p-value']}")

    def check_categorical_difference(train_data, test_data):
        '''
        Compares categorical differences between training and testing datasets, 
        using the Chi-square test to determine if there are significant differences in distribution.

        ## Parameters
        train_data: The original training data.
        test_data: The test set obtained from train_test_split.

        ## Returns
        A dictionary containing the statistical difference results for each categorical variable.
        '''
        categorical_vars = train_data.select_dtypes(include=['category']).columns.tolist()
        results = {}
        for var in categorical_vars:
            # Ensure 'missing' is added to categories in both train and test datasets
            train_data[var] = train_data[var].cat.add_categories(['missing'])
            test_data[var] = test_data[var].cat.add_categories(['missing'])

            train_categories = set(train_data[var].cat.categories)
            test_categories = set(test_data[var].cat.categories)
            common_categories = train_categories.intersection(test_categories)
            
            if len(common_categories) > 0:
                train_counts = train_data[var].fillna('missing').value_counts()
                test_counts = test_data[var].fillna('missing').value_counts()
                
                # Create the contingency table manually
                contingency_table = pd.DataFrame(index=list(common_categories), columns=['Train', 'Test'])
                for category in common_categories:
                    contingency_table.loc[category, 'Train'] = train_counts.get(category, 0)
                    contingency_table.loc[category, 'Test'] = test_counts.get(category, 0)
                
                # Remove rows with all zeros to avoid sparse issues
                contingency_table = contingency_table[(contingency_table > 0).any(axis=1)]
                
                # Ensure contingency table has at least a 2x2 structure for Chi-square test
                if contingency_table.shape[0] >= 2:
                    try:
                        chi2, p, _, _ = chi2_contingency(contingency_table.fillna(0))
                        results[var] = {'Chi-square': chi2, 'p-value': p}
                    except ValueError as e:
                        print(f"Skipping variable {var} due to error: {e}")
        return results

    # 'mydata' is the original dataset and 'testset' is the test set obtained from train_test_split
    categorical_results = check_categorical_difference(mydata, testset)

    
    print("Statistical Difference for Categorical Variables (if available):")
    for var, result in categorical_results.items():
        print(f"{var}: Chi-square = {result['Chi-square']}, p-value = {result['p-value']}")


# %% [markdown]
# ### Data overview 
# 

# %% [markdown]
# #### Display the type of the variables (columns)

# %%
mydata.dtypes

# %%
mydata.shape

# %% [markdown]
# #### Check missing values

# %%
# Identify columns with missing values
columns_with_missing_values = mydata.columns[mydata.isnull().any()]
columns_with_missing_values

# %%
mydata[outcome_var].unique()

# %%
mydata.describe()

# %%
# to report missingness both for categorical and continuous variables, it saves the results in an excel file
def calculate_missingness(data, output_file='missingness_report.xlsx'):
    """
    Reports missingness in both categorical and continuous variables and saves the results to an Excel file.

    This function calculates the percentage of missing values for each column in the input data,
    corrects these percentages for categorical columns where 'missing' is a valid category,
    and computes the mean and standard deviation of the missingness across all columns.
    
    ## Parameters
        data (pandas.DataFrame): The input data containing both categorical and continuous variables.
        output_file (str, optional): The file path to save the results. Defaults to 'missingness_report.xlsx'.

    ## Returns
        None
    """
    df = data.copy()

    # Identify categorical variables
    categorical_variables = df.select_dtypes(include=['category']).columns

    # Create a dataframe to store missing counts for categorical variables
    missing_counts = pd.DataFrame(index=categorical_variables, columns=['Missing Count'])

    # Count missing values for each categorical variable
    for column in categorical_variables:
        missing_counts.loc[column, 'Missing Count'] = (df[column] == 'missing').sum()

    # Calculate the total number of missing values for each column
    missing_values = df.isnull().sum()

    # Divide by the total number of rows to get missing percentage
    total_rows = len(data)
    missing_percentage = (missing_values / total_rows) * 100

    # Correct missing percentages for categorical columns
    for column in categorical_variables:
        if column in missing_percentage.index:
            if missing_counts.loc[column, 'Missing Count'] > 0:  # Only adjust if missing categories exist
                missing_percentage[column] = (missing_counts.loc[column, 'Missing Count'] / total_rows) * 100

    # Round the percentages to two decimal points
    missing_percentage = missing_percentage.round(2)

    # Sort the percentages in ascending order
    missing_percentage = missing_percentage.sort_values(ascending=False)

    # Calculate the mean and standard deviation of the missingness
    mean_missingness = np.mean(missing_percentage)
    std_missingness = np.std(missing_percentage)

    # Print the results
    print("Missing Value Percentages:")
    print(missing_percentage)
    print("Mean ± Standard Deviation of Missingness: {:.2f} ± {:.2f}".format(mean_missingness, std_missingness))
    
    # Save results to an Excel file
    with pd.ExcelWriter(output_file) as writer:
        # Save the missing percentage
        missing_percentage.to_excel(writer, sheet_name='Missing Percentages')
        
        # Save mean and std as a separate sheet
        summary_df = pd.DataFrame({
            'Mean Missingness': [mean_missingness],
            'Std Missingness': [std_missingness]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)



# %%
calculate_missingness(data=mydata, output_file='missingness_trainingset.xlsx')


# %%
calculate_missingness(data=testset, output_file='missingness_testset.xlsx')


# %%
df1 = pd.DataFrame(mydata)
df2 = pd.DataFrame(testset)
if external_val:
    df3 = pd.DataFrame(extval_data)
# Function to create the summary table for a single dataset
def create_summary_table(dataframe, dataset_name):
    """
    Creates a summary table for a single dataset.
    
    This function generates a table that summarizes key statistics about each variable in the dataset,
    including numerical variables (median and quartiles) and categorical variables (categories, counts, and percentages).
    Additionally, it includes information on missing values and adds a column for the dataset name.

    ## Parameters
        dataframe (pd.DataFrame): The input DataFrame to generate summary statistics from.
        dataset_name (str): The name of the dataset being summarized.

    ## Returns
        pd.DataFrame: A new DataFrame containing the summary statistics.
    """
    summary_data = {'Variable': [], 'Value': []}

    for col in sorted(dataframe.columns):  # Sort variable names alphabetically
        summary_data['Variable'].append(col)
        summary_data['Value'].append('Variable Name')

        # For numerical variables - Median (lower quantile, higher quantile)
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            median = dataframe[col].median()
            q25 = dataframe[col].quantile(0.25)
            q75 = dataframe[col].quantile(0.75)
            summary_data['Variable'].append('')
            summary_data['Value'].append(f'{median:.2f} ({q25:.2f}, {q75:.2f})')

        # For categorical variables - Categories, Counts, and Percentages
        elif pd.api.types.is_categorical_dtype(dataframe[col]):
            categories = dataframe[col].value_counts()
            total_count = len(dataframe[col])
            summary_data['Variable'].extend(['', ''])
            summary_data['Value'].extend(['Categories', 'Counts'])
            
            for category, count in categories.items():
                percentage = (count / total_count) * 100
                summary_data['Variable'].append(category)
                summary_data['Value'].append(f'{count} - {percentage:.2f}%')

        # Missing values for all variable types
        missing_count = dataframe[col].isnull().sum()
        missing_percentage = (missing_count / len(dataframe)) * 100
        summary_data['Variable'].append('')
        summary_data['Value'].append(f' {missing_percentage:.2f}%')

    # Add a column for the dataset name
    summary_data['Variable'].append('Dataset')
    summary_data['Value'].append(dataset_name)

    summary_df = pd.DataFrame(summary_data)
    return summary_df

summary_table1 = create_summary_table(df1, 'training data')
summary_table1.to_excel('summary_table_devset.xlsx', index=False)

if data_split:
    summary_table2 = create_summary_table(df2, 'test data')
    summary_table2.to_excel('summary_table_testset.xlsx', index=False)
if external_val:
    summary_table3 = create_summary_table(df3, 'external validation data')
    summary_table3.to_excel('summary_table_extvalset.xlsx', index=False)


# final summary table
print(summary_table1)

# %%
cat_features

# %%
# for details see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
if scale_data:
    # Specify the numerical features you want to scale
    numerical_columns = mydata.select_dtypes(include=['float64', 'int64']).columns
    
    robust_scaler = RobustScaler().fit(mydata[numerical_columns]) 
    
    # Use the RobustScaler to scale the numerical features
    mydata[numerical_columns] = robust_scaler.fit_transform(mydata[numerical_columns])
    testset[numerical_columns] = robust_scaler.fit_transform(testset[numerical_columns])
    if external_val:
        extval_data[numerical_columns] = robust_scaler.fit_transform(extval_data[numerical_columns])

# %% [markdown]
# ### Data imputation
# 
# Here we apply k-nearest neighbors (KNN) algorithm to impute missing values in continuous variables. This is done in fold-wise as in cross validation so that the informaiton from one fold does not leak to other folds. This means that the training data is split to a number of folds as the same as in cross validation and then the imputation is performed on the fold under test, for all folds. then they are merged back to recreate the training set with imputation. The test set and external datasets are also imputed based on the KNN algorithm.

# %%
# for details see https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html

# Separate features and outcome variable
X_train = mydata.drop(columns=[outcome_var])
y_train = mydata[outcome_var]

X_test = testset.drop(columns=[outcome_var])
y_test = testset[outcome_var]

numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = X_train.select_dtypes(include=['category','object']).columns

skf = StratifiedKFold(n_splits=cv_folds, random_state=SEED, shuffle=True)

# List to hold the imputed validation sets
imputed_train_data = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Impute missing values in continuous features of X_train using KNN
    nn = int(np.sqrt(X_train_fold.shape[0]))
    random.seed(SEED)
    cont_imputer = KNNImputer(n_neighbors=nn, weights = 'distance', keep_empty_features = True)  
    cont_imputer.fit(X_train_fold[numerical_columns])  # Fit the imputer on the training portion of the fold

    X_val_fold_filled = pd.DataFrame(cont_imputer.transform(X_val_fold[numerical_columns]), columns=numerical_columns, index=X_val_fold.index)
    
    # Combine the categorical features with the imputed continuous features for the validation set
    X_val_fold_combined = pd.concat([X_val_fold[categorical_columns], X_val_fold_filled], axis=1)

    # Combine features and target for the validation fold and append to the list
    imputed_train_data.append(pd.concat([X_val_fold_combined, y_val_fold], axis=1))

mydata_imputed = pd.concat(imputed_train_data) # this is used for cross validation

nn = int(np.sqrt(X_train.shape[0]))
random.seed(SEED)
cont_imputer = KNNImputer(n_neighbors=5, weights = 'distance', keep_empty_features = True)
X_train_filled = pd.DataFrame(cont_imputer.fit_transform(X_train[numerical_columns]), columns=numerical_columns, index=X_train.index)
mydata_imputed_nocv = pd.concat([X_train[categorical_columns], X_train_filled, y_train], axis=1) # this is used for training the model to be tested on the test set (after cross-validation)

X_test_filled = pd.DataFrame(cont_imputer.transform(X_test[numerical_columns]), columns=numerical_columns, index=X_test.index)
# Combine the categorical features with the normalized continuous features for the external validation set
testset_imputed = pd.concat([X_test[categorical_columns], X_test_filled, y_test], axis=1)
    

if external_val:
    X_extval_data = extval_data.drop(outcome_var, axis=1)
    y_extval_data = extval_data[outcome_var]
    X_extval_data_filled_cont = pd.DataFrame(cont_imputer.transform(X_extval_data[numerical_columns]), columns=numerical_columns, index=y_extval_data.index)
    # Combine the categorical features with the normalized continuous features for the external validation set
    extval_data_imputed = pd.concat([X_extval_data[categorical_columns], X_extval_data_filled_cont, y_extval_data], axis=1)
    

# %%
# Reorder the rows of mydata_imputed to match the order of rows in mydata
mydata_imputed = mydata_imputed.reindex(mydata.index)
mydata_imputed_nocv = mydata_imputed_nocv.reindex(mydata.index)

# %%
mydata_imputed.dtypes

# %%
X_train_imputed = mydata_imputed.drop(outcome_var, axis=1)
X_train_imputed_nocv = mydata_imputed_nocv.drop(outcome_var, axis=1)
X_test_imputed = testset_imputed.drop(outcome_var, axis=1)

# Perform one-hot encoding
if external_val: # if there is an external validation set (in addition to the test set)
    X_extval_data_imputed = extval_data_imputed.drop(outcome_var, axis=1)
    
    combined_imputed = pd.concat([X_train_imputed, X_test_imputed, X_train_imputed_nocv, X_extval_data_imputed], keys=['train', 'test', 'train_nocv','ext_val'])
    combined_encoded = pd.get_dummies(combined_imputed, drop_first=True)
    X_train_OHE = combined_encoded.xs('train')
    X_test_OHE = combined_encoded.xs('test')
    X_train_OHE_nocv = combined_encoded.xs('train_nocv')
    X_extval_data_OHE = combined_encoded.xs('ext_val')
    extval_data_imputed_OHE = pd.concat([X_extval_data_OHE, y_extval_data], axis=1)
else: # no external validation
    combined_imputed = pd.concat([X_train_imputed, X_test_imputed, X_train_imputed_nocv], keys=['train', 'test', 'train_nocv'])
    combined_encoded = pd.get_dummies(combined_imputed, drop_first=True)
    X_train_OHE = combined_encoded.xs('train')
    X_test_OHE = combined_encoded.xs('test')
    X_train_OHE_nocv = combined_encoded.xs('train_nocv')
    
mydata_imputed_OHE = pd.concat([X_train_OHE, y_train], axis=1) # for cross validation - imputed on folds of the training set
mydata_imputed_OHE_nocv = pd.concat([X_train_OHE_nocv, y_train], axis=1) # for external validation - imputed on the entire training set
testset_imputed_OHE = pd.concat([X_test_OHE, y_test], axis=1)

  
# Display the resulting dataframe
print(X_train_OHE.head())

# %% [markdown]
# ### Correlation analysis
# 
# Here we use univariable correlation based on point-biserial correlation and mutual informaiton between the variables (features) and the outcome variable. This is based on one-hot encoded and imputed data (only development/training set).

# %%
df_imputed = mydata_imputed_OHE.copy()
# Convert 'outcome_var' to numerical variable
df_imputed[outcome_var] = df_imputed[outcome_var].replace({class_1: 1, class_0: 0})

# Generate 1000 subsamples of df_imputed
def generate_subsample(df, seed):
    """
    Generates a subsample of the dataset using random sampling.

    ## Parameters
        df (DataFrame): Original dataset.
        seed (int): Random seed for sampling.

    ## Returns
        DataFrame: Subsample of the dataset.
    """
    rng = np.random.RandomState(seed)  # Set the random state
    return df.sample(frac=1, replace=True, random_state=rng)

# Calculate point biserial correlation for each variable against the target
def calculate_biserial_corr(subsample, outcome_var):
    """
    Calculates point-biserial correlation for each variable in the subsample against the target.

    ## Parameters
        subsample (DataFrame): Subsample of the dataset.
        outcome_var (str): Name of the target variable.

    ## Returns
        corr_values (dict): Dictionary containing correlation values for each variable.
    """
    corr_values = {}
    for col in subsample.columns:
        if col != outcome_var:
            corr_values[col] = pointbiserialr(subsample[col], subsample[outcome_var])[0]
    return corr_values

# Generate subsamples in parallel
num_iterations = 1000
seeds = np.random.randint(0, 10000, size=num_iterations)  # Generate unique seeds for each iteration

subsamples = Parallel(n_jobs=n_cpu_model_training, backend='loky')(
    delayed(generate_subsample)(df_imputed, seed) for seed in seeds
)

# Calculate point biserial correlation for each subsample
biserial_corr_values = Parallel(n_jobs=n_cpu_model_training, backend='loky')(
    delayed(calculate_biserial_corr)(subsample, outcome_var) for subsample in subsamples
)

corr_df = pd.DataFrame(biserial_corr_values)


# %%
# Calculate the lower and upper quantiles of point-biserial correlation for each feature


# Calculate lower quartile (25th percentile) excluding NaN values
lower_quantile_corr = np.nanpercentile(corr_df, 25, axis=0)

# Calculate median (50th percentile) excluding NaN values
median_corr = np.nanpercentile(corr_df, 50, axis=0)

# Calculate upper quartile (75th percentile) excluding NaN values
upper_quantile_corr = np.nanpercentile(corr_df, 75, axis=0)

# Filter features based on quantiles
significant_features = [feature for feature, lower, upper in zip(corr_df.columns, lower_quantile_corr, upper_quantile_corr) if lower > 0 or upper < 0]

# Filter the original dataframe to include only significant features
df_imputed_filtered = df_imputed[significant_features]

# Create a DataFrame with feature names, lower and upper quantiles of point-biserial correlation
corr_summary_df = pd.DataFrame({'Feature': df_imputed.drop(outcome_var, axis=1).columns,
                                'median_Corr': median_corr,
                                'Lower_Quantile_Corr': lower_quantile_corr,
                                'Upper_Quantile_Corr': upper_quantile_corr})
corr_summary_df
corr_summary_df.to_excel('pb_corr_summary_df.xlsx', index=False)

# %%
# Sort DataFrame by median correlation for better visualization
corr_summary_df = corr_summary_df.sort_values(by='median_Corr', ascending=True)
# corr_summary_df = corr_summary_df[~(corr_summary_df["median_Corr"] == 0) & ~(corr_summary_df["median_Corr"].isna())]

num_rows = corr_summary_df.shape[0]
# Set the fixed width
width = 6
# Calculate the height based on the number of rows
height = num_rows / 5  # Assuming each row takes about 0.2 inches
# Ensure height does not exceed the maximum allowed dimension
max_height = 65535 / 72  # Convert pixels to inches
if height > max_height:
    height = max_height
# Set the figure size
plt.figure(figsize=(width, height))
# Plot error bars for all features
plt.errorbar(x=corr_summary_df['median_Corr'], y=corr_summary_df['Feature'],
             xerr=[corr_summary_df['median_Corr'] - corr_summary_df['Lower_Quantile_Corr'], 
                   corr_summary_df['Upper_Quantile_Corr'] - corr_summary_df['median_Corr']],
             fmt='o', color='black', capsize=5, label='IQR')

# Add a vertical dotted line at x=0
plt.axvline(x=0, linestyle='--', color='grey', alpha=0.5)

plt.title('Median and quantile correlation coefficients for features\nbased on random subsamples of the development set\nwith replication across 1000 iterations', fontsize=10)
plt.xlabel('PB Correlation Values', fontsize=8)
plt.ylabel('Feature', fontsize=8) 
plt.legend()
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=8) 
plt.gca().legend(fontsize="medium")
plt.gca().set_facecolor('white')
# display grid lines
plt.grid(which='both', color="grey")
plt.grid(which='minor', alpha=0.1)
plt.grid(which='major', alpha=0.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.savefig("pointbiserial.tif", bbox_inches='tight')
plt.show()


# %%
significant_features

# %%
df_imputed = mydata_imputed_OHE.copy()
# Convert 'outcome_var' to numerical variable
df_imputed[outcome_var] = df_imputed[outcome_var].replace({class_1: 1, class_0: 0})

# Calculate mutual information for each variable against the target
def calculate_mutual_info(subsample, outcome_var):
    """
    Calculates mutual information between each feature in the subsample and the target variable.

    ## Parameters
        subsample (DataFrame): Subsample of the dataset.
        outcome_var (str): Name of the target variable.

    ## Returns
        mi_values (dict): Dictionary containing mutual information values for each feature.
    """
    mi_values = {}
    for col in subsample.columns:
        if col != outcome_var:
            mi_values[col] = mutual_info_classif(subsample[[col]], subsample[outcome_var])[0]
    return mi_values

np.random.seed(SEED)
num_iterations = 1000
seeds = randint(0, 10000, size=num_iterations)  # Generate unique seeds for each iteration

subsamples = Parallel(n_jobs=n_cpu_model_training, backend='loky')(
    delayed(generate_subsample)(df_imputed, seed) for seed in seeds
)

mi_values = Parallel(n_jobs=n_cpu_model_training, backend='loky')(
    delayed(calculate_mutual_info)(subsample, outcome_var) for subsample in subsamples
)

mi_df = pd.DataFrame(mi_values)


# %%
# Calculate the lower and upper quantiles of point-biserial correlation for each feature
lower_quantile_mi = np.percentile(mi_df, 25, axis=0)
median_mi = np.percentile(mi_df, 50, axis=0)
upper_quantile_mi = np.percentile(mi_df, 75, axis=0)

# Filter features based on quantiles
significant_features = [feature for feature, lower in zip(mi_df.columns, lower_quantile_mi) if lower != 0]

# Filter the original dataframe to include only significant features
df_imputed_filtered = df_imputed[significant_features]

# Create a dataframe with feature names, lower and upper quantiles of point-biserial correlation
mi_summary_df = pd.DataFrame({'Feature': df_imputed.drop(outcome_var, axis=1).columns,
                                'median_MI': median_mi,
                                'Lower_Quantile_MI': lower_quantile_mi,
                                'Upper_Quantile_MI': upper_quantile_mi})
mi_summary_df
mi_summary_df.to_excel('mi_summary_df.xlsx', index=False)

# %%

# Mark significant features with a different color
mi_summary_df['Color'] = np.where(mi_summary_df['Feature'].isin(significant_features), 'significant', 'not significant')

# Sort dataframe by median correlation for better visualization
mi_summary_df = mi_summary_df.sort_values(by='median_MI', ascending=True)

# Plotting
plt.figure(figsize=(width, height))

# Plot error bars for all features
plt.errorbar(x=mi_summary_df['median_MI'], y=mi_summary_df['Feature'],
             xerr=[mi_summary_df['median_MI'] - mi_summary_df['Lower_Quantile_MI'], 
                   mi_summary_df['Upper_Quantile_MI'] - mi_summary_df['median_MI']],
             fmt='o', color='black', capsize=5, label='IQR')

# Add a vertical dotted line at x=0
plt.axvline(x=0, linestyle='--', color='grey', alpha=0.5)

plt.title('Median and quantile mutual informaiton for features\nbased on random subsamples of the development set\nwith replication across 1000 iterations', fontsize=10)
plt.xlabel('mutual information', fontsize=8)
plt.ylabel('Feature', fontsize=8) 
plt.legend()
plt.grid(True)
plt.gca().tick_params(axis='both', labelsize=8) 
plt.gca().legend(fontsize="medium")
plt.gca().set_facecolor('white')
# display grid lines
plt.grid(which='both', color="grey")
plt.grid(which='minor', alpha=0.1)
plt.grid(which='major', alpha=0.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.savefig("mutual_information.tif", bbox_inches='tight')
plt.show()


# %%
mydata.shape

# %% [markdown]
# #### Checking the outcome variable and its categories (binary)

# %%
print(mydata[outcome_var].unique())
print(mydata_imputed[outcome_var].unique())
print(mydata_imputed_OHE[outcome_var].unique())
print(mydata_imputed_nocv[outcome_var].unique())
print(mydata_imputed_OHE_nocv[outcome_var].unique())

# %%
mydata[outcome_var] = mydata[outcome_var].map({class_0: False, class_1: True}).astype(bool)
mydata_imputed[outcome_var] = mydata_imputed[outcome_var].map({class_0: False, class_1: True}).astype(bool)
mydata_imputed_nocv[outcome_var] = mydata_imputed_nocv[outcome_var].map({class_0: False, class_1: True}).astype(bool)
mydata_imputed_OHE[outcome_var] = mydata_imputed_OHE[outcome_var].map({class_0: False, class_1: True}).astype(bool)
mydata_imputed_OHE_nocv[outcome_var] = mydata_imputed_OHE_nocv[outcome_var].map({class_0: False, class_1: True}).astype(bool)
testset_imputed[outcome_var] = testset_imputed[outcome_var].map({class_0: False, class_1: True}).astype(bool)
if external_val:
    extval_data_imputed[outcome_var] = extval_data_imputed[outcome_var].map({class_0: False, class_1: True}).astype(bool)


# %%
print(mydata[outcome_var].unique())
print(mydata_imputed[outcome_var].unique())
print(mydata_imputed_OHE[outcome_var].unique())
print(mydata_imputed_nocv[outcome_var].unique())
print(mydata_imputed_OHE_nocv[outcome_var].unique())

# %%
mydata.dtypes

# %% [markdown]
# ### Data visualization
# 
# Here we plot all variables of the dataset both categorical and numerical ones in box plots that represents the distributions of the variables. Here you can inspect if there is any outlier or data anomally (e.g., values outside range).

# %%

# Select continuous variables from the dataframe
continuous_vars = mydata_imputed.select_dtypes(include=['float64', 'int64'])
# select categorical variables
categorical_vars = mydata_imputed.select_dtypes(include=['category',"object","bool"])
# get a copy of the outcome variable
outcome_variable = mydata_imputed[outcome_var].copy()

# Calculate the number of rows and columns for subplots
num_continuous_vars = len(continuous_vars.columns)
num_categorical_vars = len(categorical_vars.columns)
num_cols_to_plot = 3
num_rows = (num_continuous_vars + num_categorical_vars + num_cols_to_plot - 1) // num_cols_to_plot + 1  # Adjust the number of rows based on the number of variables

mapping = {True: class_1, False: class_0}
outcome_variable_mapped = outcome_variable.map(mapping)

# Create subplots for continuous variables
fig, axes = plt.subplots(num_rows, num_cols_to_plot, figsize=(12, num_rows * 2)) 

# Iterate over continuous variables
for i, column in enumerate(continuous_vars.columns):
    # Determine the subplot indices
    row_idx = i // num_cols_to_plot
    col_idx = i % num_cols_to_plot

    # Check if subplot index is within the bounds of axes
    if row_idx < num_rows:
        # Get the axis for the current subplot
        ax = axes[row_idx, col_idx]

        # Iterate over each outcome category
        for outcome_category, ax_offset in zip(outcome_variable.unique(), [-0.2, 0.2]):
            # Filter the data for the current outcome category
            filtered_data = continuous_vars[outcome_variable == outcome_category][column]

            # Create a box plot for the current outcome category
            positions = np.array([1 + ax_offset])
            ax.boxplot(filtered_data.dropna(), positions=positions, widths=0.3, vert=False)  # Vert=False for horizontal box plots

        ax.set_title(f'{column}', fontsize=8)
        ax.set_yticks([1 - ax_offset, 1 + ax_offset])
        # ax.set_yticklabels(outcome_variable.unique(), fontsize=8)
        ax.set_yticklabels(outcome_variable_mapped.unique(), fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize="medium")
        ax.set_facecolor('white')
        
        # show both grid lines
        ax.grid(which='both', color = "grey")

        # modify grid lines:
        ax.grid(which='minor', alpha=0.1)
        ax.grid(which='major', alpha=0.2)

# Iterate over categorical variables
for i, column in enumerate(categorical_vars.columns):
    # Determine the subplot indices
    row_idx = (i + num_continuous_vars) // num_cols_to_plot
    col_idx = (i + num_continuous_vars) % num_cols_to_plot

    # Check if subplot index is within the bounds of axes
    if row_idx < num_rows:
        # Get the axis for the current subplot
        ax = axes[row_idx, col_idx]

        # Normalize the counts for the current categorical variable stratified by outcome variable
        category_counts = categorical_vars.groupby(outcome_variable)[column].value_counts(normalize=True).unstack()
        category_counts.plot(kind='barh', ax=ax)

        # Set the title with the feature name
        ax.set_title(f'{column}', fontsize=8)
        ax.set_yticklabels(outcome_variable_mapped.unique(), fontsize=8)
        ax.set_ylabel(None)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize="medium")
        ax.set_facecolor('white')
        
        # display grid lines
        ax.grid(which='both', color = "grey")

        # modify grid lines
        ax.grid(which='minor', alpha=0.1)
        ax.grid(which='major', alpha=0.2)

# Remove any empty subplots at the end
if num_continuous_vars + num_categorical_vars < num_rows * num_cols_to_plot:
    for i in range(num_continuous_vars + num_categorical_vars, num_rows * num_cols_to_plot):
        fig.delaxes(axes.flatten()[i])

# Remove the subplot for outcome_var at the end
last_ax_index = num_continuous_vars + num_categorical_vars - 1
fig.delaxes(axes.flatten()[last_ax_index])

# Adjust the layout and spacing
plt.tight_layout()
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.savefig("data_distribution.tif", bbox_inches='tight')

# Show the plot
plt.show()


# %%
# Count the number of samples per class in devset
ymap = {True: class_1, False: class_0}
mydata_class_counts = mydata[outcome_var].replace(ymap).value_counts()

# Calculate the percentage of samples per class in devset
mydata_class_percentages = (mydata_class_counts / len(mydata)) * 100

# Count the number of samples per class in testset
testset_class_counts = testset[outcome_var].value_counts()

# Calculate the percentage of samples per class in testset
testset_class_percentages = (testset_class_counts / len(testset)) * 100

# summary of the number of samples per class and their percentages
if data_split:
    print("training set:")
    print(mydata_class_counts)
    print(mydata_class_percentages)
    print("\nTest Set:")
    print(testset_class_counts)
    print(testset_class_percentages)
else:
    print("whole dataset:")
    print(mydata_class_counts)
    print(mydata_class_percentages)

if external_val:
    # Count the number of samples per class in extval_data
    extval_data_class_counts = extval_data[outcome_var].value_counts()

    # Calculate the percentage of samples per class in extval_data
    extval_data_class_percentages = (extval_data_class_counts / len(extval_data)) * 100
    print("\nExternal validation set:")
    print(extval_data_class_counts)
    print(extval_data_class_percentages)


# %% [markdown]
# ##### Function to evaluate models and generate ROC curve, PR curve and confusion matrix

# %%

# Define a function for bootstrap sampling
def bootstrap_sample(data, n_samples):
    """
    Perform bootstrap sampling on the input data.

    ## Parameters:
    - data (array-like): Input data to be sampled.
    - n_samples (int): Number of samples to generate.

    ## Returns
    - indices (numpy array): Indices of the original data used for bootstrapping.
    - resampled_data (array-like): Resampled data with shape (n_samples, len(data)).
    """
    np.random.seed(SEED)
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    return data[indices]

def calculate_confidence_interval(metric_values, alpha=0.95):
    """
    Calculate the confidence interval for the given metric values.

    ## Parameters:
    - metric_values (array-like): Input metric values.
    - alpha (float, optional): Confidence level. Defaults to 0.95.

    ## Returns
    - lower_bound (float or numpy array): Lower bound of the confidence interval.
    - upper_bound (float or numpy array): Upper bound of the confidence interval.
    """
    # Filter out NaN values from metric_values
    non_nan_values = metric_values[~np.isnan(metric_values)]
    
    # Check if there are non-NaN values to calculate the confidence interval
    if len(non_nan_values) == 0:
        return np.nan, np.nan
    
    # Calculate confidence intervals for non-NaN values
    lower_bound = np.percentile(non_nan_values, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(non_nan_values, (1 + alpha) / 2 * 100)
    return lower_bound, upper_bound

def evaluate_and_plot_model(model, testset, y_test, filename, class_labels = class_labels_display, threshold=0.5, bootstrap_samples=1000, min_positive_instances=1):
    """
    Calculates and visualizes model performance using ROC curve, PR curve, and confusion matrix.
    
    ## Parameters
        y_test (np.ndarray): Ground truth labels for the test dataset.
        predictions_class (np.ndarray): Predicted labels for the test dataset.
        class_labels_display (list or tuple): List of unique class labels in the data for display purposes.
        threshold (float): Threshold value for model evaluation.
        filename (str): Output file name for visualization.

    ## Returns
        results_df (DataFrame): DataFrame containing model performance metrics.
        missclassified_samples (list): List of indices of samples that were misclassified by the model.
    """

    bootstrap_values = []

    for _ in range(bootstrap_samples):
        # Perform bootstrap sampling
        bootstrap_sample_indices = np.random.choice(len(testset), len(testset), replace=True)
        bootstrap_sample_testset = testset.iloc[bootstrap_sample_indices]
        bootstrap_sample_y_test = y_test.iloc[bootstrap_sample_indices]

        if isinstance(model, (cb.CatBoostClassifier, lgb.LGBMClassifier, GaussianNB,RandomForestClassifier, LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier)):
            predictions = model.predict_proba(bootstrap_sample_testset)[:, 1]
            # print(predictions)
        else:
            predictions = model.predict(bootstrap_sample_testset)

        predictions_class = [True if x >= threshold else False for x in predictions]

        # Check if the number of positive instances is below the threshold
        if np.sum(bootstrap_sample_y_test) < min_positive_instances:
            # Set metrics to NaN or another suitable value
            PPV, NPV, sensitivity, specificity, balanced_accuracy, MCC, roc_auc, pr_auc, brier_score, f1 = [np.nan] * 10
        else:
            cm = confusion_matrix(bootstrap_sample_y_test, predictions_class)
            tn, fp, fn, tp = cm.ravel()

            PPV = tp / (tp + fp)
            NPV = tn / (tn + fn)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            balanced_accuracy = (sensitivity + specificity) / 2
            MCC = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
            roc_auc = roc_auc_score(y_true=bootstrap_sample_y_test, y_score=predictions)
            brier_score = brier_score_loss(y_true=bootstrap_sample_y_test, y_prob=predictions, pos_label=True)
            precision, recall, _ = precision_recall_curve(y_true=bootstrap_sample_y_test, probas_pred=predictions, pos_label=True)
            pr_auc = metrics.auc(x=recall, y=precision)
            f1 = f1_score(bootstrap_sample_y_test, predictions_class)

        # Store the metric values for each bootstrap iteration
        bootstrap_values.append([PPV, NPV, sensitivity, specificity, balanced_accuracy, MCC, roc_auc, pr_auc, brier_score, f1])


    # Convert the list of metric values into a numpy array for easier manipulation
    bootstrap_values = np.array(bootstrap_values)

    # Calculate confidence intervals for each metric
    lower_bounds, upper_bounds = zip(*[calculate_confidence_interval(bootstrap_values[:, i]) for i in range(bootstrap_values.shape[1])])

    # Calculate the measures for the whole testset
    if np.sum(y_test) < min_positive_instances:
        # Set metrics to NaN or another suitable value
        PPV, NPV, sensitivity, specificity, balanced_accuracy, MCC, roc_auc, pr_auc, brier_score, f1 = [np.nan] * 10
    else:
        if isinstance(model, (cb.CatBoostClassifier, lgb.LGBMClassifier, GaussianNB, RandomForestClassifier, LogisticRegression, HistGradientBoostingClassifier)):
            predictions = model.predict_proba(testset)[:, 1]
            # print(predictions)
        else:
            predictions = model.predict(testset)

        predictions_class = [True if x >= threshold else False for x in predictions]

        cm = confusion_matrix(y_test, predictions_class)
        tn, fp, fn, tp = cm.ravel()

        PPV = tp / (tp + fp)
        NPV = tn / (tn + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_accuracy = (sensitivity + specificity) / 2
        MCC = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        roc_auc = roc_auc_score(y_true=y_test, y_score=predictions)
        brier_score = brier_score_loss(y_true=y_test, y_prob=predictions, pos_label=True)
        precision, recall, _ = precision_recall_curve(y_true=y_test, probas_pred=predictions, pos_label=True)
        pr_auc = metrics.auc(x=recall, y=precision)
        f1 = f1_score(y_test, predictions_class)

    # Convert the list of metric values into a numpy array for easier manipulation
    bootstrap_values = np.array(bootstrap_values)

    # Calculate confidence intervals for each metric
    lower_bounds, upper_bounds = zip(*[calculate_confidence_interval(bootstrap_values[:, i]) for i in range(bootstrap_values.shape[1])])

    # Calculate the measures for the whole testset
    results = {
        'Metric': ['PPV', 'NPV', 'Sensitivity', 'Specificity', 'Balanced Accuracy', 'MCC', 'ROCAUC', 'PRAUC', 'Brier Score', 'F1 Score'],
        'Value': [PPV, NPV, sensitivity, specificity, balanced_accuracy, MCC, roc_auc, pr_auc, brier_score, f1],
        'Lower Bound': lower_bounds,
        'Upper Bound': upper_bounds
    }

    results_df = pd.DataFrame(results)
    results_df['Value'] = results_df['Value'].round(2)
    results_df['Lower Bound'] = results_df['Lower Bound'].round(2)
    results_df['Upper Bound'] = results_df['Upper Bound'].round(2)
    print(results_df)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=True, drop_intermediate=False)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, predictions, pos_label=True)
    
    # Find missclassified samples
    predictions_class = [True if x >= threshold else False for x in predictions]
    missclassified_samples = y_test.index[np.where(y_test != predictions_class)[0]]

    # Finding the index closest to the custom threshold instead of 0.5
    threshold_index = (np.abs(thresholds - threshold)).argmin()
    threshold_custom = thresholds[threshold_index]
    tpr_custom = tpr[threshold_index]
    fpr_custom = fpr[threshold_index]

    pr_threshold_index = (np.abs(pr_thresholds - threshold)).argmin()
    pr_threshold_custom = pr_thresholds[pr_threshold_index]
    precision_custom = precision[pr_threshold_index]
    recall_custom = recall[pr_threshold_index]

    def display_confusion_matrix(y_true, y_pred, labels):
        """
        Displays a confusion matrix for model performance evaluation.

        ## Parameters
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.
            labels (list or tuple): List of unique class labels in the data.

        ## Returns
            None: The function modifies the specified axes object and displays the plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', ax=ax3, xticks_rotation='vertical', values_format='d')
        ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=8)
        ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=8)
        plt.xlabel('Predicted', fontsize=8)
        plt.ylabel('True', fontsize=8)
        ax3.legend(fontsize=8)
        ax3.invert_yaxis()
  
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    
    ax1.set_facecolor('white')
    # show both grid lines
    ax1.grid(which='both', color = "grey")
    # modify grid lines:
    ax1.grid(which='minor', alpha=0.1)
    ax1.grid(which='major', alpha=0.2)
    ax1.plot(fpr, tpr, color='blue', label='ROC AUC ≈ {:.2f}'.format(roc_auc))
    ax1.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=0.5, label='chance level')
    ax1.scatter(fpr_custom, tpr_custom, color='red', label=f'Threshold = {threshold_custom:.2f}', s=50, marker='x')
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, 1.1])
    ax1.set_xlabel('False Positive Rate', fontsize=8)
    ax1.set_ylabel('True Positive Rate', fontsize=8)
    ax1.set_title('ROC curve', fontsize=8)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)  

    

    chance_level_precision = np.sum(y_test) / len(y_test)
    ax2.set_facecolor('white')
    # show both grid lines
    ax2.grid(which='both', color = "grey")
    # modify grid lines:
    ax2.grid(which='minor', alpha=0.1)
    ax2.grid(which='major', alpha=0.2)
    ax2.plot(recall, precision, color='green', label='PR AUC ≈ {:.2f}'.format(pr_auc))
    ax2.scatter(recall_custom, precision_custom, color='orange', label=f'Threshold = {pr_threshold_custom:.2f}', s=50, marker='x')
    ax2.axhline(y=chance_level_precision, color='black', linestyle='--', label='chance level', linewidth=0.5)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, 1.1])
    ax2.set_xlabel('Recall', fontsize=8)
    ax2.set_ylabel('Precision', fontsize=8)
    ax2.set_title('Precision-Recall curve', fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)  


    # ax2.legend(loc="lower left", fontsize=8)

    display_confusion_matrix(y_test, predictions_class, labels=class_labels_display)
    ax3.set_title('Confusion matrix', fontsize=8)
    plt.grid(False)
    plt.subplots_adjust(wspace=0.5)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 8

    plt.savefig(filename, dpi=300)

    print(f'Threshold closest to {threshold} (ROC): {threshold_custom:.2f}')
    print(f'Threshold closest to {threshold} (PR): {pr_threshold_custom:.2f}')

    return results_df, missclassified_samples

# %%
cat_features

# %%
# Convert outcome variable to boolean
outcome_mapping = {class_1: True, class_0: False}
y_train = y_train.replace(outcome_mapping).astype(bool)
y_test = y_test.replace(outcome_mapping).astype(bool)
if external_val:
    y_extval_data = y_extval_data.replace(outcome_mapping).astype(bool)


# %% [markdown]
# ## Initiate machine learning models
# 
# This part is focused on binary classification.

# %% [markdown]
# ##### Variable type encoding for QLattice model (only required for QLattice)

# %%
# empty dictionary to store the stypes
stypes = {}

# iterate over each column in the dataset
for col in mydata_imputed.columns:
    # check if the column dtype is 'category'
    if pd.api.types.is_categorical_dtype(mydata_imputed[col]):
        # if it is, add the column name to the stypes dictionary with a value of 'c'
        stypes[col] = 'c'

stypes[outcome_var] = 'b'
# print the stypes dictionary
print(stypes)


# %% [markdown]
# #### Set model weights based on class balance from the training (development) set

# %%
sample_weights = compute_sample_weight(class_weight='balanced', y=mydata_imputed[outcome_var])

# %% [markdown]
# #### Define the parameter grid for random search

# %%
# this is done when hyperparameter tuning is done
def adjust_hyperparameters(n_rows, n_cols):
    """
    Returns a dictionary of hyperparameter distributions for various machine learning models.

    ## Parameters
        n_rows (int): The number of rows in the dataset.
        n_cols (int): The number of columns in the dataset.

    ## Returns
        dict: A dictionary containing the following keys:
            - 'adjusted_rf_param_dist': Hyperparameters for Random Forest Classifier.
            - 'adjusted_lgbm_param_dist': Hyperparameters for LightGBM Classifier.
            - 'adjusted_hgbc_param_dist': Hyperparameters for Histogram-Based Gradient Boosting Classifier.
            - 'adjusted_cb_param_dist': Hyperparameters for CatBoost Classifier.
            - 'adjusted_lr_param_dist': Hyperparameters for Logistic Regression.

    """
    np.random.seed(SEED)
    # Adjust hyperparameters based on dataset size and class proportion
    # Random Forest Classifier parameters:
    adjusted_rf_param_dist = {
        # Number of trees in the forest
        "n_estimators": np.linspace(100, max(1000, int(np.sqrt(n_rows))), num=10, dtype=int),
        # Maximum depth of the tree
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, n_cols],
        # Minimum number of samples required to split a node
        'min_samples_split': [2, 5, 10, int(np.sqrt(n_rows))],
        # Minimum number of samples required at each leaf node
        'min_samples_leaf': [1, 2, 4, int(np.sqrt(n_rows))],
        # The number of features to consider when looking for the best split
        'max_features': ['sqrt', 'log2', None]
    }
    
    # LightGBM Classifier parameters:
    adjusted_lgbm_param_dist = {
        # Maximum number of leaves in one tree
        "num_leaves": np.linspace(6, 100, num=10, dtype=int),
        # Minimum number of data needed in a child (leaf) node
        'min_child_samples': randint(4, min(int(np.sqrt(n_rows)), 100), size = 10),
        # Minimum sum of instance weight (hessian) needed in a child (leaf) node
        'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
        # Subsample ratio of the training instance
        'subsample': np.linspace(0.5, 0.8, 5),
        # Subsample ratio of columns when constructing each tree
        'colsample_bytree': np.linspace(0.4, 0.6, 5),
        # L1 regularization term on weights
        'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
        # L2 regularization term on weights
        'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
        # Number of boosting iterations
        'n_estimators': np.linspace(100, max(1000, int(np.sqrt(n_rows))), num=10, dtype=int),
        # Maximum depth of tree
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]
    }

    # Histogram-Based Gradient Boosting Classifier parameters:
    adjusted_hgbc_param_dist = {
        # maximum iterations (number of trees)
        "max_iter": np.linspace(100, max(1000, int(np.sqrt(n_rows))), num=10, dtype=int),
        # validation data proportion
        "validation_fraction": np.linspace(0.1, 0.3, 10),
        # Boosting learning rate
        'learning_rate': np.linspace(0.01, 0.2, 10),
        # Maximum depth of the individual estimators
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, n_cols],
        # Minimum number of samples per leaf
        'min_samples_leaf': randint(1, min(5,n_rows), size = 10),
        # Grow trees with max_leaf_nodes in best-first fashion
        'max_leaf_nodes': randint(10, 100, size = 10),
        # L2 regularization term on weights
        'l2_regularization': np.linspace(0.01, 0.2, 10)
    }
    
    # CatBoost Classifier parameters:
    adjusted_cb_param_dist = {
        # Learning rate (like step size)
        'learning_rate': np.logspace(-3, 0, num=10),
        # Depth of the trees (the deeper the more detailed but more vulnerable to overfitting)
        'depth': [3, 4, 5, 6, 7, 8, 9, 10, n_cols],
        # L2 regularization coefficient (for generalizability)
        'l2_leaf_reg': np.logspace(-1, 3, num=100),
        # The number of trees to fit
        'iterations': np.linspace(100, max(1000, int(np.sqrt(n_rows))), num=10, dtype=int),
        # Subsample ratio of the training instance
        'subsample': np.linspace(0.1, 1, 10),
        # Random strength
        'random_strength': np.linspace(0, 10, 10)
    }

    # Logistic Regression parameters:
    adjusted_lr_param_dist = {
        # Inverse of regularization strength
        'C': [0.01, 0.1, 1, 10, 100],
        # Maximum number of iterations for optimization
        'max_iter': [500, 1000, 2000, 5000, int(np.sqrt(n_rows))], 
        # Tolerance for stopping criteria
        'tol': [1e-3, 1e-4, 1e-5]
    }

    return adjusted_rf_param_dist, adjusted_lgbm_param_dist, adjusted_hgbc_param_dist, adjusted_cb_param_dist, adjusted_lr_param_dist


# %% [markdown]
# #### Set parameters for models (when the datset is small)
# 
# This is used when there is no hyperparameter tuning. The parameters are set according to the data characteristics.

# %%
# this is done when hyperparameter tuning is not done
def set_parameters(n_rows, n_cols, class_proportion):
    """
    Sets the parameters for different machine learning classifiers based on the given dataset characteristics.

    ## Parameters:
        n_rows (int): The number of rows in the dataset.
        n_cols (int): The number of columns in the dataset.
        class_proportion (float): The proportion of classes in the dataset.

    ## Returns
        A dictionary containing the parameters for each classifier, including:
            - Balanced Random Forest Classifier
            - LightGBM Classifier
            - Histogram-Based Gradient Boosting Classifier
            - CatBoost Classifier
            - Logistic Regression

    Note that this function assumes hyperparameter tuning is not done and sets default values based on the dataset characteristics.
    """
    # Balanced Random Forest Classifier parameters:
    rf_params = {
        # Number of trees in the forest
        'n_estimators': max(100, int(n_rows / 100)),
        # Maximum depth of the tree
        'max_depth': 10,
        # Minimum number of samples required to split a node
        'min_samples_split': 2 if class_proportion > 0.1 else 10,
        # Minimum number of samples required at each leaf node
        'min_samples_leaf': 1 if class_proportion > 0.1 else 4,
        # The number of features to consider when looking for the best split
        'max_features': 'sqrt'
    }

    # LightGBM Classifier parameters:
    lgbm_params = {
        # Maximum number of leaves in one tree
        'num_leaves': min(6, min(50, 2*n_rows)),
        # Minimum number of data needed in a child (leaf) node
        'min_child_samples': min(100, int(n_rows / 20)),
        # Minimum sum of instance weight (hessian) needed in a child (leaf) node
        'min_child_weight': 1e-3,
        # Subsample ratio of the training instance
        'subsample': min(0.8, max(0.2, 0.5 + (class_proportion - 0.5) / 2)),
        # Subsample ratio of columns when constructing each tree
        'colsample_bytree': 0.8,
        # L1 regularization term on weights
        'reg_alpha': 0.01,
        # L2 regularization term on weights
        'reg_lambda': 0.01,
        # Number of boosting iterations
        'n_estimators': min(1000, 2*n_rows),
        # Maximum depth of tree
        'max_depth': 10
    }

    # Histogram-Based Gradient Boosting Classifier parameters:
    hgbc_params = {
        # maximum iterations (number of trees)
        "max_iter":min(1000, 2*n_rows),
        # Boosting learning rate
        'learning_rate': 0.1,
        # Maximum depth of the individual estimators
        'max_depth': 10,
        # Minimum number of samples per leaf
        'min_samples_leaf': 2,
        # Grow trees with max_leaf_nodes in best-first fashion
        'max_leaf_nodes': 3,
        # L2 regularization term on weights
        'l2_regularization': 0.1
    }

    # CatBoost Classifier parameters:
    cb_params = {
        # Learning rate
        'learning_rate': 0.1,
        # Depth of the trees
        'depth': min(10, int(n_cols/2)),
        # L2 regularization coefficient
        'l2_leaf_reg': 1.0,
        # The number of trees to fit
        'iterations': min(1000, 2*n_rows),
        # Subsample ratio of the training instance
        'subsample': 0.8,
        # Random strength
        'random_strength': 5
    }

    # Logistic Regression parameters:
    lr_params = {
        # Inverse of regularization strength
        'C': 1.0,
        # Maximum number of iterations for optimization
        'max_iter': 1000,
        # Tolerance for stopping criteria
        'tol': 1e-4
    }

    return rf_params, lgbm_params, hgbc_params, cb_params, lr_params


# %%
def PFI_median_wrap(PFI_folds):
    """
    Computes the median importance across all folds and normalizes the importance values.

    ## Parameters:
    PFI_folds (list of feature importance values in folds): List of DataFrames where each DataFrame contains 'Feature' and 'Importance' columns.

    ## Returns
    pd.DataFrame: DataFrame with 'Feature' and normalized 'Importance' sorted by importance.
    """

    # Get the number of folds
    num_folds = len(PFI_folds)
    
    # Start with the 'Feature' column from the first DataFrame
    merged_df = PFI_folds[0][['Feature']].copy()
    
    # Loop through each fold and add the 'Importance' column to the DataFrame
    for i in range(num_folds):
        fold_column = PFI_folds[i][['Importance']].rename(columns={'Importance': f'Importance Fold {i+1}'})
        merged_df = merged_df.merge(fold_column, left_index=True, right_index=True)
    
    # Calculate the median of importance values for each feature
    importance_columns = [f'Importance Fold {i+1}' for i in range(num_folds)]
    merged_df['Importance'] = merged_df[importance_columns].median(axis=1)
    
    PFI_median = merged_df.copy()
    # # Select only the 'Feature' and 'Importance' columns
    # PFI_median = merged_df[['Feature', 'Importance']]
    
    # Sort the DataFrame by 'Importance' in descending order
    PFI_median = PFI_median.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Normalize the 'Importance' column
    PFI_median['Importance'] = minmax_scaler.fit_transform(PFI_median[['Importance']])
    
    return PFI_median



# %%
def plot_PFI(PFI_folds, X, model_name):
    """
    Plot permutation-based feature importances (PFI) from multiple folds using a strip plot.

    ## Parameters:
    - PFI_folds (list of DataFrames): List where each DataFrame contains 'Feature' and 'Importance' columns for each fold.
    - X (DataFrame): DataFrame used to determine the number of features for plot sizing.
    - model_name (str): A string representing the name of the model or experiment, used for naming the output files.

    ## Returns:
    - Saves the plot with filenames including the model_name parameter and displays it.
    """
    # Combine feature importances from all folds into a single DataFrame
    combined_importances = PFI_median_wrap(PFI_folds)
    median_importance = combined_importances[['Feature', 'Importance']]
    
    # Plot boxplot for feature importances
    plt.figure(figsize=(5, 0.5 * X.shape[1]))
    sns.stripplot(x="Importance", y="Feature", data=pd.concat(PFI_folds, axis=0), order=median_importance['Feature'], jitter=True, alpha=0.5)
    plt.title(f"Fold-wise mean permutation importances for {model_name}", size=10)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 8
    plt.grid(linestyle=':', linewidth=0.5, alpha=0.5)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=8) 
    plt.gca().set_facecolor('white')  
    # Display grid lines
    plt.grid(which='both', color="grey")
    # Modify grid lines
    plt.grid(which='minor', alpha=0.1)
    plt.grid(which='major', alpha=0.3)
    # Add a dotted line at x = 0
    plt.axvline(x=0, color="k", linestyle="--", linewidth=1)
    
    # Save plot with model_name in the filename
    plt.savefig(f"FI_perm_{model_name}.tif", bbox_inches='tight')
    plt.show()


# %%
def plot_TFI(X, tree_FI, model_name):
    """
    Plots the tree-based feature importances for a given model.

    ## Parameters:
    X (pd.DataFrame): The training data used for feature names.
    tree_FI (list of pd.Series): List of feature importance scores for tree-based feature importance from each fold.
    model_name (str): Name of the model to use in the plot title and filenames.

    ## Returns
    None: Displays and saves the plot.
    """
    # Extract feature names from X
    feature_names = X.columns

    # Combine feature importances from all folds into a single DataFrame
    combined_importances = pd.concat(
        [pd.DataFrame({'Feature': feature_names, 'Importance': fold}) for fold in tree_FI],
        axis=0, ignore_index=True
    )

    # Calculate the median importance across folds for each feature
    median_importance = combined_importances.groupby("Feature")["Importance"].median().reset_index()

    # Sort features by their median importance
    median_importance = median_importance.sort_values(by="Importance", ascending=False)

    # Plot feature importances
    plt.figure(figsize=(5, 0.5 * X_train.shape[1]))
    sns.stripplot(x="Importance", y="Feature", data=combined_importances, 
                  order=median_importance['Feature'], jitter=True, alpha=0.5)
    plt.title(f"Mean tree-based feature importances per fold ({model_name})", size=10)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 8
    plt.grid(linestyle=':', linewidth=0.5, alpha=0.5)
    plt.axvline(x=0, color="k", linestyle="--", linewidth=1)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=8)
    plt.gca().set_facecolor('white')
    plt.grid(which='both', color="grey")
    plt.grid(which='minor', alpha=0.1)
    plt.grid(which='major', alpha=0.3)
    
    # Save the plot to files
    plt.savefig(f"FI_tree_{model_name}.tif", bbox_inches='tight')
    
    # Show the plot
    plt.show()


# %%
def shap_summary_plot(shap_values, data, model_name):
    """
    Generates and saves a SHAP summary plot based on provided SHAP values and data from cross validation
    
    Parameters:
    - shap_values: concatenated list of SHAP values arrays from different folds
    - data: DataFrames (trainset or testset)
    - model_name: Name of the model (e.g., "CB" for CatBoost)
    
    ## Returns
    - None: Saves the plot as a .tif file and displays it.
    """
   
    # Create the SHAP summary plot
    shap.summary_plot(
        shap_values, 
        data, 
        color=plt.get_cmap("viridis"), 
        show=False, 
        alpha = 0.8, max_display=top_n_f
    )
    
    # Customize the plot appearance
    plt.gcf().axes[-1].tick_params(labelsize=8)
    plt.grid(linestyle=':', linewidth=0.5, alpha=0.5)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 8
    plt.yticks(size=8)
    plt.xticks(size=8)
    plt.xlabel('SHAP value', size=8)
    plt.ylabel('feature', size=8)
    fig, ax = plt.gcf(), plt.gca()
    fig.axes[-1].set_ylabel('feature value', size=8)
    plt.xticks(size=8)
    plt.grid(True)
    plt.gca().tick_params(axis='both', labelsize=8)
    plt.gca().set_facecolor('white')
    
    # Display grid lines
    plt.grid(which='both', color="grey")
    plt.grid(which='minor', alpha=0.1)
    plt.grid(which='major', alpha=0.2)
    
    # Save and display the plot
    plot_filename = f"SHAP_summary_plot_{model_name}.tif"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()
    
    # Check if there are categorical features and generate a plot with categories if needed
    if 'category' in data.dtypes.values:
        feature_names_with_shapvalues = [
            f"{feature}: {round(value, 2)}"
            for feature, value in zip(data.columns, np.mean(np.abs(shap_values), axis=0))
        ]
        # SHAP summary plot with categories
        categorical_shap_plot(
            shap_values=shap_values, 
            data=data,
            top_n=min(len(feature_names_with_shapvalues),top_n_f),
            jitter=0.1
        )
        
        # Customize the second plot appearance
        plt.gca().set_facecolor('white')
        plt.grid(which='both', color="grey")
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.3)
        
        # Save the plot with categories
        plot_filename_with_cats = f"SHAP_summary_plot_{model_name}_withcats.tif"
        plt.savefig(plot_filename_with_cats, dpi = 300)
        plt.show()



# %% [markdown]
# #### K-fold stratified cross validation of binary classification models
# 
# This block contains the code to perform cross-validation for all selected binary classification models. The code calculates performance measures for all models, applies hyperparameter tuning and training, and generates visualizations for feature importance using various approaches (e.g., SHAP, feature permutation, tree-based feature importance). Additionally, it produces performance metrics such as ROC and PR curves.

# %%

# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculates various evaluation metrics for binary classification models.

    ## Parameters
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_pred_proba (array-like): Predicted probabilities of positive outcomes.

    ## Returns
        dict: Dictionary containing the following evaluation metrics:
            - PPV (Positive Predictive Value)
            - NPV (Negative Predictive Value)
            - Sensitivity (True Positive Rate)
            - Specificity (True Negative Rate)
            - Balanced Accuracy
            - Matthews Correlation Coefficient (MCC)
            - Receiver Operating Characteristic AUC Score (ROCAUC)
            - Precision-Recall AUC Score (PRAUC)
            - Brier Score
            - F1 Score

    Notes:
        The evaluation metrics are calculated based on the true labels, predicted labels, and predicted probabilities.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels= [False, True])
    tn, fp, fn, tp = cm.ravel()
    # Positive Predictive Value (Precision)
    PPV = tp / (tp + fp)
    # Negative Predictive Value
    NPV = tn / (tn + fn)
    # True Positive Rate (Recall)
    Sensitivity = tp / (tp + fn)
    # True Negative Rate
    Specificity = tn / (tn + fp)
    Balanced_Accuracy = (Sensitivity + Specificity) / 2 # Balanced Accuracy
    MCC = matthews_corrcoef(y_true, y_pred) # Matthews Correlation Coefficient
    ROC_AUC = roc_auc_score(y_true, y_pred_proba) # ROC AUC Score
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba, pos_label=True)
    PR_AUC = auc(recall, precision) # Precision-Recall AUC Score
    Brier_Score = brier_score_loss(y_true, y_pred_proba, pos_label=True) # Brier Score
    F1_Score = f1_score(y_true, y_pred) # F1 Score
    return {
        'PPV': PPV,
        'NPV': NPV,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'Balanced Accuracy': Balanced_Accuracy,
        'MCC': MCC,
        'ROCAUC': ROC_AUC,
        'PRAUC': PR_AUC,
        'Brier Score': Brier_Score,
        'F1 Score': F1_Score
    }
# Function to cross-validate the model
def cross_validate_model(model_class, X, y, sample_weights=None, n_splits=cv_folds, random_state=SEED, measures=None,
                         use_default_threshold=False, **model_params):
    """
    Perform k-fold cross-validation and evaluate the model.

    ## Parameters:
        X (array-like): Feature data.
        y (array-like): Target labels.
        model: Trained model instance.
        use_default_threshold (bool, optional): Use default threshold (0.5) for classification. Defaults to True.

    ## Returns
        tuple: Fold results, aggregated results table, optimal threshold, feature importance lists, and SHAP values list.
    """
    n_repeats = n_rep_feature_permutation # number of repetitions for feature permutation
    if measures is None:
        measures = ['PPV', 'NPV', 'Sensitivity', 'Specificity', 'Balanced Accuracy', 'MCC', 'ROCAUC', 'PRAUC', 'Brier Score', 'F1 Score']
    fold_data = [] # to save the data that are split by folds for subsequent analyses
    y_fold_data = []
    fold_results = pd.DataFrame()
    fold_results_plt = pd.DataFrame()
    aggregated_thr = np.array([]) # aggregated list of estimated optimal thresholds per fold
    aggregated_predictions = np.array([])
    aggregated_labels = np.array([])
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    feature_importance_list = []
    treebased_feature_importance_list = []
    shap_values_list = []  # To store SHAP values for each fold
    missclassified_samples = [] # to store the index of missclassified samples
    ########
    overlapping_samples = False
    test_indices_list = []  # To store test indices of samples in each fold
    ########
    predictions_proba_fold_list = []
    predictions_proba_fold_train_list = []
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Check for overlapping test samples across previous folds
        if fold > 1:
            current_test_index = set(test_index)
            for prev_test_index in test_indices_list:
                if not current_test_index.isdisjoint(prev_test_index):
                    overlapping_samples = True
                    break
        # Store test indices of samples in this fold
        test_indices_list.append(set(test_index))
        #####################
        sample_weights_fold = sample_weights[train_index] if sample_weights is not None else None
        # Calculate necessary information for adjusting hyperparameters
        n_rows = X_train_fold.shape[0]
        n_cols = X_train_fold.shape[1]
        class_proportion = y_train_fold.mean()  # binary classification
        # Adjust hyperparameters based on the training data in this fold
        rf_param_dist, lgbm_param_dist, hgbc_param_dist, cb_param_dist, lr_param_dist = adjust_hyperparameters(n_rows, n_cols)
        rf_params, lgbm_params, hgbc_params, cb_params, lr_params = set_parameters(n_rows, n_cols, class_proportion)
        # Check if the model is a RandomForestClassifier
        if model_class == RandomForestClassifier:
            # Explicitly set sampling_strategy to 'all'
            rf_model = RandomForestClassifier(random_state=SEED, n_jobs=n_cpu_model_training, **rf_params) # , class_weight = 'balanced'
            # if hyperparameter tuning should be done or not
            if hp_tuning:
                random_search = RandomizedSearchCV(
                estimator=rf_model, 
                param_distributions=rf_param_dist, 
                n_iter=n_iter_hptuning,
                scoring= custom_scorer,  
                cv=cv_folds_hptuning,
                refit=True, 
                random_state=random_state,
                verbose=0, 
                n_jobs = n_cpu_for_tuning)
                # Fit the RandomizedSearchCV object to the data
                random_search.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
                # Get the best parameters and best estimator from the random search
                best_params = random_search.best_params_
                rf_model = RandomForestClassifier(random_state=SEED,n_jobs=n_cpu_model_training, **best_params) # , class_weight = 'balanced' (note: class weight should not be used simultaneously with sample weight, it messes up the sample weighting)
            else:
                rf_model = RandomForestClassifier(random_state=SEED, n_jobs=n_cpu_model_training, **rf_params) # , class_weight = 'balanced'
            # Fit the best estimator on the entire training data
            rf_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
            # Get predictions on the test data
            predictions_proba = rf_model.predict_proba(X_test_fold)[:, 1]
            treebased_feature_importance = rf_model.feature_importances_
            treebased_feature_importance_list.append(treebased_feature_importance)
            # Use permutation_importance to get feature importances
            perm_result = permutation_importance(
                rf_model, X_test_fold, y_test_fold, n_repeats=n_repeats, random_state=random_state, n_jobs=n_cpu_model_training, scoring = custom_scorer # "roc_auc"
            )
            # Get feature importances and sort them
            feature_importance = perm_result.importances_mean # Mean of feature importance over n_repeats
            feature_importance_df = pd.DataFrame(
                {"Feature": X_train_fold.columns, "Importance": feature_importance}
            )
            feature_importance_df = feature_importance_df.sort_values(
                by="Importance", ascending=False
            )
            # Append to the list
            feature_importance_list.append(feature_importance_df)
            # Compute SHAP values
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_test_fold)
            shap_values_list.append(shap_values)
            fold_data.append(X_test_fold) # for subsequent SHAP vs feature value analyses
            y_fold_data.append(y_test_fold)
            
            ############
            predictions_proba_fold = rf_model.predict_proba(X_test_fold)[:, 1]
            predictions_proba_fold_train = rf_model.predict_proba(X_train_fold)[:, 1]
            predictions_proba_fold_list.append(predictions_proba_fold)
            predictions_proba_fold_train_list.append(predictions_proba_fold_train)
            # Check if the model is a QLattice
        elif model_class == 'QLattice':
            X_train_fold_ql = X_train_fold.copy()
            X_train_fold_ql[outcome_var] = y_train_fold.values
            if hp_tuning:
                best_composite_score = 0
                best_parameters = {'n_epochs': 50, 'max_complexity': 10}
                def evaluate_params_kfold(n_epochs, max_complexity):
                    """
                    Evaluate a QLattice model for given hyperparameters.

                    ## Parameters:
                        n_epochs (int): The number of epochs used for training.
                        max_complexity (int): The maximum complexity of the model.

                    ## Returns
                        QL_composite_score (float): The composite score of the model.
                        params (dict): The hyperparameters used to achieve this score.
                    """
                    ql = feyn.QLattice(random_seed=random_state)
                    models = ql.auto_run(
                        data=X_train_fold_ql,
                        output_name=outcome_var,
                        kind='classification',
                        n_epochs=n_epochs,
                        stypes=stypes,
                        criterion="aic",
                        loss_function='binary_cross_entropy',
                        max_complexity=max_complexity,
                        sample_weights=sample_weights_fold
                    )
                    best_model = models[0]
                    predictions_proba = best_model.predict(X_test_fold)
                    QL_composite_score = (roc_auc_score(y_true = y_test_fold, y_score = predictions_proba) +
                                          average_precision_score(y_true = y_test_fold, y_score = predictions_proba))/2
                    return QL_composite_score, {'n_epochs': n_epochs, 'max_complexity': max_complexity}
                results = Parallel(n_jobs=n_cpu_for_tuning, backend='loky')(
                    delayed(evaluate_params_kfold)(n_epochs, max_complexity)
                    for n_epochs in [50, 100]
                    for max_complexity in [5, 10]
                )
                for QL_composite_score, params in results:
                    if QL_composite_score > best_composite_score:
                        best_composite_score = QL_composite_score
                        best_parameters = params
                print("Best Parameters:", best_parameters)
                print("Best composite score:", best_composite_score)
                # Use the best parameters from the grid search
                best_n_epochs = best_parameters['n_epochs']
                best_max_complexity = best_parameters['max_complexity']
            else:
                best_n_epochs = 50
                best_max_complexity = 10
            # Train the final model with the best parameters
            ql = feyn.QLattice(random_seed=random_state)
            models = ql.auto_run(
                data=X_train_fold_ql,
                output_name=outcome_var,
                kind='classification',
                n_epochs=best_n_epochs,
                stypes=stypes,
                criterion="aic",
                loss_function='binary_cross_entropy',
                max_complexity=best_max_complexity,
                sample_weights=sample_weights_fold
            )
            model = models[0]
            predictions_proba = model.predict(X_test_fold)
            # Calculate the baseline custom score = (AUC+PRAUC)/2
            baseline_score = combined_metric(y_true = y_test_fold, y_pred_proba=predictions_proba)
            # baseline_roc_auc = roc_auc_score(y_test_fold, predictions_proba)
            # Initialize an array to store the permutation importances
            perm_importances = []
            # Iterate over each feature and permute its values
            for feature in X_test_fold.columns:
                # Permute the feature values
                permuted_features = X_test_fold.copy()
                permuted_features[feature] = np.random.permutation(permuted_features[feature])
                # Make predictions on the entire dataset with permuted feature
                permuted_predictions = model.predict(permuted_features)
                # Calculate the custom score = (AUC+PRAUC)/2 with permuted feature
                permuted_score = combined_metric(y_true=y_test_fold, y_pred_proba=permuted_predictions)
                # Calculate permutation importance for the feature
                perm_importance = baseline_score - permuted_score
                perm_importances.append((feature, perm_importance))
            # Sort the permutation importances
            perm_importances.sort(key=lambda x: x[1], reverse=True)
            # Get feature importances and sort them
            feature_importance = [importance for feature, importance in perm_importances]
            feature_importance_df = pd.DataFrame(
                {"Feature": X_test_fold.columns, "Importance": feature_importance}
            )
            feature_importance_df = feature_importance_df.sort_values(
                by="Importance", ascending=False
            )
            # Append to the list
            feature_importance_list.append(feature_importance_df)
            treebased_feature_importance = []
            treebased_feature_importance_list.append(treebased_feature_importance)
            predictions_proba_fold = model.predict(X_test_fold)
            predictions_proba_fold_train = model.predict(X_train_fold)
            predictions_proba_fold_list.append(predictions_proba_fold)
            predictions_proba_fold_train_list.append(predictions_proba_fold_train)
            # Clear memory from QLattice models
            ql = None
            models = None
        elif model_class == HistGradientBoostingClassifier:
            # Create a HistGradientBoostingClassifier instance
            hgbc_model = HistGradientBoostingClassifier(random_state=random_state, early_stopping=True, **hgbc_params)
            if hp_tuning:
                # Create a RandomizedSearchCV instance
                random_search = RandomizedSearchCV(
                    estimator=hgbc_model, 
                    param_distributions=hgbc_param_dist, 
                    n_iter=n_iter_hptuning,
                    scoring= custom_scorer, 
                    cv=cv_folds_hptuning,
                    refit=True, 
                    random_state=random_state,
                    verbose=0,
                n_jobs = n_cpu_for_tuning)
                # Perform the random search on the training data
                random_search.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
                # Get the best parameters and best estimator
                best_params = random_search.best_params_
                hgbc_model = HistGradientBoostingClassifier(random_state=random_state, early_stopping=True, **best_params)
            else:
                hgbc_model = HistGradientBoostingClassifier(random_state=random_state, early_stopping=True, **hgbc_params)
            # model = random_search.best_estimator_
            # Fit the best estimator on the entire training data
            hgbc_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
            # Get predictions on the test data
            predictions_proba = hgbc_model.predict_proba(X_test_fold)[:, 1]
            treebased_feature_importance = [] # model.feature_importances_ is not implemented for HistGradientBoostingClassifier
            treebased_feature_importance_list.append(treebased_feature_importance)
            # feature_importance = model.feature_importances_
            # feature_importance_list.append(feature_importance)
            perm_result = permutation_importance(
                hgbc_model, X_test_fold, y_test_fold, n_repeats=n_repeats, random_state=random_state, n_jobs=n_cpu_model_training, scoring = custom_scorer # "roc_auc"
            )
            # Get feature importances and sort them
            feature_importance = perm_result.importances_mean # Mean of feature importance over n_repeats
            feature_importance_df = pd.DataFrame(
                {"Feature": X_train_fold.columns, "Importance": feature_importance}
            )
            feature_importance_df = feature_importance_df.sort_values(
                by="Importance", ascending=False
            )
            # Append to the list
            feature_importance_list.append(feature_importance_df)
            # Compute SHAP values
            explainer = shap.TreeExplainer(hgbc_model)
            shap_values = explainer.shap_values(X_test_fold)
            shap_values_list.append(shap_values)
            fold_data.append(X_test_fold) # for subsequent SHAP vs feature value analyses
            y_fold_data.append(y_test_fold)
            
            predictions_proba_fold = hgbc_model.predict_proba(X_test_fold)[:, 1]
            predictions_proba_fold_train = hgbc_model.predict_proba(X_train_fold)[:, 1]
            predictions_proba_fold_list.append(predictions_proba_fold)
            predictions_proba_fold_train_list.append(predictions_proba_fold_train)
        elif model_class == lgb.LGBMClassifier:
            # LightGBM instance
            if GPU_avail:
                lgbm_model = lgb.LGBMClassifier(random_state=random_state, n_jobs=n_cpu_model_training, verbose=-1,device="gpu", **lgbm_params) 
            else:
                lgbm_model = lgb.LGBMClassifier(random_state=random_state, n_jobs=n_cpu_model_training, verbose=-1, **lgbm_params) 
            if hp_tuning:
                random_search = RandomizedSearchCV(
                    estimator=lgbm_model, 
                    param_distributions=lgbm_param_dist, 
                    n_iter=n_iter_hptuning,
                    scoring= custom_scorer, 
                    cv=cv_folds_hptuning,
                    refit=True, 
                    random_state=random_state,
                    verbose=0, 
                    n_jobs = n_cpu_for_tuning)
                # Perform the random search on the training data
                random_search.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
                # Get the best parameters and best estimator
                best_params = random_search.best_params_
                if GPU_avail:
                    lgbm_model = lgb.LGBMClassifier(random_state=random_state, n_jobs=n_cpu_model_training, verbose=-1, device="gpu", **best_params) 
                else:
                    lgbm_model = lgb.LGBMClassifier(random_state=random_state, n_jobs=n_cpu_model_training, verbose=-1, **best_params) 
            # Fit the best estimator on the entire training data
            lgbm_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
            # Get predictions on the test data
            predictions_proba = lgbm_model.predict_proba(X_test_fold)[:, 1]
            treebased_feature_importance = lgbm_model.feature_importances_
            treebased_feature_importance_list.append(treebased_feature_importance)
            # feature_importance = model.feature_importances_
            # feature_importance_list.append(feature_importance)
            perm_result = permutation_importance(
                lgbm_model, X_test_fold, y_test_fold, n_repeats=n_repeats, random_state=random_state, n_jobs=n_cpu_model_training, scoring = custom_scorer # "roc_auc"
            )
            # Get feature importances and sort them
            feature_importance = perm_result.importances_mean # Mean of feature importance over n_repeats
            feature_importance_df = pd.DataFrame(
                {"Feature": X_train_fold.columns, "Importance": feature_importance}
            )
            feature_importance_df = feature_importance_df.sort_values(
                by="Importance", ascending=False
            )
            # Append to the list
            feature_importance_list.append(feature_importance_df)
            # Compute SHAP values
            explainer = shap.TreeExplainer(lgbm_model)
            shap_values = explainer.shap_values(X_test_fold)
            shap_values_list.append(shap_values)
            fold_data.append(X_test_fold) # for subsequent SHAP vs feature value analyses
            y_fold_data.append(y_test_fold)
            
            predictions_proba_fold = lgbm_model.predict_proba(X_test_fold)[:, 1]
            predictions_proba_fold_train = lgbm_model.predict_proba(X_train_fold)[:, 1]
            predictions_proba_fold_list.append(predictions_proba_fold)
            predictions_proba_fold_train_list.append(predictions_proba_fold_train)
        elif model_class == cb.CatBoostClassifier: # cb.CatBoostClassifier
            # Define the CatBoost classifier
            # if GPU_avail:
            #     cb_model = cb.CatBoostClassifier(random_state=random_state, cat_features=cat_features, silent=True, task_type="GPU", bootstrap_type = "No") # , logging_level='Silent' verbose=0, 
            # else:
            cb_model = cb.CatBoostClassifier(random_state=random_state, cat_features=cat_features, silent=True,**cb_params) # , **cb_params, logging_level='Silent' verbose=0, silent=True,
            if hp_tuning:
                # Create a RandomizedSearchCV instance
                random_search = RandomizedSearchCV(
                    estimator=cb_model, 
                    param_distributions=cb_param_dist, 
                    n_iter=n_iter_hptuning,
                    scoring= custom_scorer, 
                    cv=cv_folds_hptuning,
                    refit=True, 
                    random_state=random_state,
                    verbose=0, 
                    n_jobs = n_cpu_for_tuning
                )
                # Perform the random search on the training data
                random_search.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
                # Get the best parameters and best estimator
                best_params = random_search.best_params_
                cb_model = cb.CatBoostClassifier(random_state=random_state, cat_features=cat_features, silent=True,**best_params) # logging_level='Silent', verbose=0, silent=True,
            # Fit the best estimator on the entire training data
            cb_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
            # Get predictions on the test data
            predictions_proba = cb_model.predict_proba(X_test_fold)[:, 1]
            treebased_feature_importance = cb_model.feature_importances_
            treebased_feature_importance_list.append(treebased_feature_importance)

            perm_result = permutation_importance(
                cb_model, X_test_fold, y_test_fold, n_repeats=n_repeats, random_state=random_state, n_jobs=n_cpu_model_training, scoring = custom_scorer # "roc_auc"
            )
            # Get feature importances and sort them
            feature_importance = perm_result.importances_mean # Mean of feature importance over n_repeats
            feature_importance_df = pd.DataFrame(
                {"Feature": X_train_fold.columns, "Importance": feature_importance}
            )
            feature_importance_df = feature_importance_df.sort_values(
                by="Importance", ascending=False
            )
            # Append to the list
            feature_importance_list.append(feature_importance_df)
            # Compute SHAP values
            explainer = shap.TreeExplainer(cb_model)
            shap_values = explainer.shap_values(X_test_fold)
            shap_values_list.append(shap_values)
            fold_data.append(X_test_fold) # for subsequent SHAP vs feature value analyses
            y_fold_data.append(y_test_fold)
            
            predictions_proba_fold = cb_model.predict_proba(X_test_fold)[:, 1]
            predictions_proba_fold_train = cb_model.predict_proba(X_train_fold)[:, 1]
            predictions_proba_fold_list.append(predictions_proba_fold)
            predictions_proba_fold_train_list.append(predictions_proba_fold_train)

        # Check if the specified model class is LogisticRegression (logistic regression regularized on both L1 and L2 terms - elasticnet)
        elif model_class == LogisticRegression:
            # Define the Logistic Regression classifier (configured as elasticnet)
            lr_model = LogisticRegression(penalty='l1', random_state=random_state, solver="liblinear", **lr_params)
            if hp_tuning:
                # Create a RandomizedSearchCV instance
                random_search = RandomizedSearchCV(
                    estimator=lr_model, 
                    param_distributions=lr_param_dist, 
                    n_iter=n_iter_hptuning,
                    scoring= custom_scorer, 
                    cv=cv_folds_hptuning,
                    refit=True, 
                    random_state=random_state,
                    verbose= 0,
                    n_jobs=n_cpu_for_tuning
                )
                # Perform the random search on the training data
                random_search.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
                # Get the best parameters and best estimator
                best_params = random_search.best_params_
                lr_model = LogisticRegression(penalty='l1',random_state=random_state, solver="liblinear", **best_params)
            else:
                lr_model = LogisticRegression(penalty='l1',random_state=random_state, solver="liblinear", **lr_params)
            # Fit the best estimator on the entire training data
            lr_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
            # Get predictions on the test data
            predictions_proba = lr_model.predict_proba(X_test_fold)[:, 1]
            perm_result = permutation_importance(
                lr_model, X_test_fold, y_test_fold, n_repeats=n_repeats, random_state=random_state, n_jobs=n_cpu_model_training, scoring = custom_scorer #"roc_auc"
            )
            # Get feature importances and sort them
            feature_importance = perm_result.importances_mean # Mean of feature importance over n_repeats
            feature_importance_df = pd.DataFrame(
                {"Feature": X_train_fold.columns, "Importance": feature_importance}
            )
            feature_importance_df = feature_importance_df.sort_values(
                by="Importance", ascending=False
            )
            # Append to the list
            feature_importance_list.append(feature_importance_df)
            # Compute SHAP values
            explainer = shap.LinearExplainer(lr_model, X_train_fold)
            shap_values = explainer.shap_values(X_test_fold)
            shap_values_list.append(shap_values)
            fold_data.append(X_test_fold) # for subsequent SHAP vs feature value analyses
            y_fold_data.append(y_test_fold)
            
            predictions_proba_fold = lr_model.predict_proba(X_test_fold)[:, 1]
            predictions_proba_fold_train = lr_model.predict_proba(X_train_fold)[:, 1]
            predictions_proba_fold_list.append(predictions_proba_fold)
            predictions_proba_fold_train_list.append(predictions_proba_fold_train)
        elif model_class == GaussianNB: # Naive Bayes
            nb_model = GaussianNB()
            nb_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
            predictions_proba = nb_model.predict_proba(X_test_fold)[:, 1]
            perm_result = permutation_importance(
                nb_model, X_test_fold, y_test_fold, n_repeats=n_repeats, random_state=random_state, n_jobs=n_cpu_model_training, scoring = custom_scorer # "roc_auc"
            )
            # Get feature importances and sort them
            feature_importance = perm_result.importances_mean # Mean of feature importance over n_repeats
            feature_importance_df = pd.DataFrame(
                {"Feature": X_train_fold.columns, "Importance": feature_importance}
            )
            feature_importance_df = feature_importance_df.sort_values(
                by="Importance", ascending=False
            )
            # Append to the list
            feature_importance_list.append(feature_importance_df)
            treebased_feature_importance = [] # empty as it is not defined for Naive Bayes model 
            treebased_feature_importance_list.append(treebased_feature_importance)
            # Compute SHAP values 
            explainer = shap.Explainer(nb_model.predict_proba, X_train_fold)
            shap_values = explainer(X_test_fold)
            shap_values_list.append(shap_values)
            fold_data.append(X_test_fold) # for subsequent SHAP vs feature value analyses
            y_fold_data.append(y_test_fold)
            
            predictions_proba_fold = nb_model.predict_proba(X_test_fold)[:, 1]
            predictions_proba_fold_train = nb_model.predict_proba(X_train_fold)[:, 1]
            predictions_proba_fold_list.append(predictions_proba_fold)
            predictions_proba_fold_train_list.append(predictions_proba_fold_train)
        # Aggregate predictions and labels
        aggregated_predictions = np.concatenate((aggregated_predictions, predictions_proba))
        aggregated_labels = np.concatenate((aggregated_labels, y_test_fold))
    # Other processing for each fold goes here
    if overlapping_samples:
        print("Warning: Overlapping test samples found across folds.")
    
    # Initialize plot objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.set_title('ROC Curve')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax2.set_title('Precision-Recall Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    # Initialize a list to store thresholds for each fold
    thresholds_per_fold = [0.5]
    # Calculate and store metrics for each fold
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_test_fold = X.iloc[test_index]
        y_test_fold = y.iloc[test_index]
        predictions_proba_fold = predictions_proba_fold_list[fold - 1]
        predictions_proba_fold_train = predictions_proba_fold_train_list[fold - 1]
        y_train_fold = y.iloc[train_index]
        # # Get predictions for the current fold using the optimal threshold
            # Use default threshold if specified
        if use_default_threshold:
            opt_threshold_fold = 0.5
        else: # we use prediction probabilities from the train subsets to estimate the optimal threshold for classification
            class_1_probs = predictions_proba_fold_train[y_train_fold == True]
            class_0_probs = predictions_proba_fold_train[y_train_fold == False]
            median_class_1_probs = np.median(class_1_probs)
            median_class_0_probs = np.median(class_0_probs)
            # Update threshold based on previous folds
            opt_threshold_fold = np.median([threshold for threshold in thresholds_per_fold])
            # Append current fold's threshold to the list
            threshold = np.mean([median_class_1_probs, median_class_0_probs])
            thresholds_per_fold.append(threshold)
        predictions_class_fold = np.where(predictions_proba_fold >= opt_threshold_fold, True, False)
        ###########
        # Find the indices where y_test_fold does not equal predictions_class_fold
        missclassified_samples_fold = y_test_fold.index[np.where(y_test_fold != predictions_class_fold)[0]]
        missclassified_samples.extend(missclassified_samples_fold.tolist())
        ###########
        # Calculate metrics
        metrics = calculate_metrics(y_test_fold, predictions_class_fold, predictions_proba_fold)
        metrics['Fold'] = fold
        # fold_results = fold_results.append(metrics, ignore_index=True)
        fold_results = pd.concat([fold_results, pd.DataFrame(metrics, index=[0])], ignore_index=True)
        # Compute ROC and PR curve values
        fpr, tpr, _ = roc_curve(y_test_fold, predictions_proba_fold, pos_label=True, drop_intermediate=False)
        precision, recall, _ = precision_recall_curve(y_test_fold, predictions_proba_fold, pos_label=True)
        # Create a DataFrame for the current fold's results
        fold_results_df = pd.DataFrame({
            'fold': fold,
            'fpr': list(fpr) + [None] * (len(recall) - len(fpr)),  # Padding to match lengths
            'tpr': list(tpr) + [None] * (len(recall) - len(tpr)),  # Padding to match lengths
            'precision': list(precision),
            'recall': list(recall)
        })
        # Append the current fold's results to the existing results DataFrame
        fold_results_plt = pd.concat([fold_results_plt, fold_results_df], ignore_index=True)
        # Plot ROC and PR curves for the current fold
        ax1.plot(fpr, tpr, label=f'Fold {fold}', alpha=0.5)
        ax2.plot(recall, precision, label=f'Fold {fold}', alpha=0.5)
    # Finalize plots
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower left')
    ax1.set_facecolor('white')
    # show both grid lines
    ax1.grid(which='both', color = "grey")
    # modify grid lines:
    ax1.grid(which='minor', alpha=0.1)
    ax1.grid(which='major', alpha=0.2)
    ax1.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=0.5, label='chance level')
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, 1.1])
    ax1.set_xlabel('False Positive Rate', fontsize=8)
    ax1.set_ylabel('True Positive Rate', fontsize=8)
    ax1.set_title('ROC curve', fontsize=8)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)  
    ax2.set_facecolor('white')
    # show both grid lines
    ax2.grid(which='both', color = "grey")
    # modify grid lines:
    ax2.grid(which='minor', alpha=0.1)
    ax2.grid(which='major', alpha=0.2)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, 1.1])
    ax2.set_xlabel('Recall', fontsize=8)
    ax2.set_ylabel('Precision', fontsize=8)
    ax2.set_title('Precision-Recall curve', fontsize=8)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)  
    plt.show()
    # Aggregate results across folds
    if use_default_threshold:
        opt_threshold = 0.5
    else:
        # Drop the first entry of thresholds_per_fold that is 0.5
        thresholds_per_fold = thresholds_per_fold[1:]
        opt_threshold = np.median(thresholds_per_fold)
    aggregated_results = {metric: np.nanmean(fold_results[metric].values).round(2) for metric in measures}
    aggregated_results_sd = {metric: np.nanstd(fold_results[metric].values).round(2) for metric in measures}
    # Combining mean and standard deviation
    combined_results = {metric: f"{mean} ± {sd}" for metric, mean in aggregated_results.items() for _, sd in aggregated_results_sd.items() if metric == _}
    # Creating a DataFrame for tabular display
    results_table = pd.DataFrame(list(combined_results.items()), columns=['Metric', 'Result'])
    # Displaying the results
    print("Aggregated Results:")
    print(results_table.to_string(index=False))
    return fold_results, results_table, opt_threshold, feature_importance_list, treebased_feature_importance_list, shap_values_list, fold_results_plt, fold_data, missclassified_samples, y_fold_data, predictions_proba_fold_list

# %% [markdown]
# ##### SHAP summary plot for when the model uses categorical features
# 
# This function resolves the issue of not showing the levels of categorical features on the SHAP summary plot from shap package in Python.

# %%
def categorical_shap_plot(shap_values, data, top_n=10, jitter=0.1, **kwargs):
    """
    This function creates a plot of SHAP values for the top N features in a dataset, where categorical features are displayed as scatter plots with different colors and numerical features are displayed as individual points. The plot includes a colorbar for numerical features and labels for categorical features.

    ## Parameters:
        shap_values (numpy array): Matrix of SHAP values to be plotted.
        data (pandas DataFrame): Dataset containing feature names and values.
        top_n (int, optional): Number of top features to include in the plot. Defaults to 10.
        jitter (float, optional): Jitter value for scatter plots. Defaults to 0.1.

    ## Returns
        fig: Matplotlib figure object containing the SHAP plot.
    """
    # Ensure data and shap_values are consistent
    assert shap_values.shape[1] == data.shape[1], "Mismatch between shap_values and data"

    feature_names = data.columns

    # Calculate the mean absolute SHAP values to rank feature importance
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    sorted_indices = np.argsort(mean_shap_values)[::-1][:top_n]

    top_n_features = feature_names[sorted_indices]

    # an empty dictionary to store categorical features and their categories
    cat_features = {}

    # Identify and store categories for categorical features
    for feature in top_n_features:
        if data[feature].dtype.name in ['category', 'object']:
            categories = data[feature].unique().tolist()
            cat_features[feature] = categories

    # Extract all unique categories for the top n features
    unique_categories = set().union(*cat_features.values())
    num_categories = len(unique_categories)
    cmap_grey = plt.get_cmap('Set1')
    category_colors = cmap_grey(np.linspace(0, 1, num_categories))

    # Create a dictionary to map each category to a distinct marker
    category_marker_dictionary = {}
    category_markers = ['s', 'D', 'P', 'X', '^', 'v', '<', '>', 'H', 'h', '+', 'x', 'p', 'd', '1', '2', '3', '4', '|', '_', ',', '.', '8']

    for category in unique_categories:
        if category_markers:
            category_marker_dictionary[category] = category_markers.pop(0)
        else:
            print("Warning: Not enough markers available.")
            break

    # Calculate the height based on the number of rows
    height = round(np.max([10, np.log(top_n)]))
    max_height = 65535 / 72  # Convert pixels to inches
    if height > max_height:
        height = max_height

    fig, ax = plt.subplots(figsize=(10, height))
    cmap = plt.get_cmap('bwr')

    displayed_categories = set()
    legend_handles = []
    legend_labels = []

    for i, idx in enumerate(sorted_indices):
        feature_shap_values = shap_values[:, idx]
        feature_values = data.iloc[:, idx]
        ax.axhline(i+1, linestyle='--', color='gray', linewidth=0.5)

        if feature_names[idx] in cat_features:
            # Handle categorical features
            for j, category in enumerate(unique_categories):
                mask = (data[feature_names[idx]] == category)
                if np.sum(mask) > 0:
                    jitter_values = jitter * (np.random.randn(np.sum(mask)) - 0.5)
                    ax.scatter(feature_shap_values[mask],
                               [top_n - i] * np.sum(mask) + jitter_values,
                               facecolors=category_colors[j % num_categories],
                               edgecolors='grey',
                               marker=category_marker_dictionary[category], 
                               s=15,
                               alpha=0.7, linewidths=0.5)

                    if category not in displayed_categories:
                        legend_handles.append(plt.Line2D([0], [0], 
                                                         marker=category_marker_dictionary[category], 
                                                         color='white',
                                                         markerfacecolor=category_colors[j % num_categories], 
                                                         markeredgecolor='grey', 
                                                         markersize=5))
                        legend_labels.append(category)
                        displayed_categories.add(category)
        else:
            # Handle numerical features
            numeric_values = np.array(feature_values, dtype=float)
            missing_mask = np.isnan(numeric_values)  # Handle missing values

            # Plot missing values in grey
            if np.any(missing_mask):
                jitter_values_missing = jitter * (np.random.randn(np.sum(missing_mask)) - 0.5)
                ax.scatter(feature_shap_values[missing_mask], 
                        [top_n - i] * np.sum(missing_mask) + jitter_values_missing,
                        c='grey', 
                        marker='x', 
                        edgecolors='grey',
                        alpha=0.7,
                        label='Missing', s=15,
                       linewidths=0.5)
            
            normalized_values = QuantileTransformer(output_distribution='uniform').fit_transform(numeric_values.reshape(-1, 1)).flatten()
            jitter_values = jitter * (np.random.randn(len(feature_shap_values)) - 0.5)
            ax.scatter(feature_shap_values, 
                       [top_n - i] * len(feature_shap_values) + jitter_values,
                       c=normalized_values,
                       cmap=cmap,
                       marker="o",
                       edgecolors='grey',
                       alpha=0.7,
                       s=15,
                       linewidths=0.5)


    # Set y-axis ticks and labels
    ax.set_yticks(range(1, top_n + 1))
    ax.set_yticklabels([feature_names[idx] for idx in sorted_indices[::-1]], rotation='horizontal', fontsize=8)

    # Set x-axis label
    ax.set_xlabel('SHAP values', fontsize=8)

    # Add colorbar for numerical features
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.ax.tick_params(axis='both', labelsize=8)

    # Add legend for categorical features
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='lower right', fontsize=8)

    # Add a midline
    plt.axvline(0, linestyle='--', color='gray', alpha=0.5)
    cbar.set_label(label='Feature value', size=8)

    plt.rcParams.update({'font.size': 8})
    plt.grid(linestyle=':', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='x', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig


# %% [markdown]
# ##### QLattice model
# 
# The QLattice, integrated into the Feyn Python library, represents a cutting-edge approach to supervised machine learning known as symbolic regression. It specializes in identifying the most suitable mathematical models to describe complex datasets. Through an iterative process of training, the QLattice prioritizes simplicity while maintaining high performance.
# 
# More information: https://docs.abzu.ai/docs/guides/getting_started/qlattice

# %%
if "QLattice_mdl" in models_to_include:
    fold_results_QLattice, aggregated_results_QLattice, opt_threshold_QLattice, FI_QLattice, treeFI_QLattice, shap_QLattice, fold_results_plt_Qlattice, _, missclassified_samples_QLattice, y_fold, pp_fold_QLattice = cross_validate_model(model_class='QLattice',
                                                                                                                X=X_train_imputed,
                                                                                                                y=y_train,
                                                                                                                sample_weights=sample_weights,
                                                                                                                random_state=SEED,
                                                                                                                use_default_threshold=use_default_threshold)

# %%
if "QLattice_mdl" in models_to_include:
    # plot permutation-based feature importance
    plot_PFI(PFI_folds=FI_QLattice, X=X_train, model_name="QLattice")

# %%
if "QLattice_mdl" in models_to_include:
    if export_missclassified:
        misclassified_ids = mydata_backup.loc[missclassified_samples_QLattice, 'ID']
        
        misclassified_ids_df = pd.DataFrame(misclassified_ids.tolist(), columns=['Misclassified_IDs'])
        misclassified_ids_df.to_excel('misclassified_ids_QLattice.xlsx', index=False)

# %%
cat_features

# %% [markdown]
# ##### Gaussian Naive Bayes
# 
# Gaussian Naive Bayes (GaussianNB) is a classification algorithm implemented in Python's scikit-learn library. It assumes that the likelihood of features follows a Gaussian distribution. The algorithm estimates parameters using maximum likelihood. In practice, GaussianNB is commonly used for classification tasks when dealing with continuous data.
# 
# Read more here: https://scikit-learn.org/stable/modules/naive_bayes.html
# 

# %%
if "NaiveBayes_mdl" in models_to_include:
    fold_results_NB, aggregated_results_NB, opt_threshold_NB, fi_NB, treeFI_NB, shap_NB, fold_results_plt_NB, fold_data_NB, missclassified_samples_NB, y_fold, pp_fold_NB = cross_validate_model(model_class=GaussianNB,
                                                                                                X=X_train_OHE,
                                                                                                y=y_train,
                                                                                                sample_weights=sample_weights,
                                                                                                random_state=SEED,
                                                                                                use_default_threshold=use_default_threshold)

# %%
if "NaiveBayes_mdl" in models_to_include:
    if export_missclassified:
        misclassified_ids = mydata_backup.loc[missclassified_samples_NB, 'ID']
        
        misclassified_ids_df = pd.DataFrame(misclassified_ids.tolist(), columns=['Misclassified_IDs'])
        misclassified_ids_df.to_excel('misclassified_ids_NB.xlsx', index=False)

# %%
if "NaiveBayes_mdl" in models_to_include:
    # plot permutation-based feature importance
    plot_PFI(PFI_folds=fi_NB, X=X_train_OHE, model_name="NB")

# %%
if "NaiveBayes_mdl" in models_to_include:
    shap_values = np.concatenate([fold.values for fold in shap_NB], axis=0)
    # Concatenate SHAP values and DataFrames
    all_columns = fold_data_NB[0].columns
    fold_data_all = pd.concat([fold for fold in fold_data_NB], axis=0)
    fold_data_all.columns = all_columns
    # SHAP summary plot based on the cross validation results
    shap_summary_plot(shap_values=shap_values[:,:,1], data=fold_data_all, model_name="NB")

# %%
if "NaiveBayes_mdl" in models_to_include:
    
    PFI_median = PFI_median_wrap(fi_NB)
    PFI_median = PFI_median[['Feature', 'Importance']]
    
    # a DataFrame with SHAP values for positive class predictions
    shap_df_positive = pd.DataFrame(shap_values[:,:,1], columns=fold_data_NB[0].columns)

    # Calculate the median absolute SHAP value across folds for each feature
    median_abs_shap_positive = shap_df_positive.abs().median()

    # Sort features by their median absolute SHAP value
    sorted_features_positive = median_abs_shap_positive.sort_values(ascending=False).index

    # Take absolute values of SHAP dataframe
    # Reorder SHAP dataframe based on sorted features
    shap_df_sorted_positive = shap_df_positive[sorted_features_positive].abs()
    # Take absolute values of SHAP dataframe
    shap_df_sorted_positive_T = shap_df_sorted_positive.T

    # Calculate the median importance across samples
    SHAPFI_median = shap_df_sorted_positive_T.median(axis=1)
    SHAPFI_median = pd.DataFrame({'Feature': SHAPFI_median.index, 'Importance': SHAPFI_median.values})
    # normalization
    SHAPFI_median['Importance'] = minmax_scaler.fit_transform(SHAPFI_median[['Importance']])

    # Merge PFI_median, SHAPFI_median, and TFI_median dataframes by "Feature"
    FI_merged_df = PFI_median.merge(SHAPFI_median, on="Feature", how='outer', suffixes=('_PFI', '_SHAP'))

    # Take the mean of importance values across different methods
    FI_merged_df['Normalized_Mean_Importance'] = FI_merged_df[['Importance_PFI', 'Importance_SHAP']].mean(axis=1)

    # Sort features by their mean importance
    FI_merged_df = FI_merged_df.sort_values(by="Normalized_Mean_Importance", ascending=False)
    print(FI_merged_df)



# %%
if "LogisticRegression_mdl" in models_to_include:
    fold_results_LR, aggregated_results_LR, opt_threshold_LR, FI_LR, treeFI_LR, shap_LR, fold_results_plt_LR, fold_data_LR, missclassified_samples_LR, y_fold, pp_fold_LR = cross_validate_model(model_class=LogisticRegression,
                                                                                                X=X_train_OHE,
                                                                                                y=y_train,
                                                                                                sample_weights=sample_weights,
                                                                                                random_state=SEED,
                                                                                                use_default_threshold=use_default_threshold)

# %%
if "LogisticRegression_mdl" in models_to_include:
    if export_missclassified:
        misclassified_ids = mydata_backup.loc[missclassified_samples_LR, 'ID']
        
        misclassified_ids_df = pd.DataFrame(misclassified_ids.tolist(), columns=['Misclassified_IDs'])
        misclassified_ids_df.to_excel('misclassified_ids_LR.xlsx', index=False)

# %%
if "LogisticRegression_mdl" in models_to_include:
    # plot permutation-based feature importance
    plot_PFI(PFI_folds=FI_LR, X=X_train_OHE, model_name="LR")

# %%
if "LogisticRegression_mdl" in models_to_include:
    # Concatenate SHAP values and DataFrames
    all_columns = fold_data_LR[0].columns
    fold_data_all = pd.concat([fold for fold in fold_data_LR], axis=0)
    fold_data_all.columns = all_columns
    shap_values = np.concatenate([fold for fold in shap_LR], axis=0)
    # SHAP summary plot based on the cross validation results
    shap_summary_plot(shap_values=shap_values, data=fold_data_all, model_name="LR")

# %%
if "LogisticRegression_mdl" in models_to_include:
    
    # permutation-based feature importance
    PFI_median = PFI_median_wrap(FI_LR)
    PFI_median = PFI_median[['Feature', 'Importance']]
    # Create a DataFrame with SHAP values for positive class predictions
    shap_df_positive = pd.DataFrame(shap_values, columns=fold_data_LR[0].columns)

    # Calculate the median absolute SHAP value across folds for each feature
    median_abs_shap_positive = shap_df_positive.abs().median()

    # Sort features by their median absolute SHAP value
    sorted_features_positive = median_abs_shap_positive.sort_values(ascending=False).index

    # Take absolute values of SHAP dataframe
    # Reorder SHAP dataframe based on sorted features
    shap_df_sorted_positive = shap_df_positive[sorted_features_positive].abs()
    # Take absolute values of SHAP dataframe
    shap_df_sorted_positive_T = shap_df_sorted_positive.T

    # Calculate the median importance across samples
    SHAPFI_median = shap_df_sorted_positive_T.median(axis=1)
    SHAPFI_median = pd.DataFrame({'Feature': SHAPFI_median.index, 'Importance': SHAPFI_median.values})
    # normalization
    SHAPFI_median['Importance'] = minmax_scaler.fit_transform(SHAPFI_median[['Importance']])

    # Merge PFI_median, SHAPFI_median, and TFI_median dataframes by "Feature"
    FI_merged_df = PFI_median.merge(SHAPFI_median, on="Feature", how='outer', suffixes=('_PFI', '_SHAP'))

    # Take the mean of importance values across different methods
    FI_merged_df['Normalized_Mean_Importance'] = FI_merged_df[['Importance_PFI', 'Importance_SHAP']].mean(axis=1)

    # Sort features by their mean importance
    FI_merged_df = FI_merged_df.sort_values(by="Normalized_Mean_Importance", ascending=False)
    print(FI_merged_df)



# %% [markdown]
# ##### Random Forest Classifier (RF)
# 
# The `RandomForestClassifier`, part of the `sklearn.ensemble` module in scikit-learn, is a versatile and powerful tool for classification tasks. It operates as a meta estimator that fits multiple decision tree classifiers on various sub-samples of the dataset, using averaging to enhance predictive accuracy and mitigate overfitting. By default, the classifier uses bootstrap sampling (`bootstrap=True`), and each tree is built using a random subset of features (`max_features='sqrt'`).
# 
# Key parameters include:
# - `n_estimators`: Number of trees in the forest.
# - `criterion`: Function to measure the quality of a split (`'gini'` or `'entropy'`).
# - `max_depth`: Maximum depth of the trees.
# - `min_samples_split`: Minimum number of samples required to split an internal node.
# - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
# - `class_weight`: Adjusts weights inversely proportional to class frequencies to handle imbalanced datasets.
# 
# The `RandomForestClassifier` is highly customizable, allowing for fine-tuning to suit specific datasets and classification challenges. It provides robust performance, especially in scenarios where feature interactions are complex or when the dataset contains a mix of categorical and numerical features.
# 
# Read more here: [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
# 

# %%
if "RandomForest_mdl" in models_to_include:
    fold_results_rf, aggregated_results_rf, opt_threshold_rf, FI_rf, treeFI_rf, shap_rf, fold_results_plt_rf, fold_data_rf, missclassified_samples_rf, y_fold, pp_fold_rf = cross_validate_model(model_class=RandomForestClassifier,
                                                                                                    X = X_train_OHE,
                                                                                                    y = y_train,
                                                                                                    sample_weights = sample_weights,
                                                                                                    random_state = SEED,
                                                                                                    use_default_threshold=use_default_threshold)


# %%
if "RandomForest_mdl" in models_to_include:
    if export_missclassified:
        misclassified_ids = mydata_backup.loc[missclassified_samples_rf, 'ID']
        
        misclassified_ids_df = pd.DataFrame(misclassified_ids.tolist(), columns=['Misclassified_IDs'])
        misclassified_ids_df.to_excel('misclassified_ids_rf.xlsx', index=False)

# %%
if "RandomForest_mdl" in models_to_include:
    # plot permutation-based feature importance
    plot_PFI(PFI_folds=FI_rf, X=X_train_OHE, model_name="RF")
    

# %%
if "RandomForest_mdl" in models_to_include:
    # plot tree-based feature importance
    plot_TFI(X=X_train_OHE, tree_FI=treeFI_rf, model_name="RF")
    


# %%
if "RandomForest_mdl" in models_to_include:
    # Concatenate SHAP values and DataFrames
    shap_values = np.concatenate([fold[:,:,1] for fold in shap_rf], axis=0)
    all_columns = fold_data_rf[0].columns
    fold_data_all = pd.concat([fold for fold in fold_data_rf], axis=0)
    fold_data_all.columns = all_columns
    # SHAP summary plot based on the cross validation results
    shap_summary_plot(shap_values=shap_values, data=fold_data_all, model_name="RF")

# %%
if "RandomForest_mdl" in models_to_include:
    
    # permutation-based feature importance
    PFI_median = PFI_median_wrap(FI_rf)
    PFI_median = PFI_median[['Feature', 'Importance']]
    
    # Extract feature names from X_train_OHE
    feature_names = X_train_OHE.columns
    
    # a DataFrame with SHAP values for positive class predictions
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    shap_df_sorted_positive_T = shap_df.T
    
    # Calculate the median importance across samples
    SHAPFI_median = shap_df_sorted_positive_T.abs().median(axis=1)
    SHAPFI_median = pd.DataFrame({'Feature': SHAPFI_median.index, 'Importance': SHAPFI_median.values})
    # normalization
    SHAPFI_median['Importance'] = minmax_scaler.fit_transform(SHAPFI_median[['Importance']])

    # Combine feature importances from all folds into a single DataFrame TFI: tree-based feature importance
    TFI = pd.concat([pd.DataFrame({'Feature': feature_names, 'Importance': fold}) for fold in treeFI_rf], axis=0, ignore_index=True)

    # Calculate the median importance across folds for each feature
    TFI_median = TFI.groupby("Feature")["Importance"].median().reset_index()

    # Sort features by their median importance
    TFI_median = TFI_median.sort_values(by="Importance", ascending=False)
    TFI_median['Importance'] = minmax_scaler.fit_transform(TFI_median[['Importance']])
    # Rename the 'Importance' column in TFI_median to 'Importance_TFI'
    TFI_median = TFI_median.rename(columns={'Importance': 'Importance_TFI'})
    # Merge PFI_median, SHAPFI_median, and TFI_median dataframes by "Feature"
    FI_merged_df = PFI_median.merge(SHAPFI_median, on="Feature", how='outer', suffixes=('_PFI', '_SHAP')).merge(TFI_median, on="Feature", how='outer')

    # Take the mean of importance values across different methods
    FI_merged_df['Normalized_Mean_Importance'] = FI_merged_df[['Importance_PFI', 'Importance_SHAP', 'Importance_TFI']].mean(axis=1)

    # Sort features by their mean importance
    FI_merged_df = FI_merged_df.sort_values(by="Normalized_Mean_Importance", ascending=False)
    print(FI_merged_df)



# %% [markdown]
# ##### Histogram-based Gradient Boosting Classification Tree (HGBC)
# 
# The HistGradientBoostingClassifier, part of the scikit-learn library, offers a histogram-based approach to gradient boosting for classification tasks. Notably, it exhibits significantly faster performance on large datasets (with n_samples >= 10,000) compared to the traditional GradientBoostingClassifier. The implementation of HistGradientBoostingClassifier is inspired by LightGBM and offers various parameters for customization, such as learning rate, maximum depth of trees, and early stopping criteria. This classifier is an excellent choice for classification tasks with large datasets, providing both speed and accuracy.
# 
# Read more here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

# %%
if "HistGBC_mdl" in models_to_include:
    fold_results_HGBC, aggregated_results_HGBC, opt_threshold_HGBC, FI_HGBC, treeFI_HGBC, shap_HGBC, fold_results_plt_HGBC, fold_data_HGBC, missclassified_samples_HGBC, y_fold, pp_fold_HGBC = cross_validate_model(model_class=HistGradientBoostingClassifier,
                                                                                                        X = X_train_OHE,
                                                                                                        y = y_train,
                                                                                                        sample_weights = sample_weights,
                                                                                                        random_state = SEED,
                                                                                                    use_default_threshold=use_default_threshold)

# %%
if "HistGBC_mdl" in models_to_include:
    if export_missclassified:
        misclassified_ids = mydata_backup.loc[missclassified_samples_HGBC, 'ID']
        
        misclassified_ids_df = pd.DataFrame(misclassified_ids.tolist(), columns=['Misclassified_IDs'])
        misclassified_ids_df.to_excel('misclassified_ids_HGBC.xlsx', index=False)

# %%
if "HistGBC_mdl" in models_to_include:
    # plot permutation-based feature importance
    plot_PFI(PFI_folds=FI_HGBC, X=X_train_OHE, model_name="HGBC")


# %%
if "HistGBC_mdl" in models_to_include:
    shap_values = np.concatenate([fold for fold in shap_HGBC], axis=0)
    all_columns = fold_data_HGBC[0].columns
    fold_data_all = pd.concat([fold for fold in fold_data_HGBC], axis=0)
    fold_data_all.columns = all_columns
    # SHAP summary plot based on the cross validation results
    shap_summary_plot(shap_values=shap_values, data=fold_data_all, model_name="HGBC")

# %%
if "HistGBC_mdl" in models_to_include:
    
    # permutation-based feature importance
    PFI_median = PFI_median_wrap(FI_HGBC)
    PFI_median = PFI_median[['Feature', 'Importance']]
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    shap_df_sorted_positive_T = shap_df.T
    
    # Calculate the median importance across samples
    SHAPFI_median = shap_df_sorted_positive_T.abs().median(axis=1)
    SHAPFI_median = pd.DataFrame({'Feature': SHAPFI_median.index, 'Importance': SHAPFI_median.values})
    # normalization
    SHAPFI_median['Importance'] = minmax_scaler.fit_transform(SHAPFI_median[['Importance']])

    # Merge PFI_median, SHAPFI_median, and TFI_median dataframes by "Feature"
    FI_merged_df = PFI_median.merge(SHAPFI_median, on="Feature", how='outer', suffixes=('_PFI', '_SHAP'))

    # Take the mean of importance values across different methods
    FI_merged_df['Normalized_Mean_Importance'] = FI_merged_df[['Importance_PFI', 'Importance_SHAP']].mean(axis=1)

    # Sort features by their mean importance
    FI_merged_df = FI_merged_df.sort_values(by="Normalized_Mean_Importance", ascending=False)
    print(FI_merged_df)



# %% [markdown]
# ##### Light gradient-boosting machine (LightGBM)
# 
# LightGBM represents an open-source, distributed, and high-performance gradient boosting framework, engineered by Microsoft, to tackle machine learning challenges with precision and efficiency. It operates on decision trees, finely tuned to optimize model efficiency while minimizing memory consumption. A key innovation is the Gradient-based One-Side Sampling (GOSS) method, which intelligently retains instances with significant gradients during training, thereby optimizing memory usage and training duration. Additionally, LightGBM employs histogram-based algorithms for rapid and resource-efficient tree construction. These advanced techniques, alongside optimizations such as leaf-wise tree growth and streamlined data storage formats, collectively contribute to LightGBM's remarkable efficiency and competitive edge in the realm of gradient boosting frameworks.
# 
# Read more here: https://lightgbm.readthedocs.io/en/stable/
# 

# %%
if "LightGBM_mdl" in models_to_include:
    fold_results_LGBM, aggregated_results_LGBM, opt_threshold_LGBM, FI_LGBM, treeFI_LGBM, shap_LGBM, fold_results_plt_LGBM, fold_data_LGBM, missclassified_samples_LGBM, y_fold, pp_fold_LGBM = cross_validate_model(model_class=lgb.LGBMClassifier,
                                                                                                        X = X_train,
                                                                                                        y = y_train,
                                                                                                        sample_weights = sample_weights,
                                                                                                        random_state = SEED,
                                                                                                        use_default_threshold=use_default_threshold)

# %%
if "LightGBM_mdl" in models_to_include:
    if export_missclassified:
        misclassified_ids = mydata_backup.loc[missclassified_samples_LGBM, 'ID']
        
        misclassified_ids_df = pd.DataFrame(misclassified_ids.tolist(), columns=['Misclassified_IDs'])
        misclassified_ids_df.to_excel('misclassified_ids_LGBM.xlsx', index=False)

# %%
if "LightGBM_mdl" in models_to_include:
    # plot permutation-based feature importance
    plot_PFI(PFI_folds=FI_LGBM, X=X_train, model_name="LGBM")


# %%
if "LightGBM_mdl" in models_to_include:
    # plot tree-based feature importance
    plot_TFI(X=X_train, tree_FI=treeFI_LGBM, model_name="LGBM")

# %%
if "LightGBM_mdl" in models_to_include:
    # each fold contains two arrays: one for the SHAP values of the negative class predictions (index 0) and one for the SHAP values of the positive class predictions (index 1). 
    # Therefore, to extract the arrays for the positive class predictions, you should use index 1.
    shap_values = np.concatenate([fold for fold in shap_LGBM], axis=0)
    all_columns = fold_data_LGBM[0].columns
    fold_data_all = pd.concat([fold for fold in fold_data_LGBM], axis=0)
    fold_data_all.columns = all_columns
    # SHAP summary plot based on the cross validation results
    shap_summary_plot(shap_values=shap_values, data=fold_data_all, model_name="LGBM")

# %%
if "LightGBM_mdl" in models_to_include:
    
    # permutation-based feature importance
    PFI_median = PFI_median_wrap(FI_LGBM)
    PFI_median = PFI_median[['Feature', 'Importance']]
    shap_values_positive = np.concatenate([fold[1] for fold in shap_LGBM], axis=0)
    feature_names = X_train.columns

    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    shap_df_sorted_positive_T = shap_df.T
    
    # Calculate the median importance across samples
    SHAPFI_median = shap_df_sorted_positive_T.abs().median(axis=1)
    SHAPFI_median = pd.DataFrame({'Feature': SHAPFI_median.index, 'Importance': SHAPFI_median.values})
    # normalization
    SHAPFI_median['Importance'] = minmax_scaler.fit_transform(SHAPFI_median[['Importance']])

    # Extract feature names from X_train
    feature_names = X_train.columns

    # Combine feature importances from all folds into a single DataFrame TFI: tree-based feature importance
    TFI = pd.concat([pd.DataFrame({'Feature': feature_names, 'Importance': fold}) for fold in treeFI_LGBM], axis=0, ignore_index=True)

    # Calculate the median importance across folds for each feature
    TFI_median = TFI.groupby("Feature")["Importance"].median().reset_index()

    # Sort features by their median importance
    TFI_median = TFI_median.sort_values(by="Importance", ascending=False)
    TFI_median['Importance'] = minmax_scaler.fit_transform(TFI_median[['Importance']])
    # Rename the 'Importance' column in TFI_median to 'Importance_TFI'
    TFI_median = TFI_median.rename(columns={'Importance': 'Importance_TFI'})
    # Merge PFI_median, SHAPFI_median, and TFI_median dataframes by "Feature"
    FI_merged_df = PFI_median.merge(SHAPFI_median, on="Feature", how='outer', suffixes=('_PFI', '_SHAP')).merge(TFI_median, on="Feature", how='outer')

    # Take the mean of importance values across different methods
    FI_merged_df['Normalized_Mean_Importance'] = FI_merged_df[['Importance_PFI', 'Importance_SHAP', 'Importance_TFI']].mean(axis=1)

    # Sort features by their mean importance
    FI_merged_df = FI_merged_df.sort_values(by="Normalized_Mean_Importance", ascending=False)
    print(FI_merged_df)



# %% [markdown]
# ##### Categorical boosting (CATBoost)
# 
# CatBoost is a supervised machine learning method utilized for classification and regression tasks, particularly useful for handling categorical data without the need for extensive preprocessing. Employing gradient boosting, CatBoost iteratively constructs decision trees to refine predictions, achieving enhanced accuracy over time. Notably, CatBoost employs ordered encoding to effectively handle categorical features, utilizing target statistics from all rows to inform encoding decisions. Additionally, it introduces symmetric trees, ensuring uniformity in split conditions at each depth level. Compared to similar methods like XGBoost, CatBoost have often demonstrates superior performance across datasets of varying sizes, retaining key features such as cross-validation, regularization, and support for missing values.
# 
# Read more here: https://catboost.ai/docs/features/categorical-features
# 

# %%
cat_features = list(cat_features)

# %%
cat_features

# %%
if "CatBoost_mdl" in models_to_include:
    fold_results_CB, aggregated_results_CB, opt_threshold_CB, FI_CB, treeFI_CB, shap_CB, fold_results_plt_CB, fold_data_CB, missclassified_samples_CB, y_fold, pp_fold_CB = cross_validate_model(model_class= cb.CatBoostClassifier,
                                                                                                X = X_train,
                                                                                                y = y_train,
                                                                                                sample_weights = sample_weights,
                                                                                                random_state = SEED,
                                                                                                cat_features = cat_features,
                                                                                                use_default_threshold = use_default_threshold)

# %%
if "CatBoost_mdl" in models_to_include:
    if export_missclassified:
        misclassified_ids = mydata_backup.loc[missclassified_samples_CB, 'ID']
        
        misclassified_ids_df = pd.DataFrame(misclassified_ids.tolist(), columns=['Misclassified_IDs'])
        misclassified_ids_df.to_excel('misclassified_ids_CB.xlsx', index=False)

# %%
if "CatBoost_mdl" in models_to_include:
    # plot permutation-based feature importance
    plot_PFI(PFI_folds=FI_CB, X=X_train, model_name="CB")

# %%
if "CatBoost_mdl" in models_to_include:
    # plot tree-based feature importance
    plot_TFI(X=X_train, tree_FI=treeFI_CB, model_name="CB")

# %%
if "CatBoost_mdl" in models_to_include:
    shap_values = np.concatenate([fold for fold in shap_CB], axis=0)
    all_columns = fold_data_CB[0].columns
    fold_data_all = pd.concat([fold for fold in fold_data_CB], axis=0)
    fold_data_all.columns = all_columns
    # SHAP summary plot based on the cross validation results
    shap_summary_plot(shap_values=shap_values, data=fold_data_all, model_name="CB")

# %%
if "CatBoost_mdl" in models_to_include:
    
    # permutation-based feature importance
    PFI_median = PFI_median_wrap(FI_CB)
    PFI_median = PFI_median[['Feature', 'Importance']]
    shap_values_positive = np.concatenate([fold for fold in shap_CB], axis=0)
    feature_names = X_train.columns
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    shap_df_sorted_positive_T = shap_df.T
    
    # Calculate the median importance across samples
    SHAPFI_median = shap_df_sorted_positive_T.abs().median(axis=1)
    SHAPFI_median = pd.DataFrame({'Feature': SHAPFI_median.index, 'Importance': SHAPFI_median.values})
    # normalization
    SHAPFI_median['Importance'] = minmax_scaler.fit_transform(SHAPFI_median[['Importance']])

    # Extract feature names from X_train
    feature_names = X_train.columns

    # Combine feature importances from all folds into a single DataFrame TFI: tree-based feature importance
    TFI = pd.concat([pd.DataFrame({'Feature': feature_names, 'Importance': fold}) for fold in treeFI_CB], axis=0, ignore_index=True)

    # Calculate the median importance across folds for each feature
    TFI_median = TFI.groupby("Feature")["Importance"].median().reset_index()

    # Sort features by their median importance
    TFI_median = TFI_median.sort_values(by="Importance", ascending=False)
    TFI_median['Importance'] = minmax_scaler.fit_transform(TFI_median[['Importance']])
    # Rename the 'Importance' column in TFI_median to 'Importance_TFI'
    TFI_median = TFI_median.rename(columns={'Importance': 'Importance_TFI'})
    # Merge PFI_median, SHAPFI_median, and TFI_median dataframes by "Feature"
    FI_merged_df = PFI_median.merge(SHAPFI_median, on="Feature", how='outer', suffixes=('_PFI', '_SHAP')).merge(TFI_median, on="Feature", how='outer')

    # Take the mean of importance values across different methods
    FI_merged_df['Normalized_Mean_Importance'] = FI_merged_df[['Importance_PFI', 'Importance_SHAP', 'Importance_TFI']].mean(axis=1)

    # Sort features by their mean importance
    FI_merged_df = FI_merged_df.sort_values(by="Normalized_Mean_Importance", ascending=False)
    print(FI_merged_df)



# %%
y_fold_all = pd.concat([fold for fold in y_fold], axis=0)

# %% [markdown]
# ##### summary of the cross validation results

# %%
# Define the available models and their corresponding DataFrames
models_dict = {}
if "QLattice_mdl" in models_to_include:
    models_dict["QLattice_mdl"] = aggregated_results_QLattice
if "RandomForest_mdl" in models_to_include:
    models_dict["RandomForest_mdl"] = aggregated_results_rf
if "LightGBM_mdl" in models_to_include:
    models_dict["LightGBM_mdl"] = aggregated_results_LGBM
if "NaiveBayes_mdl" in models_to_include:
    models_dict["NaiveBayes_mdl"] = aggregated_results_NB
if "CatBoost_mdl" in models_to_include:
    models_dict["CatBoost_mdl"] = aggregated_results_CB
if "LogisticRegression_mdl" in models_to_include:
    models_dict["LogisticRegression_mdl"] = aggregated_results_LR
if "HistGBC_mdl" in models_to_include:
    models_dict["HistGBC_mdl"] = aggregated_results_HGBC

# Initialize an empty list to store selected models' DataFrames
selected_models = []

# Select the DataFrames based on user's choice
for model_name in models_to_include:
    if model_name in models_dict:
        selected_models.append(models_dict[model_name])

# Set 'Metric' as the index for each model's DataFrame
for model in selected_models:
    model.set_index('Metric', inplace=True)

# Concatenate the DataFrames along the columns
aggregated_results_all = pd.concat(selected_models, axis=1)

# Set the column names based on the selected models
aggregated_results_all.columns = models_to_include


print(aggregated_results_all)

# Save the results to an Excel file
aggregated_results_all.to_excel('aggregated_results_all.xlsx', index=True)

# %%
# Function to extract mean values from strings
def extract_mean(value):
    """
    Extract the mean value from a string.

    This function uses regular expressions to search for a decimal number in the input string.
    If a match is found, it returns the extracted value as a float. Otherwise, it returns None.

    Parameters:
        value (str): The input string to extract the mean value from.

    ## Returns
        float or None: The extracted mean value as a float, or None if no match is found.
    """
    mean = re.search(r'(\d+\.\d+)', value)
    if mean:
        return float(mean.group())
    else:
        return None

# Extracting mean values from the DataFrame
mean_values = aggregated_results_all.applymap(extract_mean)

# Calculate mean values for MCC, ROCAUC, and PRAUC for each model
mean_values_per_model = mean_values.T.groupby(level=0).mean()

# Calculate the average of MCC, ROCAUC, and PRAUC for each model
mean_values_per_model['MRPAvg'] = mean_values_per_model[['MCC', 'ROCAUC', 'PRAUC']].mean(axis=1)

print(mean_values_per_model)

# Find the model with the highest average of MCC, ROCAUC, and PRAUC (termed as MRPavg)
best_model = mean_values_per_model['MRPAvg'].idxmax()


# %%
# Create a dictionary to map model abbreviations to full names
model_names = {
    'RandomForest_mdl': 'rf',
    'HistGBC_mdl': 'HGBC',
    'LogisticRegression_mdl': 'LR',
    'CatBoost_mdl': 'CB',
    'NaiveBayes_mdl': 'NB',
    'LightGBM_mdl': 'LGBM',
    'QLattice_mdl' : 'QLattice'
}

# Get the full name of the best model
best_model_name = model_names.get(best_model)

# Print the best model
print(f"Model with the highest average of MCC, ROCAUC, and PRAUC: {best_model_name} ({best_model})")


# %% [markdown]
# ##### Statistical test to compare the performance of the models on cross validation
# 
# Note: this is done only for AUC but can be extended for other measures.
# 
# Using the Kruskal-Wallis test allows you to compare the mean AUC values of multiple models without relying on the assumptions of normality and homogeneity of variances. It provides a robust nonparametric approach to assess whether there are significant differences between the models in terms of their performance.
# 
# The Kruskal-Wallis test is a nonparametric equivalent of the ANOVA test and is suitable when the assumptions of normality and homogeneity of variances are not met.
# 
# Here's an outline of the steps to perform a Kruskal-Wallis test:
# 
# Null Hypothesis (H0): The mean AUC values of all models are equal.
# Alternative Hypothesis (HA): At least one mean AUC value is significantly different from the others.
# 
# Collect the mean AUC values of each model obtained from cross-validation.
# 
# Perform a Kruskal-Wallis test, which tests for differences in the distribution of a continuous variable (AUC) among multiple groups (models).
# 
# Calculate the test statistic (H-statistic) and obtain the corresponding p-value.
# 
# Interpret the results:
# 
# If the p-value is less than a predetermined significance level (e.g., 0.05), reject the null hypothesis. It suggests that at least one model's mean AUC value is significantly different from the others.
# If the p-value is greater than the significance level, fail to reject the null hypothesis. It indicates that there is no significant difference between the mean AUC values of the models.
# If the null hypothesis is rejected (i.e., significant differences exist), you can perform post-hoc tests to determine which specific models are significantly different from each other. Common post-hoc tests for nonparametric data include the Dunn test or the Bonferroni correction.

# %%
# a dictionary to map model names to their corresponding AUC values
# Define the available models and their corresponding fold results
model_auc_dict = {}
if "QLattice_mdl" in models_to_include:
    model_auc_dict['QLattice_mdl'] = fold_results_QLattice['ROCAUC'].values
if "RandomForest_mdl" in models_to_include:
    model_auc_dict['RandomForest_mdl'] = fold_results_rf['ROCAUC'].values
if "LightGBM_mdl" in models_to_include:
    model_auc_dict['LightGBM_mdl'] = fold_results_LGBM['ROCAUC'].values
if "NaiveBayes_mdl" in models_to_include:
    model_auc_dict['NaiveBayes_mdl'] = fold_results_NB['ROCAUC'].values
if "CatBoost_mdl" in models_to_include:
    model_auc_dict['CatBoost_mdl'] = fold_results_CB['ROCAUC'].values
if "LogisticRegression_mdl" in models_to_include:
    model_auc_dict['LogisticRegression_mdl'] = fold_results_LR['ROCAUC'].values
if "HistGBC_mdl" in models_to_include:
    model_auc_dict['HistGBC_mdl'] = fold_results_HGBC['ROCAUC'].values

# Initialize an empty list to store selected AUC values
selected_auc_values = []

# Select the AUC values based on user's choice
for model_name in models_to_include:
    if model_name in model_auc_dict:
        selected_auc_values.append(model_auc_dict[model_name])

# Perform Kruskal-Wallis test
statistic, p_value = kruskal(*selected_auc_values)

# Interpret the results
alpha = 0.05  # Significance level

if p_value < alpha:
    print("At least one model's mean AUC value is significantly different from the others.")
else:
    print("No significant difference between the mean AUC values of the models.")

print(f"Kruskal-Wallis test statistic: {statistic}")
print(f"P-value: {p_value}")


# %% [markdown]
# #### Model Uncertainty Reduction (MUR) - optional
# 
# The following code chunk identifies a margin around the prediction probability threshold and SHAP percentile to filter out samples where the predicted probabilities and SHAP values fall within a predefined uncertainty margin. This margin is determined through a grid search over a limited search space for SHAP percentile values and prediction probability margins in binary classification models. The approach ensures that the number of discarded samples does not exceed a specified maximum percentage, thereby balancing the trade-off between model uncertainty reduction and sample retention. This trade-off involves maintaining a high number of samples while ensuring the model has high certainty in its predictions.

# %%
if model_uncertainty_reduction:
    # Initialize an empty list to store selected models' DataFrames
    selected_models = []

    # Calculate SHAP values for the positive class
    positive_class_index = 1
    
    # Define the probability margin around the threshold
    # probability_threshold = 0.5
    margin_grid = [0.01, 0.02, 0.05, 0.1] # Margin from the prediction probability threshold
    SHAP_percentile_grid = [1, 2, 5, 10, 20] # Percentile of absolute SHAP values
    max_sample_loss_perc = 20 # Maximum percentage of samples that can be discarded

    aggregated_CV_results_filtered_mdl = pd.DataFrame()
    # Iterate over the combinations of margin and SHAP_percentile
    for margin in margin_grid:
        for SHAP_percentile in SHAP_percentile_grid:

            for selected_model in models_to_include:
                if selected_model=="HistGBC_mdl":
                    fold_data_all_OHE = pd.concat([fold for fold in fold_data_HGBC], axis=0)
                    shap_values = np.concatenate([fold for fold in shap_HGBC], axis=0)
                    predicted_probabilities = np.concatenate([fold for fold in pp_fold_HGBC], axis=0)
                    shap_sum = shap_values.sum(axis=1)
                    shap_sum_abs = np.abs(shap_sum)
                    SHAP_thr_HGBC = np.percentile(shap_sum_abs, SHAP_percentile)

                    X_train_filtered_shap = fold_data_all_OHE[(shap_sum_abs > SHAP_thr_HGBC) & 
                                                        ((predicted_probabilities < (opt_threshold_HGBC - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_HGBC + margin)))]
                    y_train_filtered_shap = y_fold_all[(shap_sum_abs > SHAP_thr_HGBC) & 
                                                    ((predicted_probabilities < (opt_threshold_HGBC - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_HGBC + margin)))]
                    pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_HGBC) & 
                                                    ((predicted_probabilities < (opt_threshold_HGBC - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_HGBC + margin)))]
                    pc_filtered = np.where(pp_filtered >= opt_threshold_HGBC, True, False)
                        
                elif selected_model=="RandomForest_mdl":
                    fold_data_all_OHE = pd.concat([fold for fold in fold_data_rf], axis=0)
                    shap_values = np.concatenate([fold[1] for fold in shap_rf], axis=0)
                    predicted_probabilities = np.concatenate([fold for fold in pp_fold_rf], axis=0)
                    shap_sum = shap_values.sum(axis=1)
                    shap_sum_abs = np.abs(shap_sum)
                    SHAP_thr_rf = np.percentile(shap_sum_abs, SHAP_percentile)

                    X_train_filtered_shap = fold_data_all_OHE[(shap_sum_abs > SHAP_thr_rf) & 
                                                        ((predicted_probabilities < (opt_threshold_rf - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_rf + margin)))]
                    y_train_filtered_shap = y_fold_all[(shap_sum_abs > SHAP_thr_rf) & 
                                                    ((predicted_probabilities < (opt_threshold_rf - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_rf + margin)))]
                    pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_rf) & 
                                                    ((predicted_probabilities < (opt_threshold_rf - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_rf + margin)))]
                    pc_filtered = np.where(pp_filtered >= opt_threshold_rf, True, False)
                    
                elif selected_model=="CatBoost_mdl":
                    fold_data_all = pd.concat([fold for fold in fold_data_CB], axis=0)
                    shap_values = np.concatenate([fold for fold in shap_CB], axis=0)
                    shap_sum = shap_values.sum(axis=1)
                    shap_sum_abs = np.abs(shap_sum)
                    SHAP_thr_CB = np.percentile(shap_sum_abs, SHAP_percentile)
                    predicted_probabilities = np.concatenate([fold for fold in pp_fold_CB], axis=0)

                    X_train_filtered_shap = fold_data_all[(shap_sum_abs > SHAP_thr_CB) & 
                                                    ((predicted_probabilities < (opt_threshold_CB - margin)) | 
                                                    (predicted_probabilities > (opt_threshold_CB + margin)))]
                    y_train_filtered_shap = y_fold_all[(shap_sum_abs > SHAP_thr_CB) & 
                                                    ((predicted_probabilities < (opt_threshold_CB - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_CB + margin)))]
                    pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_CB) & 
                                                    ((predicted_probabilities < (opt_threshold_CB - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_CB + margin)))]
                    pc_filtered = np.where(pp_filtered >= opt_threshold_CB, True, False)
                    
                elif selected_model=="LightGBM_mdl":
                    fold_data_all = pd.concat([fold for fold in fold_data_LGBM], axis=0)
                    shap_values = np.concatenate([fold[1] for fold in shap_LGBM], axis=0)
                    shap_sum = shap_values.sum(axis=1)
                    shap_sum_abs = np.abs(shap_sum)
                    SHAP_thr_LGBM = np.percentile(shap_sum_abs, SHAP_percentile)
                    predicted_probabilities = np.concatenate([fold for fold in pp_fold_LGBM], axis=0)
                    X_train_filtered_shap = fold_data_all[(shap_sum_abs > SHAP_thr_LGBM) & 
                                                        ((predicted_probabilities < (opt_threshold_LGBM - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_LGBM + margin)))]
                    y_train_filtered_shap = y_fold_all[(shap_sum_abs > SHAP_thr_LGBM) & 
                                                    ((predicted_probabilities < (opt_threshold_LGBM - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_LGBM + margin)))]
                    pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_LGBM) & 
                                                    ((predicted_probabilities < (opt_threshold_LGBM - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_LGBM + margin)))]
                    pc_filtered = np.where(pp_filtered >= opt_threshold_LGBM, True, False)
                    
                elif selected_model=="LogisticRegression_mdl":
                    fold_data_all_OHE = pd.concat([fold for fold in fold_data_LR], axis=0)
                    shap_values = np.concatenate([fold for fold in shap_LR], axis=0)
                    predicted_probabilities = np.concatenate([fold for fold in pp_fold_LR], axis=0)
                    shap_sum = shap_values.sum(axis=1)
                    shap_sum_abs = np.abs(shap_sum)
                    SHAP_thr_LR = np.percentile(shap_sum_abs, SHAP_percentile)

                    X_train_filtered_shap = fold_data_all_OHE[(shap_sum_abs > SHAP_thr_LR) & 
                                                        ((predicted_probabilities < (opt_threshold_LR - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_LR + margin)))]
                    y_train_filtered_shap = y_fold_all[(shap_sum_abs > SHAP_thr_LR) & 
                                                    ((predicted_probabilities < (opt_threshold_LR - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_LR + margin)))]
                    pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_LR) & 
                                                    ((predicted_probabilities < (opt_threshold_LR - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_LR + margin)))]
                    pc_filtered = np.where(pp_filtered >= opt_threshold_LR, True, False)                   
                   
                elif selected_model=="NaiveBayes_mdl":
                    fold_data_all_OHE = pd.concat([fold for fold in fold_data_NB], axis=0)
                    shap_values = np.concatenate([fold.values for fold in shap_NB], axis=0)
                    predicted_probabilities = np.concatenate([fold for fold in pp_fold_NB], axis=0)
                    shap_sum = shap_values[:,:,1].sum(axis=1)
                    shap_sum_abs = np.abs(shap_sum)
                    SHAP_thr_NB = np.percentile(shap_sum_abs, SHAP_percentile)

                    X_train_filtered_shap = fold_data_all_OHE[(shap_sum_abs > SHAP_thr_NB) & 
                                                        ((predicted_probabilities < (opt_threshold_NB - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_NB + margin)))]
                    y_train_filtered_shap = y_fold_all[(shap_sum_abs > SHAP_thr_NB) & 
                                                    ((predicted_probabilities < (opt_threshold_NB - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_NB + margin)))]
                    pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_NB) & 
                                                    ((predicted_probabilities < (opt_threshold_NB - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_NB + margin)))]
                    pc_filtered = np.where(pp_filtered >= opt_threshold_NB, True, False)
                    
                elif selected_model=="QLattice_mdl":
                    fold_data_all_OHE = pd.concat([fold for fold in fold_data_QLattice], axis=0)
                    predicted_probabilities = np.concatenate([fold for fold in pp_fold_QLattice], axis=0)
                    X_train_filtered_shap = fold_data_all_OHE[((predicted_probabilities < (opt_threshold_QLattice - margin)) | 
                                                    (predicted_probabilities > (opt_threshold_QLattice + margin)))]
                    y_train_filtered_shap = y_fold_all[((predicted_probabilities < (opt_threshold_QLattice - margin)) | 
                                                        (predicted_probabilities > (opt_threshold_QLattice + margin)))]
                    pp_filtered = predicted_probabilities[((predicted_probabilities < (opt_threshold_QLattice - margin)) | (predicted_probabilities > (opt_threshold_QLattice + margin)))]
                    pc_filtered = np.where(pp_filtered >= opt_threshold_QLattice, True, False)

                CV_results_filtered_mdl = calculate_metrics(y_true = y_train_filtered_shap, y_pred = pc_filtered, y_pred_proba = pp_filtered)
                CV_results_filtered_mdl = pd.DataFrame(CV_results_filtered_mdl, index=[0])
                # Add the model name to the results
                CV_results_filtered_mdl['Model'] = selected_model
                CV_results_filtered_mdl['Margin'] = margin
                CV_results_filtered_mdl['SHAP_percentile'] = SHAP_percentile
                CV_results_filtered_mdl['Sample_size'] = len(y_train_filtered_shap)
                CV_results_filtered_mdl['Sample_loss'] = np.round(100-(len(y_train_filtered_shap)/len(y_fold_all))*100,2)
                # Append results
                aggregated_CV_results_filtered_mdl = aggregated_CV_results_filtered_mdl.append(CV_results_filtered_mdl, ignore_index=True)
                # Save the results to an Excel file
                
    # Save the aggregated results to an Excel file
    aggregated_CV_results_filtered_mdl.to_excel('agg_results.xlsx', index=False)
    # Define a filter for sample loss
    sample_loss_filter = aggregated_CV_results_filtered_mdl["Sample_loss"] < max_sample_loss_perc

    # Find the maximum values for ROCAUC, PRAUC, and MCC under the sample loss filter
    max_rocauc = aggregated_CV_results_filtered_mdl.loc[sample_loss_filter, "ROCAUC"].max()
    max_prauc = aggregated_CV_results_filtered_mdl.loc[sample_loss_filter, "PRAUC"].max()
    max_mcc = aggregated_CV_results_filtered_mdl.loc[sample_loss_filter, "MCC"].max()

    # Filter the DataFrame based on the maximum values and sample loss filter
    best_combination = aggregated_CV_results_filtered_mdl.loc[
        sample_loss_filter &
        (aggregated_CV_results_filtered_mdl["ROCAUC"] == max_rocauc) &
        (aggregated_CV_results_filtered_mdl["PRAUC"] == max_prauc) &
        (aggregated_CV_results_filtered_mdl["MCC"] == max_mcc)
    ]
    # Filter the DataFrame based on sample loss
    filtered_df = aggregated_CV_results_filtered_mdl.loc[sample_loss_filter]

    # Filter the DataFrame based on the maximum values and sample loss filter
    best_combination = filtered_df[
        (filtered_df["ROCAUC"] == max_rocauc) &
        (filtered_df["PRAUC"] == max_prauc) &
        (filtered_df["MCC"] == max_mcc)
    ]
    # Check if best_combination is empty
    if best_combination.empty:
        print("No exact match found. Finding the closest combination.")

        # Define a closeness metric: using inverse of distance to the maximum values
        filtered_df['Closeness'] = (
            np.abs(filtered_df["ROCAUC"] - max_rocauc) +
            np.abs(filtered_df["PRAUC"] - max_prauc) +
            np.abs(filtered_df["MCC"] - max_mcc)
        )

        # Find the row with the smallest closeness value
        closest_combination = filtered_df.loc[filtered_df['Closeness'].idxmin()]

        print("Closest combination found:")
        print(closest_combination)
        best_combination = closest_combination
    else:
        print("Best combination found:")
        print(best_combination)
        best_combination = best_combination.sort_values(by=["Sample_loss", "Margin", "SHAP_percentile"]).iloc[0]
    

    best_margin = best_combination["Margin"]
    best_SHAP_percentile = best_combination["SHAP_percentile"]
    best_model = best_combination["Model"]
    print(f"The best margin is {best_margin} and the best SHAP_percentile is {best_SHAP_percentile} and the best model is {best_model}")

# %%
if model_uncertainty_reduction:
    best_combination_agg_perf_df = aggregated_CV_results_filtered_mdl.loc[(aggregated_CV_results_filtered_mdl["Margin"] == best_margin) & (aggregated_CV_results_filtered_mdl["SHAP_percentile"] == best_SHAP_percentile)]
    # List of columns to be rounded
    columns_to_round = [
        "PPV", "NPV", "Sensitivity", "Specificity", 
        "Balanced Accuracy", "MCC", "ROCAUC", 
        "PRAUC", "Brier Score", "F1 Score"
    ]

    # Round the selected columns to 2 decimal places
    best_combination_agg_perf_df[columns_to_round] = best_combination_agg_perf_df[columns_to_round].round(2)

    print(best_combination_agg_perf_df)
    # Save the aggregated results to an Excel file
    best_combination_agg_perf_df.to_excel('best_combination_agg_perf_df.xlsx', index=False)
    
    # reassign the best model after applying the MUR approach
    selected_model = best_model
    
    # Get the full name of the best model
    best_model_name = model_names.get(best_model)

# %% [markdown]
# #### Stopping if there is no data split
# 
# If data split is not done then the following code should stop the pipeline here.

# %%
if not data_split:
    class IpyExit(SystemExit):
        """
        Exit Exception for IPython.
        This defines a custom exception class named `IpyExit`, which inherits from Python's built-in `SystemExit` class. This custom exception is used to handle the exit mechanism in IPython environments when the `data_split` variable is not set.
        """
        def __init__(self):
            """
            This defines the constructor (`__init__`) and destructor (`__del__`) methods for the `IpyExit` custom exception class.
            """
            sys.stderr = StringIO()

        def __del__(self):
            """
            The `__del__` method closes the captured output buffer and restores it to its original value, ensuring proper cleanup when exiting.
            """
            sys.stderr.close()
            sys.stderr = sys.__stderr__

    def ipy_exit():
        """
        This function raises an exception to terminate the execution of a Jupyter Notebook or IPython environment. When raised, it creates an instance of `SystemExit` which is handled by Python's built-in exit mechanism.

        """
        raise IpyExit

    if get_ipython():    # If running in IPython (e.g., Jupyter)
        exit = ipy_exit
    else:
        exit = sys.exit

    exit()  # Stop the execution


# %% [markdown]
# ### Prediction block for binary classification models
# 
# The following blocks are for the case that there is an independent dataset that can be used to validate a trained model (external validation). 

# %% [markdown]
# #### QLattice model

# %%
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if hp_tuning:
            
        best_composite_score = 0
        best_parameters = {'n_epochs': 50, 'max_complexity': 10}
    
        def evaluate_params(n_epochs, max_complexity):
            """
            Evaluate a composite model by tuning hyperparameters using the Feyn framework.

            Parameters:
                n_epochs (int): Number of epochs for training the model.
                max_complexity (int): Maximum complexity of the model.

            ## Returns
                tuple: The composite score (mean of AUC and AP) and the best parameters used to achieve this score.

            Notes:
                This function uses the Feyn framework to perform a hyperparameter tuning search.
                It assumes that `mydata_imputed_nocv`, `outcome_var`, `y_train`, `stypes`, `sample_weights` are defined elsewhere in the code.
                The `random_seed` parameter is set to `SEED` for reproducibility.

            Steps:
                1. Initialize a QLattice object with a random seed.
                2. Use `auto_run` to perform a hyperparameter tuning search.
                    - The model is run on the training data (`mydata_imputed_nocv`) for `n_epochs` epochs.
                    - The criterion used for hyperparameter selection is AIC (Akaike information criterion).
                    - The loss function used for classification is binary cross-entropy.
                3. Select the best model from the set of models generated by `auto_run`.
                4. Use this best model to make predictions on the training data.
                5. Calculate the composite score, which is the mean of AUC (Receiver Operating Characteristic) and AP (Area Under the Precision-Recall Curve).
            """
            ql = feyn.QLattice(random_seed=SEED)
            models = ql.auto_run(
                data=mydata_imputed_nocv,
                output_name=outcome_var,
                kind='classification',
                n_epochs=n_epochs,
                stypes=stypes,
                criterion="aic",
                loss_function='binary_cross_entropy',
                max_complexity=max_complexity,
                sample_weights=sample_weights
            )
            best_model = models[0]
            
            predictions_proba = best_model.predict(mydata_imputed_nocv)
            QL_composite_score = (roc_auc_score(y_true = y_train, y_score = predictions_proba) + average_precision_score(y_true = y_train, y_score = predictions_proba))/2
            print(QL_composite_score)
            return QL_composite_score, {'n_epochs': n_epochs, 'max_complexity': max_complexity}
    
        results = Parallel(n_jobs=n_cpu_for_tuning, backend='loky')(
            delayed(evaluate_params)(n_epochs, max_complexity)
            for n_epochs in [50, 100]
            for max_complexity in [5, 10]
        )
    
        for QL_composite_score, params in results:
            if QL_composite_score > best_composite_score:
                best_composite_score = QL_composite_score
                best_parameters = params
        
        print("Best Parameters:", best_parameters)
        print("Best composite score:", best_composite_score)
        # Use the best parameters from the grid search
        best_n_epochs = best_parameters['n_epochs']
        best_max_complexity = best_parameters['max_complexity']
    else:
        best_n_epochs = 50
        best_max_complexity = 10
            
    # Train the final model with the best parameters
    ql = feyn.QLattice(random_seed=SEED)
    models = ql.auto_run(
        data=mydata_imputed_nocv,
        output_name=outcome_var,
        kind='classification',
        n_epochs=best_n_epochs,
        stypes=stypes,
        criterion="aic",
        loss_function='binary_cross_entropy',
        max_complexity=best_max_complexity,
        sample_weights=sample_weights
    )

    best_model = models[0]
            


# %%
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if "QLattice_mdl" in models_to_include:
        best_model.plot_signal(mydata_imputed_nocv,corr_func='spearman')

# %%
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if "QLattice_mdl" in models_to_include:
        best_model.plot_signal(testset_imputed,corr_func='spearman')

# %%
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if "QLattice_mdl" in models_to_include:
        best_model.plot_signal(mydata_imputed_nocv,corr_func='mutual_information')

# %%
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if "QLattice_mdl" in models_to_include:
        best_model.plot_signal(testset_imputed,corr_func='mutual_information')

# %%
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if "QLattice_mdl" in models_to_include:
        results_df_QLattice, missclassified_samples = evaluate_and_plot_model(model = best_model,
                                                    threshold = opt_threshold_QLattice,
                                                    testset = testset_imputed,
                                                    y_test = y_test,
                                                    filename= f'ROC_CM_QLattice.{fig_file_format}')
        if external_val:
            # Reorder extval_data_imputed columns to match testset_imputed
            extval_data_imputed = extval_data_imputed[testset_imputed.columns]
            results_df_QLattice_extval, missclassified_samples_external = evaluate_and_plot_model(model = best_model,
                                                        threshold = opt_threshold_QLattice,
                                                        testset = extval_data_imputed,
                                                        y_test = y_extval_data,
                                                        filename= f'ROC_CM_QLattice_extval.{fig_file_format}')
    
        if export_missclassified: # extend the code if you have external validation set and want to check this 
            misclassified_ids = mydata_backup.loc[missclassified_samples, 'ID']
            
            misclassified_ids_df.to_excel('testset_misclassified_ids_QLattice.xlsx', index=False)
        

# %%
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if "QLattice_mdl" in models_to_include:
        init_printing()
        display(best_model.plot(mydata_imputed_nocv, testset_imputed))

# %%
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if "QLattice_mdl" in models_to_include:
        # feature selected by the model
        print(best_model.features)

# %%
# distribution of model predicted probabilities for each class
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if "QLattice_mdl" in models_to_include:
        best_model.plot_probability_scores(testset_imputed)


# %%
# model representation as a closed-form expression
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if "QLattice_mdl" in models_to_include:
        init_printing()
        sympy_model = best_model.sympify(symbolic_lr=True, include_weights=True)

        display(sympy_model.as_expr())

# %%
if test_only_best_cvmodel and best_model_name != "QLattice":
    pass
else:
    if "QLattice_mdl" in models_to_include:
        # Save a model to a file
        best_model.save('QLattice_model.json')
        
        # to load the model use the following script
        # from feyn import Model
        # model = Model.load('QLattice_model.json')
        # prediction = model.predict(testset_imputed)


# %% [markdown]
# #### Test dummy models
# 
# See how dummy models (models that are not trained on the data) perform. This is done to estimate the performance level of dummy models as compared with the models that are trained.

# %%

# Train Dummy Classifier
dummy_classifier = DummyClassifier(strategy='most_frequent')  # you can choose different strategies based on your requirements
dummy_classifier.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)

results_df_dummy, missclassified_samples = evaluate_and_plot_model(model = dummy_classifier,
                                        threshold = 0.5,
                                        testset = X_test_OHE,
                                        y_test = y_test,
                                        filename= f'ROC_CM_dummy_most_frequent.{fig_file_format}')

# %%
# Train Dummy Classifier with 'stratified' strategy
dummy_classifier = DummyClassifier(strategy='stratified')
dummy_classifier.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)

results_df_dummy, missclassified_samples = evaluate_and_plot_model(model = dummy_classifier,
                                        threshold = 0.5,
                                        testset = X_test_OHE,
                                        y_test = y_test,
                                        filename= f'ROC_CM_dummy_stratified.{fig_file_format}')

# %% [markdown]
# #### Gaussian Naive Bayes

# %%
if test_only_best_cvmodel and best_model_name != "NB":
    pass
else:
    if "NaiveBayes_mdl" in models_to_include:
        # Train Naive Bayes
        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)

        results_df_NB, missclassified_samples = evaluate_and_plot_model(model = nb_classifier,
                                                threshold = opt_threshold_NB,
                                                testset = X_test_OHE,
                                                y_test = y_test,
                                                filename= f'ROC_CM_NB.{fig_file_format}')
        
        if external_val:
            # Reorder X_extval_data_OHE columns to match X_test_OHE
            X_extval_data_OHE = X_extval_data_OHE[X_test_OHE.columns]
            results_df_NB_extval, missclassified_samples_external = evaluate_and_plot_model(model = nb_classifier,
                                                    threshold = opt_threshold_NB,
                                                    testset = X_extval_data_OHE,
                                                    y_test = y_extval_data,
                                                    filename= f'ROC_CM_NB_extval.{fig_file_format}')
    
        if export_missclassified: # extend the code if you have external validation set and want to check this 
            misclassified_ids = mydata_backup.loc[missclassified_samples, 'ID']
            
            misclassified_ids_df.to_excel('testset_misclassified_ids_NB.xlsx', index=False)

# %% [markdown]
# #### Logistic Regression

# %%
if test_only_best_cvmodel and best_model_name != "LR":
    pass
else:
    if "LogisticRegression_mdl" in models_to_include:
        # Calculate necessary information for adjusting hyperparameters
        n_rows = X_train_OHE_nocv.shape[0]
        n_cols = X_train_OHE_nocv.shape[1]
        class_proportion = y_train.mean()  # binary classification
        rf_params, lgbm_params, hgbc_params, cb_params, lr_params = set_parameters(n_rows, n_cols, class_proportion)
        # Create a Logistic Regression instance
        lr = LogisticRegression(penalty='l1',random_state=SEED, solver="liblinear", **lr_params)
        if hp_tuning:
        
            # Adjust hyperparameters based on the training data in this fold
            rf_param_dist, lgbm_param_dist, hgbc_param_dist, cb_param_dist, lr_param_dist = adjust_hyperparameters(n_rows, n_cols)
            # a RandomizedSearchCV instance
            random_search = RandomizedSearchCV(
                estimator=lr,
                param_distributions=lr_param_dist,
                n_iter=n_iter_hptuning,
                scoring= custom_scorer, 
                cv=cv_folds_hptuning,
                refit=True, 
                random_state=SEED,
                verbose=0,
                n_jobs=n_cpu_for_tuning
            )

            # Perform the random search on the training data
            random_search.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)

            # Get the best parameters and best estimator
            best_params = random_search.best_params_
            lr = LogisticRegression(penalty='l1',random_state=SEED, solver="liblinear", **best_params)
        else:
            lr = LogisticRegression(penalty='l1',random_state=SEED, solver="liblinear", **lr_params)

        # Fit the best estimator on the entire training data
        lr.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)


# %%
if test_only_best_cvmodel and best_model_name != "LR":
    pass
else:
    if "LogisticRegression_mdl" in models_to_include:
        results_df_LR, missclassified_samples = evaluate_and_plot_model(model = lr,
                                                threshold = opt_threshold_LR,
                                                testset = X_test_OHE,
                                                y_test = y_test,
                                                filename= f'ROC_CM_LR.{fig_file_format}')
        if external_val:
            # Reorder X_extval_data_OHE columns to match X_test_OHE
            X_extval_data_OHE = X_extval_data_OHE[X_test_OHE.columns]
            results_df_LR_extval, missclassified_samples_external = evaluate_and_plot_model(model = lr,
                                                threshold = opt_threshold_LR,
                                                testset = X_extval_data_OHE,
                                                y_test = y_extval_data,
                                                filename= f'ROC_CM_LR_extval.{fig_file_format}')
            
        if export_missclassified: # extend the code if you have external validation set and want to check this 
            misclassified_ids = mydata_backup.loc[missclassified_samples, 'ID']
            
            misclassified_ids_df.to_excel('testset_misclassified_ids_LR.xlsx', index=False)


# %% [markdown]
# #### HistGBC

# %%
if test_only_best_cvmodel and best_model_name != "HGBC":
    pass
else:
    if "HistGBC_mdl" in models_to_include:
        # Calculate necessary information for adjusting hyperparameters
        n_rows = X_train_OHE_nocv.shape[0]
        n_cols = X_train_OHE_nocv.shape[1]
        class_proportion = y_train.mean()  # binary classification
        rf_params, lgbm_params, hgbc_params, cb_params, lr_params = set_parameters(n_rows, n_cols, class_proportion)
        # a HistGradientBoostingClassifier instance
        HGBC = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True, **hgbc_params)
        if hp_tuning:
        
            # Adjust hyperparameters based on the training data in this fold
            rf_param_dist, lgbm_param_dist, hgbc_param_dist, cb_param_dist, lr_param_dist = adjust_hyperparameters(n_rows, n_cols)
            # a RandomizedSearchCV instance
            random_search = RandomizedSearchCV(
                estimator=HGBC, 
                param_distributions=hgbc_param_dist, 
                n_iter=n_iter_hptuning,
                scoring= custom_scorer, 
                cv=cv_folds_hptuning,
                refit=True, 
                random_state=SEED,
                verbose=0,
                n_jobs = n_cpu_for_tuning)

            # Perform the random search on the training data
            random_search.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)

            # Get the best parameters and best estimator
            best_params = random_search.best_params_
            HGBC = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True, **best_params)
        else:
            HGBC = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True, **hgbc_params)

        # Fit the best estimator on the entire training data
        HGBC.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)


# %%
if test_only_best_cvmodel and best_model_name != "HGBC":
    pass
else:
    if "HistGBC_mdl" in models_to_include:
        results_df_HGBC, missclassified_samples = evaluate_and_plot_model(model = HGBC,
                                                threshold = opt_threshold_HGBC,
                                                testset = X_test_OHE,
                                                y_test = y_test,
                                                filename= f'ROC_CM_HGBC.{fig_file_format}')
        
        if external_val:
            # Reorder X_extval_data_OHE columns to match X_test_OHE
            X_extval_data_OHE = X_extval_data_OHE[X_test_OHE.columns]
            results_df_HGBC_extval, missclassified_samples_external = evaluate_and_plot_model(model = HGBC,
                                                threshold = opt_threshold_HGBC,
                                                testset = X_extval_data_OHE,
                                                y_test = y_extval_data,
                                                filename= f'ROC_CM_HGBC_extval.{fig_file_format}')
    
        if export_missclassified: # extend the code if you have external validation set and want to check this 
            misclassified_ids = mydata_backup.loc[missclassified_samples, 'ID']
            
            misclassified_ids_df.to_excel('testset_misclassified_ids_HGBC.xlsx', index=False)

# %% [markdown]
# #### Random Forest (RF)

# %%
if test_only_best_cvmodel and best_model_name != "rf":
    pass
else:
    if "RandomForest_mdl" in models_to_include:
        # Calculate necessary information for adjusting hyperparameters
        n_rows = X_train_OHE_nocv.shape[0]
        n_cols = X_train_OHE_nocv.shape[1]
        class_proportion = y_train.mean()  # binary classification
        rf_params, lgbm_params, hgbc_params, cb_params, lr_params = set_parameters(n_rows, n_cols, class_proportion)
        rf = RandomForestClassifier(random_state=SEED, n_jobs=n_cpu_model_training, **rf_params) # , class_weight= "balanced"
        # rf = RandomForestClassifier(random_state=SEED, sampling_strategy='all', n_jobs=n_cpu_model_training, **rf_params)
        if hp_tuning:      
            # Adjust hyperparameters based on the training data in this fold
            rf_param_dist, lgbm_param_dist, hgbc_param_dist, cb_param_dist, lr_param_dist = adjust_hyperparameters(n_rows, n_cols)
            # Create RandomizedSearchCV object with balanced accuracy as the scoring metric
            random_search = RandomizedSearchCV(
                estimator=rf, 
                param_distributions=rf_param_dist, 
                n_iter=n_iter_hptuning,
                scoring= custom_scorer, 
                cv=cv_folds_hptuning,
                refit=True, 
                random_state=SEED,
                verbose=0,n_jobs = n_cpu_for_tuning)

            # Fit the RandomizedSearchCV object to the data
            random_search.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)

            # Get the best parameters and best estimator from the random search
            best_params = random_search.best_params_

            # Reinitialize a new rf model with the best parameters
            rf = RandomForestClassifier(random_state=SEED, class_weight= "balanced",n_jobs=n_cpu_model_training, **best_params)
            # Print the best parameters
            print("Best Parameters:", best_params)
        else:
            rf = RandomForestClassifier(random_state=SEED, class_weight= "balanced",n_jobs=n_cpu_model_training, **rf_params)
            # Print the best parameters
            print("Best Parameters:", best_params)
        # Train the new model on the training data
        rf.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)




# %%
if test_only_best_cvmodel and best_model_name != "rf":
    pass
else:
    if "RandomForest_mdl" in models_to_include:
        results_df_rf, missclassified_samples = evaluate_and_plot_model(model = rf, threshold = opt_threshold_rf, testset = X_test_OHE, y_test = y_test, filename= f'ROC_CM_rf.{fig_file_format}')
        if external_val:
            # Reorder X_extval_data_OHE columns to match X_test_OHE
            X_extval_data_OHE = X_extval_data_OHE[X_test_OHE.columns]
            results_df_rf_extval, missclassified_samples_external = evaluate_and_plot_model(model = rf,
                                                            threshold = opt_threshold_rf,
                                                            testset = X_extval_data_OHE,
                                                            y_test = y_extval_data,
                                                            filename= f'ROC_CM_rf_extval.{fig_file_format}')

        if export_missclassified: # extend the code if you have external validation set and want to check this 
            misclassified_ids = mydata_backup.loc[missclassified_samples, 'ID']
            
            misclassified_ids_df.to_excel('testset_misclassified_ids_rf.xlsx', index=False)

# %% [markdown]
# #### LightGBM

# %%
if test_only_best_cvmodel and best_model_name != "LGBM":
    pass
else:
    if "LightGBM_mdl" in models_to_include:
        # Calculate necessary information for adjusting hyperparameters
        n_rows = X_train.shape[0]
        n_cols = X_train.shape[1]
        class_proportion = y_train.mean()  # binary classification
        rf_params, lgbm_params, hgbc_params, cb_params, lr_params = set_parameters(n_rows, n_cols, class_proportion)
        print(lgbm_params)
        # Define the classifier
        if GPU_avail:
            lgbm = lgb.LGBMClassifier(random_state=SEED, n_jobs=n_cpu_model_training, verbose=-1, device="gpu", **lgbm_params) 
        else:
            lgbm = lgb.LGBMClassifier(random_state=SEED, n_jobs=n_cpu_model_training, verbose=-1, **lgbm_params) 

        if hp_tuning:
            
            # Adjust hyperparameters based on the training data in this fold
            rf_param_dist, lgbm_param_dist, hgbc_param_dist, cb_param_dist, lr_param_dist = adjust_hyperparameters(n_rows, n_cols)
            
            # Define the search strategy and scoring metric
            # a RandomizedSearchCV instance
            random_search = RandomizedSearchCV(
                estimator=lgbm, 
                param_distributions=lgbm_param_dist, 
                n_iter=n_iter_hptuning,
                scoring= custom_scorer, 
                cv=cv_folds_hptuning,
                refit=True, 
                random_state=SEED,
                verbose=0,
                n_jobs = n_cpu_for_tuning)

            # Perform the random search on the data
            random_search.fit(X_train, y_train, sample_weight=sample_weights)

            # Get the best parameters and best model
            best_params = random_search.best_params_

            # Print the best parameters
            print("Best Hyperparameters:", best_params)

            # Reinitialize a new lgbm model with the best parameters
            if GPU_avail:
                lgbm = lgb.LGBMClassifier(random_state=SEED, n_jobs=n_cpu_model_training, verbose=-1, device="gpu", **best_params) 
            else:
                lgbm = lgb.LGBMClassifier(random_state=SEED, n_jobs=n_cpu_model_training, verbose=-1, **best_params) 

        # Train the new model on the training data
        lgbm.fit(X_train, y_train, sample_weight=sample_weights)


# %%
if test_only_best_cvmodel and best_model_name != "LGBM":
    pass
else:
    if "LightGBM_mdl" in models_to_include:
        results_df_LGBM, missclassified_samples = evaluate_and_plot_model(model = lgbm,
                                                threshold = opt_threshold_LGBM,
                                                testset = X_test,
                                                y_test = y_test,
                                                filename= f'ROC_CM_LGBM.{fig_file_format}')
        if external_val:
            # Reorder X_extval_data columns to match X_test
            X_extval_data = X_extval_data[X_test.columns]
            results_df_LGBM_extval, missclassified_samples_external = evaluate_and_plot_model(model = lgbm,
                                            threshold = opt_threshold_LGBM,
                                            testset = X_extval_data,
                                            y_test = y_extval_data,
                                            filename= f'ROC_CM_LGBM_extval.{fig_file_format}')
        
        if export_missclassified: # extend the code if you have external validation set and want to check this 
            misclassified_ids = mydata_backup.loc[missclassified_samples, 'ID']
            
            misclassified_ids_df.to_excel('testset_misclassified_ids_LGBM.xlsx', index=False)

# %% [markdown]
# #### CatBoost
# 
# It may generate some unimportant warning messages about that can be ignored and cleaned up after running the pipeline.

# %%
if test_only_best_cvmodel and best_model_name != "CB":
    pass
else:
    if "CatBoost_mdl" in models_to_include:
        # Calculate necessary information for adjusting hyperparameters
        n_rows = X_train.shape[0]
        n_cols = X_train.shape[1]
        class_proportion = y_train.mean()  # binary classification
        rf_params, lgbm_params, hgbc_params, cb_params, lr_params = set_parameters(n_rows, n_cols, class_proportion)
        if hp_tuning:
            # Adjust hyperparameters based on the training data in this fold
            rf_param_dist, lgbm_param_dist, hgbc_param_dist, cb_param_dist, lr_param_dist = adjust_hyperparameters(n_rows, n_cols)
            catb = cb.CatBoostClassifier(random_state=SEED, cat_features=cat_features, silent=True) # , logging_level='Silent' verbose=0, 

            # Perform random search
            random_search = RandomizedSearchCV(
                estimator=catb, 
                param_distributions=cb_param_dist, 
                n_iter=n_iter_hptuning,
                scoring= custom_scorer, 
                cv=cv_folds_hptuning,
                refit=True, 
                random_state=SEED,
                verbose=0,
                n_jobs=n_cpu_for_tuning
            )

            # Fit the random search on your data
            random_search.fit(X_train, y_train, sample_weight=sample_weights)

            # Get the best parameters and best model
            best_params = random_search.best_params_
            
            catb = cb.CatBoostClassifier(random_state=SEED, cat_features=cat_features, silent=True, **best_params) 
        else:
            catb = cb.CatBoostClassifier(random_state=SEED, cat_features=cat_features, silent=True, **cb_params) 
            
        # Train the new model on the training data
        catb.fit(X_train, y_train, sample_weight=sample_weights)


# %%
if test_only_best_cvmodel and best_model_name != "CB":
    pass
else:
    if "CatBoost_mdl" in models_to_include:
        results_df_CB, missclassified_samples = evaluate_and_plot_model(model = catb,
                                                threshold = opt_threshold_CB,
                                                testset = X_test,
                                                y_test = y_test,
                                                filename= f'ROC_CM_CB.{fig_file_format}')
        
        if external_val:
            # Reorder X_extval_data columns to match X_test
            X_extval_data = X_extval_data[X_test.columns]
            results_df_CB_extval, missclassified_samples_external = evaluate_and_plot_model(model = catb,
                                            threshold = opt_threshold_CB,
                                            testset = X_extval_data,
                                            y_test = y_extval_data,
                                            filename= f'ROC_CM_CB_extval.{fig_file_format}')
        
        if export_missclassified: # extend the code if you have external validation set and want to check this 
            misclassified_ids = mydata_backup.loc[missclassified_samples, 'ID']
            
            misclassified_ids_df.to_excel('testset_misclassified_ids_CB.xlsx', index=False)

# %% [markdown]
# #### Model interpretation for the best performing model
# 
# The best performing model is chosen based on the performance of the models on cross validation as the model with the highest mean of MCC, AUC, and PRAUC. This model may not necessarily have the best performance on the test set, especially if the models perform closely similar on the cross validation. Since most of the data is used in cross validation, the model that is chosen based on that is prefered to the best performing model based only on the test set.

# %%
if best_model_name != "QLattice":
    if test_only_best_cvmodel:
        if best_model_name == "rf":
            selected_model = rf
        elif best_model_name == "LGBM":
            selected_model = lgbm
        elif best_model_name == "NB":
            selected_model = nb_classifier
        elif best_model_name == "CB":
            selected_model = catb
        elif best_model_name == "LR":
            selected_model = lr
        elif best_model_name == "HGBC":
            selected_model = HGBC
    else:
        selected_model = model_dictionary[best_model_name]
else:
    skip_block = True
    
    # raise Exception("QLattice is already explained - Stopping code execution")


# %% [markdown]
# ##### SHAP values association with predicted probabilities

# %%
if not skip_block:
    # Calculate SHAP values for the positive class
    positive_class_index = 1 

    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier)):
        explainer = shap.TreeExplainer(selected_model)
        shap_values = explainer.shap_values(X_test_OHE)
        if isinstance(selected_model, RandomForestClassifier):
            # shap_values = shap_values[positive_class_index]
            shap_values = shap_values[:,:,1]
    elif isinstance(selected_model, cb.CatBoostClassifier):
        explainer = shap.TreeExplainer(selected_model)
        shap_values = explainer.shap_values(X_test)
    elif isinstance(selected_model, LogisticRegression):
        explainer = shap.LinearExplainer(selected_model, X_train_OHE_nocv)
        shap_values = explainer.shap_values(X_test_OHE)
    elif isinstance(selected_model, GaussianNB):  
        explainer = shap.Explainer(selected_model.predict_proba, X_train_OHE_nocv)
        shap_values = explainer(X_test_OHE)
        shap_values = shap_values.values
        shap_values = shap_values[:, :, 1]
    else:
        explainer = shap.TreeExplainer(selected_model)
        shap_values = explainer.shap_values(X_test)
        # shap_values = shap_values[positive_class_index]

    # Calculate the sum of SHAP values for each sample
    shap_sum = shap_values.sum(axis=1)

    # Get the predicted probabilities of the model
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier)):
        predicted_probabilities = selected_model.predict_proba(X_test_OHE)[:, positive_class_index]
    elif isinstance(selected_model, cb.CatBoostClassifier):
        predicted_probabilities = selected_model.predict_proba(X_test)[:, positive_class_index]
    elif isinstance(selected_model, (LogisticRegression, GaussianNB)):
        predicted_probabilities = selected_model.predict_proba(X_test_OHE)[:, positive_class_index]
    else:
        predicted_probabilities = selected_model.predict_proba(X_test)[:, positive_class_index]

    # Plot the SHAP sum against the predicted probabilities
    plt.scatter(shap_sum, predicted_probabilities)

    plt.xlabel('sum of SHAP values')
    plt.ylabel('predicted probability')
    plt.title('sum of SHAP values vs. predicted probability', size=10)
    plt.grid(linestyle=':', linewidth=0.5, alpha=0.5)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 8
    plt.grid(True)
    plt.gca().tick_params(axis='both', labelsize=8) 
    plt.gca().set_facecolor('white')
    # display grid lines
    plt.grid(which='both', color="grey")
    plt.grid(which='minor', alpha=0.1)
    plt.grid(which='major', alpha=0.2)
    plt.show()


# %%
if not skip_block:
    if model_uncertainty_reduction:

        # Calculate SHAP values for the positive class
        positive_class_index = 1 
        if isinstance(selected_model, HistGradientBoostingClassifier):
            explainer = shap.TreeExplainer(selected_model)
            shap_values = explainer.shap_values(X_test_OHE)
            predicted_probabilities = selected_model.predict_proba(X_test_OHE)[:, positive_class_index]
            shap_sum = shap_values.sum(axis=1)
            shap_sum_abs = np.abs(shap_sum)
            SHAP_thr_HGBC = np.percentile(shap_sum_abs, best_SHAP_percentile)
            opt_threshold_selectedmodel = opt_threshold_HGBC

            X_test_filtered_shap = X_test_OHE[(shap_sum_abs > SHAP_thr_HGBC) & 
                                                ((predicted_probabilities < (opt_threshold_HGBC - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_HGBC + best_margin)))]
            y_test_filtered_shap = y_test[(shap_sum_abs > SHAP_thr_HGBC) & 
                                            ((predicted_probabilities < (opt_threshold_HGBC - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_HGBC + best_margin)))]
            pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_HGBC) & 
                                            ((predicted_probabilities < (opt_threshold_HGBC - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_HGBC + best_margin)))]
            pc_filtered = np.where(pp_filtered >= opt_threshold_HGBC, True, False)
            
        elif isinstance(selected_model, RandomForestClassifier):
            explainer = shap.TreeExplainer(selected_model)
            shap_values = explainer.shap_values(X_test_OHE)
            shap_values = shap_values[:,:,1]
            predicted_probabilities = selected_model.predict_proba(X_test_OHE)[:, positive_class_index]
            shap_sum = shap_values.sum(axis=1)
            shap_sum_abs = np.abs(shap_sum)
            SHAP_thr_rf = np.percentile(shap_sum_abs, best_SHAP_percentile)
            opt_threshold_selectedmodel = opt_threshold_rf
            X_test_filtered_shap = X_test_OHE[(shap_sum_abs > SHAP_thr_rf) & 
                                                ((predicted_probabilities < (opt_threshold_rf - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_rf + best_margin)))]
            y_test_filtered_shap = y_test[(shap_sum_abs > SHAP_thr_rf) & 
                                            ((predicted_probabilities < (opt_threshold_rf - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_rf + best_margin)))]
            pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_rf) & 
                                            ((predicted_probabilities < (opt_threshold_rf - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_rf + best_margin)))]
            pc_filtered = np.where(pp_filtered >= opt_threshold_rf, True, False)
            
        elif isinstance(selected_model, cb.CatBoostClassifier):
            explainer = shap.TreeExplainer(selected_model)
            shap_values = explainer.shap_values(X_test)
            shap_sum = shap_values.sum(axis=1)
            shap_sum_abs = np.abs(shap_sum)
            SHAP_thr_CB = np.percentile(shap_sum_abs, best_SHAP_percentile)
            opt_threshold_selectedmodel = opt_threshold_CB
            predicted_probabilities = selected_model.predict_proba(X_test)[:, positive_class_index]
            X_test_filtered_shap = X_test[(shap_sum_abs > SHAP_thr_CB) & 
                                            ((predicted_probabilities < (opt_threshold_CB - best_margin)) | 
                                            (predicted_probabilities > (opt_threshold_CB + best_margin)))]
            y_test_filtered_shap = y_test[(shap_sum_abs > SHAP_thr_CB) & 
                                            ((predicted_probabilities < (opt_threshold_CB - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_CB + best_margin)))]
            pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_CB) & 
                                            ((predicted_probabilities < (opt_threshold_CB - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_CB + best_margin)))]
            pc_filtered = np.where(pp_filtered >= opt_threshold_CB, True, False)
            
        elif isinstance(selected_model, LogisticRegression):
            opt_threshold_selectedmodel = opt_threshold_LR
            explainer = shap.LinearExplainer(selected_model, X_train_OHE_nocv)
            shap_values = explainer.shap_values(X_test_OHE)
            predicted_probabilities = selected_model.predict_proba(X_test_OHE)[:, positive_class_index]
            shap_sum = shap_values.sum(axis=1)
            shap_sum_abs = np.abs(shap_sum)
            SHAP_thr_LR = np.percentile(shap_sum_abs, best_SHAP_percentile)

            X_test_filtered_shap = X_test_OHE[(shap_sum_abs > SHAP_thr_LR) & 
                                                ((predicted_probabilities < (opt_threshold_LR - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_LR + best_margin)))]
            y_test_filtered_shap = y_test[(shap_sum_abs > SHAP_thr_LR) & 
                                            ((predicted_probabilities < (opt_threshold_LR - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_LR + best_margin)))]
            pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_LR) & 
                                            ((predicted_probabilities < (opt_threshold_LR - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_LR + best_margin)))]
            pc_filtered = np.where(pp_filtered >= opt_threshold_LR, True, False)
            
        elif isinstance(selected_model, GaussianNB):  
            explainer = shap.Explainer(selected_model.predict_proba, X_train_OHE_nocv)
            opt_threshold_selectedmodel = opt_threshold_NB
            shap_values = explainer(X_test_OHE)  
            shap_values = shap_values.values[:,:,1]
            predicted_probabilities = selected_model.predict_proba(X_test_OHE)[:, positive_class_index]
            shap_sum = shap_values.sum(axis=1)
            shap_sum_abs = np.abs(shap_sum)
            SHAP_thr_NB = np.percentile(shap_sum_abs, best_SHAP_percentile)

            X_test_filtered_shap = X_test_OHE[(shap_sum_abs > SHAP_thr_NB) & 
                                                ((predicted_probabilities < (opt_threshold_NB - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_NB + best_margin)))]
            y_test_filtered_shap = y_test[(shap_sum_abs > SHAP_thr_NB) & 
                                            ((predicted_probabilities < (opt_threshold_NB - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_NB + best_margin)))]
            pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_NB) & 
                                            ((predicted_probabilities < (opt_threshold_NB - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_NB + best_margin)))]
            pc_filtered = np.where(pp_filtered >= opt_threshold_NB, True, False)
            
        else: # LGBM
            opt_threshold_selectedmodel = opt_threshold_LGBM
            explainer = shap.TreeExplainer(selected_model)
            shap_values = explainer.shap_values(X_test)
            # shap_values = shap_values[positive_class_index]
            shap_sum = shap_values.sum(axis=1)
            shap_sum_abs = np.abs(shap_sum)
            SHAP_thr_LGBM = np.percentile(shap_sum_abs, best_SHAP_percentile)
            predicted_probabilities = selected_model.predict_proba(X_test)[:, positive_class_index]
            X_test_filtered_shap = X_test[(shap_sum_abs > SHAP_thr_LGBM) & 
                                                ((predicted_probabilities < (opt_threshold_LGBM - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_LGBM + best_margin)))]
            y_test_filtered_shap = y_test[(shap_sum_abs > SHAP_thr_LGBM) & 
                                            ((predicted_probabilities < (opt_threshold_LGBM - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_LGBM + best_margin)))]
            pp_filtered = predicted_probabilities[(shap_sum_abs > SHAP_thr_LGBM) & 
                                            ((predicted_probabilities < (opt_threshold_LGBM - best_margin)) | 
                                                (predicted_probabilities > (opt_threshold_LGBM + best_margin)))]
            pc_filtered = np.where(pp_filtered >= opt_threshold_LGBM, True, False)

        # Plot the SHAP sum against the predicted probabilities
        plt.scatter(shap_sum_abs, predicted_probabilities)
        print(np.round(len(X_test_filtered_shap)/len(X_test),2))
        plt.xlabel('Sum of absolute SHAP values')
        plt.ylabel('Predicted probability')
        plt.title('Sum of absolute SHAP values vs. Predicted probability', size=10)
        plt.grid(linestyle=':', linewidth=0.5, alpha=0.5)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 8
        plt.grid(True)
        plt.gca().tick_params(axis='both', labelsize=8) 
        plt.gca().set_facecolor('white')
        # display grid lines
        plt.grid(which='both', color="grey")
        plt.grid(which='minor', alpha=0.1)
        plt.grid(which='major', alpha=0.2)
        plt.show()


# %%
if not skip_block:
    if model_uncertainty_reduction:
        results_df_selected_model, missclassified_samples_selected_model = evaluate_and_plot_model(model = selected_model,
                                                        threshold = opt_threshold_selectedmodel,
                                                        testset = X_test_filtered_shap,
                                                        y_test = y_test_filtered_shap,
                                                        filename= f'ROC_CM_selected_model.{fig_file_format}')
        
        if isinstance(selected_model, (LogisticRegression, GaussianNB, HistGradientBoostingClassifier, RandomForestClassifier)): # then we discard samples that will have uncertain predictions
            X_test_OHE = X_test_filtered_shap
        else:
            X_test = X_test_filtered_shap
            
        y_test = y_test_filtered_shap

# %%
if not skip_block:
    # Calculate SHAP values for the positive class
    positive_class_index = 1 

    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier)):
        explainer = shap.TreeExplainer(selected_model)
        shap_values = explainer.shap_values(X_test_OHE)
        if isinstance(selected_model, RandomForestClassifier):
            shap_values = shap_values[:,:,1]
    elif isinstance(selected_model, cb.CatBoostClassifier):
        explainer = shap.TreeExplainer(selected_model)
        shap_values = explainer.shap_values(X_test)
    elif isinstance(selected_model, LogisticRegression):
        explainer = shap.LinearExplainer(selected_model, X_train_OHE_nocv)
        shap_values = explainer.shap_values(X_test_OHE)
    elif isinstance(selected_model, GaussianNB):  
        explainer = shap.Explainer(selected_model.predict_proba, X_train_OHE_nocv)
        shap_values = explainer(X_test_OHE)  
        shap_values = shap_values.values[:,:,1]
    else:
        explainer = shap.TreeExplainer(selected_model)
        shap_values = explainer.shap_values(X_test)
        # shap_values = shap_values[positive_class_index]

    # Calculate the sum of SHAP values for each sample
    shap_sum = shap_values.sum(axis=1)

    # Get the predicted probabilities of the model
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier)):
        predicted_probabilities = selected_model.predict_proba(X_test_OHE)[:, positive_class_index]
    elif isinstance(selected_model, cb.CatBoostClassifier):
        predicted_probabilities = selected_model.predict_proba(X_test)[:, positive_class_index]
    elif isinstance(selected_model, (LogisticRegression, GaussianNB)):
        predicted_probabilities = selected_model.predict_proba(X_test_OHE)[:, positive_class_index]
    else:
        predicted_probabilities = selected_model.predict_proba(X_test)[:, positive_class_index]

    # Plot the SHAP sum against the predicted probabilities
    plt.scatter(shap_sum, predicted_probabilities)

    plt.xlabel('sum of SHAP values')
    plt.ylabel('predicted probability')
    plt.title('sum of SHAP values vs. predicted probability', size=10)
    plt.grid(linestyle=':', linewidth=0.5, alpha=0.5)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 8
    plt.grid(True)
    plt.gca().tick_params(axis='both', labelsize=8) 
    plt.gca().set_facecolor('white')
    # display grid lines
    plt.grid(which='both', color="grey")
    plt.grid(which='minor', alpha=0.1)
    plt.grid(which='major', alpha=0.2)
    plt.show()


# %% [markdown]
# ##### Interpret the model based on SHAP analysis

# %% [markdown]
# ##### SHAP summary plot
# 
# Note: the plot cannot show categorical features in color codes and thus they are plotted in grey (not mistaken with missing values)
# In the case of having categorical features, two SHAP plots are displayed, once with categories shown and once using the original SHAP plot that does not show the categories

# %%
if not skip_block:
    # Determine which features to use based on the model type
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
        feature_names_with_shapvalues = [
            data_dictionary.get(feature, feature) + ": " + str(round(value, 2))
            for feature, value in zip(X_test_OHE.columns, np.mean(np.abs(shap_values), axis=0)) 
        ]
    else:
        feature_names_with_shapvalues = [
            data_dictionary.get(feature, feature) + ": " + str(round(value, 2))
            for feature, value in zip(X_test.columns, np.mean(np.abs(shap_values), axis=0)) 
        ]
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
        shap_summary_plot(shap_values=shap_values, data=X_test_OHE, model_name = f'finalmodel_{type(selected_model).__name__}')
    else:
        shap_summary_plot(shap_values=shap_values, data=X_test, model_name = f'finalmodel_{type(selected_model).__name__}')


# %% [markdown]
# ##### Significance of features
# 
# Sometimes it is favorable to point out significant features, like statistical analysis, and here we so far had a list of most important (impactful in terms of SHAP values). SHAP summary gives an idea on both population-based importance and individual-based importance of features.
# To have more emphasize on population-based importance (global importance in explainable AI) we apply the following approach based on bootstrap testing.
# 
# The significance test is based on the subsampling method (with replication), where if the IQR crosses zero less than 5% of the time (95% confidence) via subsample_iqr_test function, the feature is marked as significant. The results will be depicted as boxplots with indication of significant features with light green color (as oppposed to light red color for non-significant features) and an "*" in front of the feature name via f_imp_shapboxplot function. This is similarly done for survival models.
# 
# Derivation and interpretation:
# 
# Data-driven threshold: By using the sum of absolute SHAP values and defining the threshold based on the 1st percentile, you're taking into account the overall contribution of each feature across all instances. Features with lower total contributions are compared against this data-derived threshold, rather than simply comparing them against zero.
# 
# Significance Test: For each feature, you conduct a subsampling test to see how often the IQR of the SHAP values crosses this threshold. If it crosses less than 5% of the time, the feature is considered significant and marked with an asterisk.

# %%
# significant features found from this method are denoted by "*" and shown in light green color
def subsample_iqr_test(shap_values, num_subsamples=1000, threshold=0, confidence_level=0.95, random_seed=None):
    """
    Perform subsampling and check if the IQR crosses zero in the SHAP values.
    
    Parameters:
    - shap_values: Array of SHAP values for a given feature
    - num_subsamples: Number of subsamples to generate
    - threshold: Threshold to determine significance (default None means use zero)
    - confidence_level: Threshold for determining significance (default 95%)
    - random_seed: Seed for reproducibility of random sampling
    
    ## Returns
    - proportion_crossing_zero: The proportion of subsamples where the IQR crosses zero
    """
    n = len(shap_values)
    zero_crossings = 0  # Counter for IQR crossing zero

    # Set the random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    for _ in range(num_subsamples):
        # Subsample with replacement
        subsample = np.random.choice(shap_values, size=n, replace=True)
        lower_iqr, upper_iqr = np.percentile(subsample, [25, 75])
        
        # Check if IQR crosses a threshold
        if lower_iqr <= threshold:
            zero_crossings += 1
    
    # Proportion of times the IQR crosses zero
    proportion_crossing_zero = zero_crossings / num_subsamples
    return proportion_crossing_zero

def f_imp_shapboxplot(shap_values, X_test_OHE, X_test, selected_model, data_dictionary, num_features=20, num_subsamples=1000, random_seed=None, apply_threshold = False):
    """
    Plot the SHAP values for the top N most important features.

    This function uses the SHAP (SHapley Additive exPlanations) method to explain the contribution of each feature to a model's predictions.
    It plots a boxplot of the absolute SHAP values for the top N features, with colors indicating significance based on the IQR crossing test.
    
    Parameters:
        shap_values (np.array): The SHAP values for the input data.
        X_test_OHE (pd.DataFrame): The one-hot encoded test data.
        X_test (pd.DataFrame): The raw test data.
        selected_model (class): The model used to select the top N most important features.
        data_dictionary (dict): A dictionary mapping feature names to their corresponding indices in the dataset.
        num_features (int, optional): The number of top features to plot. Defaults to 20.
        num_subsamples (int, optional): The number of subsamples to use for the IQR crossing test. Defaults to 1000.
        random_seed (int, optional): The seed used for randomization. Defaults to None.
        apply_threshold (bool, optional): Whether to apply a threshold to the sum of SHAP values before performing the IQR crossing test. Defaults to False.
    
    ## Returns
        pd.DataFrame: A DataFrame containing the top N most important features, including their median absolute SHAP value, lower and upper quantiles, and subsample proportion crossing zero.
        plt: The plot of the boxplot of absolute SHAP values for the top N features.
    """
    # Use absolute SHAP values for median and quantiles
    abs_shap_values = np.abs(shap_values)
    median_abs_shap_values = np.median(abs_shap_values, axis=0)
    lower_quantiles = np.percentile(abs_shap_values, 25, axis=0)  # 25th percentile of absolute values
    upper_quantiles = np.percentile(abs_shap_values, 75, axis=0)  # 75th percentile of absolute values

    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
        feature_importance_df = pd.DataFrame({
            'Feature': [data_dictionary.get(feature, feature) for feature in X_test_OHE.columns.tolist()],
            'Median_SHAP': median_abs_shap_values,
            'Lower_Quantile': lower_quantiles,
            'Upper_Quantile': upper_quantiles
        })
    else:
        feature_importance_df = pd.DataFrame({
            'Feature': [data_dictionary.get(feature, feature) for feature in X_test.columns.tolist()],
            'Median_SHAP': median_abs_shap_values,
            'Lower_Quantile': lower_quantiles,
            'Upper_Quantile': upper_quantiles
        })

    # Sort the features by median absolute SHAP values in descending order
    feature_importance_df = feature_importance_df.sort_values('Median_SHAP', ascending=False)
    if apply_threshold:
        # Compute the sum of SHAP values for instance
        sum_shap_values = np.sum(shap_values, axis=1)
        print(len(sum_shap_values))

        # Define threshold as the 1st percentile of the sum of SHAP values
        shap_threshold = np.percentile(np.abs(sum_shap_values), 1)
        print(shap_threshold)
    else:
        shap_threshold = 0

    # Select the top N most important features
    top_features = feature_importance_df.head(num_features)

    # Initialize lists to store subsample results
    subsample_results = []
    is_significant = []
    
    for i in top_features.index:
        feature_shap_values = np.abs(shap_values[:, i])
        # Perform the IQR crossing test with subsamples
        proportion_crossing_zero = subsample_iqr_test(feature_shap_values, num_subsamples=num_subsamples, threshold=shap_threshold, random_seed=random_seed)
        
        # A feature is significant if less than (1 - confidence_level)% of the subsamples cross zero
        significant = proportion_crossing_zero <= (1 - 0.95)
        is_significant.append(significant)
        
        subsample_results.append(proportion_crossing_zero)

    # Add the subsample results and significance to the DataFrame
    top_features['Subsample_Proportion_Crossing_Zero'] = subsample_results
    top_features['Significant'] = is_significant
    
    # Mark significant features with an asterisk
    top_features['Feature'] = top_features.apply(lambda row: row['Feature'] + ('*' if row['Significant'] else ''), axis=1)
    
    # Prepare colors based on significance: light green for significant, light red for non-significant
    colors = ['lightgreen' if sig else 'lightcoral' for sig in top_features['Significant']]

    plt.figure(figsize=(10, round(np.max([10, np.log(num_features)]))))
    sns.boxplot(data=np.abs(shap_values[:, top_features.index[:num_features]]), orient='h', whis=[25, 75], 
                width=.5, flierprops={"marker": "x", "alpha": 0.5}, palette=colors, linewidth=0.9)

    # Customize the plot
    plt.yticks(np.arange(num_features), top_features['Feature'], size=8)
    plt.grid(linestyle=':', linewidth=0.5, alpha=0.5)
    plt.xlabel('Absolute SHAP value')
    plt.ylabel('Feature')
    plt.title(f'Distribution of absolute SHAP values for all available features', size=10)
    
    return top_features, plt

# %%
if not skip_block:
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
        f_imp_shapboxplot_fn = X_test_OHE.shape[1]
    else:
        f_imp_shapboxplot_fn = X_test.shape[1]
        
    f_imp_shap_table_testset, f_imp_shapboxplot_testset = f_imp_shapboxplot(shap_values, X_test_OHE, X_test, selected_model, data_dictionary, num_features=f_imp_shapboxplot_fn, random_seed= SEED, apply_threshold=True)

    f_imp_shapboxplot_testset.tight_layout()
    f_imp_shapboxplot_testset.savefig("f_imp_shapboxplot_testset.tif", bbox_inches='tight') 
    f_imp_shapboxplot_testset.show()
    print(f_imp_shap_table_testset)
    f_imp_shap_table_testset.to_excel('f_imp_shap_table_testset.xlsx', index=False)


# %%
if not skip_block:
    if external_val:
        # Calculate SHAP values for the positive class
        positive_class_index = 1 

        if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(selected_model)
            shap_values_extval = explainer.shap_values(X_extval_OHE)
            if isinstance(selected_model, RandomForestClassifier):
                shap_values = shap_values[:,:,1]
        elif isinstance(selected_model, cb.CatBoostClassifier):
            explainer = shap.TreeExplainer(selected_model)
            shap_values_extval = explainer.shap_values(X_extval_data)
        elif isinstance(selected_model, LogisticRegression):
            explainer = shap.LinearExplainer(selected_model, X_train_OHE_nocv)
            shap_values_extval = explainer.shap_values(X_extval_OHE)
        elif isinstance(selected_model, GaussianNB):  
            explainer = shap.Explainer(selected_model.predict_proba, X_train_OHE_nocv)
            shap_values_extval = explainer(X_extval_OHE)  
            shap_values_extval = shap_values_extval.values[:,:,1]
        else:
            explainer = shap.TreeExplainer(selected_model)
            shap_values_extval = explainer.shap_values(X_extval_data)
            
        # Determine which features to use based on the model type
        if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
            feature_names_with_shapvalues = [
                data_dictionary.get(feature, feature) + ": " + str(round(value, 2))
                for feature, value in zip(X_extval_OHE.columns, np.mean(np.abs(shap_values_extval), axis=0)) 
            ]
        else:
            feature_names_with_shapvalues = [
                data_dictionary.get(feature, feature) + ": " + str(round(value, 2))
                for feature, value in zip(X_test.columns, np.mean(np.abs(shap_values_extval), axis=0)) 
            ]
        if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
            shap_summary_plot(shap_values=shap_values_extval, data=X_extval_OHE, model_name = f'finalmodel_extval_{type(selected_model).__name__}')
        else:
            shap_summary_plot(shap_values=shap_values_extval, data=X_extval_data, model_name = f'finalmodel_extval_{type(selected_model).__name__}')

# %%
if not skip_block:
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):

        predictions = selected_model.predict_proba(X_test_OHE)
        predictions = predictions[:, 1]
        if isinstance(selected_model, (HistGradientBoostingClassifier)):
            y_pred = [True if x >= opt_threshold_HGBC else False for x in predictions]
        elif isinstance(selected_model, (RandomForestClassifier)):
            y_pred = [True if x >= opt_threshold_rf else False for x in predictions]
        elif isinstance(selected_model, (LogisticRegression)):
            y_pred = [True if x >= opt_threshold_LR else False for x in predictions]
        else:
            y_pred = [True if x >= opt_threshold_NB else False for x in predictions]
        
        misclassified = y_pred != y_test
    elif isinstance(selected_model, (cb.CatBoostClassifier)):
        predictions = selected_model.predict_proba(X_test)
        predictions = predictions[:, 1]
        y_pred = [True if x >= opt_threshold_CB else False for x in predictions]
        misclassified = y_pred != y_test
    elif isinstance(selected_model, (lgb.LGBMClassifier)):
        predictions = selected_model.predict_proba(X_test)
        predictions = predictions[:, 1]
        y_pred = [True if x >= opt_threshold_LGBM else False for x in predictions]
        misclassified = y_pred != y_test


# %% [markdown]
# ##### Model interpretation only based on correctly classified samples

# %%
if not skip_block:
    shap_values_CorrectClassified = shap_values[misclassified == False]
    # Retrieve feature names from the data dictionary
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier,LogisticRegression, GaussianNB)):
        X_test_CorrectClassified_OHE = X_test_OHE[misclassified == False]
    else:
        X_test_CorrectClassified = X_test[misclassified == False]
        
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
        shap_summary_plot(shap_values=shap_values_CorrectClassified, data=X_test_CorrectClassified_OHE, model_name = f'finalmodel_{type(selected_model).__name__}')
    else:
        shap_summary_plot(shap_values=shap_values_CorrectClassified, data=X_test_CorrectClassified, model_name = f'finalmodel_{type(selected_model).__name__}')


# %% [markdown]
# ##### Feature interactions based on SHAP method
# 
# The below code generates a heatmap visualization representing the interaction between features using SHAP (SHapley Additive exPlanations) values. First, it computes the sum of absolute SHAP values for each pair of features, averaging them over all samples. These interaction scores are stored in an interaction matrix. Next, the interaction matrix is converted into a dataframe for easier plotting, with features as both rows and columns. A mask is created to hide the upper triangle of the heatmap, to eliminate redundant information. Finally, the heatmap is plotted using Seaborn, with feature names on both axes and interaction scores as annotations, providing a visual representation of feature interactions in the model predictions.

# %%
if not skip_block:
    # Separate SHAP values and y_test into two classes
    class_0_indices = np.where(y_test == False)[0]
    class_1_indices = np.where(y_test == True)[0]

    shap_values_class_0 = shap_values[class_0_indices]
    shap_values_class_1 = shap_values[class_1_indices]

    # Get the number of features
    num_samples, num_features = shap_values.shape

    # Initialize matrices to store the median, min, and max SHAP values for each pair of features
    interaction_matrix_median = np.zeros((num_features, num_features))
    interaction_matrix_min = np.zeros((num_features, num_features))
    interaction_matrix_max = np.zeros((num_features, num_features))

    # Calculate the median, min, and max SHAP values for each pair of features
    for i in range(num_features):
        for j in range(i, num_features):
            pairwise_shap_values = shap_values[:, i] + shap_values[:, j]
            interaction_matrix_median[i, j] = np.median(pairwise_shap_values)
            interaction_matrix_median[j, i] = interaction_matrix_median[i, j]
            interaction_matrix_min[i, j] = np.min(pairwise_shap_values)
            interaction_matrix_min[j, i] = interaction_matrix_min[i, j]
            interaction_matrix_max[i, j] = np.max(pairwise_shap_values)
            interaction_matrix_max[j, i] = interaction_matrix_max[i, j]

    # Select appropriate test dataset based on the model type
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
        df = X_test_OHE.copy()
    else:
        df = X_test.copy()

    # DataFrames from the interaction matrices for easier plotting
    interaction_df_median = pd.DataFrame(interaction_matrix_median, index=df.columns, columns=df.columns)
    interaction_df_min = pd.DataFrame(interaction_matrix_min, index=df.columns, columns=df.columns)
    interaction_df_max = pd.DataFrame(interaction_matrix_max, index=df.columns, columns=df.columns)

    # a mask for the upper triangle excluding the diagonal
    mask = np.triu(np.ones_like(interaction_df_median, dtype=bool))

    height = round(np.max([10, np.log(interaction_df_median.shape[0]**2)])) 
    # Ensure height does not exceed the maximum allowed dimension
    max_height = 20000 / 72  # Convert pixels to inches
    if height > max_height:
        height = max_height

    # Plot heatmaps for median, min, and max interactions
    fig, axs = plt.subplots(3, 1, figsize=(10, height*3))

    sns.heatmap(interaction_df_median, annot=True, mask=mask, cmap='coolwarm', linewidths=0.5, fmt=".1f", annot_kws={"size": 8}, ax=axs[0])
    axs[0].set_title('Feature interaction heatmap based on median SHAP values')

    sns.heatmap(interaction_df_min, annot=True, mask=mask, cmap='coolwarm', linewidths=0.5, fmt=".1f", annot_kws={"size": 8}, ax=axs[1])
    axs[1].set_title('Feature interaction heatmap based on minimum SHAP values')

    sns.heatmap(interaction_df_max, annot=True, mask=mask, cmap='coolwarm', linewidths=0.5, fmt=".1f", annot_kws={"size": 8}, ax=axs[2])
    axs[2].set_title('Feature interaction heatmap based on maximum SHAP values')

    plt.tight_layout()
    plt.savefig('f_interaction_heatmap.tif', bbox_inches='tight')
    plt.show()

    # a box plot to show the distribution of interactions for each feature pair
    interaction_values = []
    feature_pairs = []

    for i in range(num_features):
        for j in range(i, num_features):
            pairwise_shap_values = shap_values[:, i] + shap_values[:, j]
            interaction_values.extend(pairwise_shap_values)
            feature_pairs.extend([f"{df.columns[i]} & {df.columns[j]}"] * num_samples)

    interaction_df = pd.DataFrame({'Feature Pair': feature_pairs, 'Interaction Value': interaction_values})

    # Calculate median SHAP values for each feature pair
    median_values = interaction_df.groupby('Feature Pair')['Interaction Value'].median().sort_values(ascending=False)

    # Determine the cutoff values for the top 10% and lowest 10%
    top_10_percent_threshold = median_values.quantile(0.99)
    bottom_10_percent_threshold = median_values.quantile(0.1)

    # Identify the feature pairs within the top 10% and lowest 10%
    top_10_percent_pairs = median_values[median_values >= top_10_percent_threshold].index
    bottom_10_percent_pairs = median_values[median_values <= bottom_10_percent_threshold].index

    # Combine the top and bottom feature pairs
    selected_pairs = top_10_percent_pairs.append(bottom_10_percent_pairs)

    # Filter the DataFrame to include only the selected feature pairs
    interaction_df_filtered = interaction_df[interaction_df['Feature Pair'].isin(selected_pairs)]

    # Determine figure size based on the number of variables
    height = round(np.max([10, np.log(interaction_df_filtered.shape[0]**2)])) 

    # Ensure height does not exceed the maximum allowed dimension
    max_height = 20000 / 72  # Convert pixels to inches
    if height > max_height:
        height = max_height

    # Plot the box plot with the filtered data
    plt.figure(figsize=(height, 5))
    sns.boxplot(x='Feature Pair', y='Interaction Value', data=interaction_df_filtered, order=selected_pairs)
    # Enable autoscaling
    plt.autoscale(enable=True, axis='both')
    plt.xticks(rotation=90)
    plt.title('Box plot of feature interaction values for top and bottom 1% median SHAP values')
    plt.savefig('f_interaction_filtered_bplot.tif')
    plt.show()

# %% [markdown]
# The following script generates several plots to visualize the feature interactions based on SHAP (SHapley Additive exPlanations) values for different classes in a binary classification problem. The specific plots created are:
# 
# 1. **Heatmaps of Feature Interactions:**
#    - **Median SHAP values**: A heatmap showing the median SHAP interaction values between each pair of features.
#    - **Minimum SHAP values**: A heatmap displaying the minimum SHAP interaction values for each pair of features.
#    - **Maximum SHAP values**: A heatmap depicting the maximum SHAP interaction values for each pair of features.
#    
#    These heatmaps are created separately for each class (`False` and `True`). The upper triangle of the heatmap (excluding the diagonal) is masked to avoid redundant information.
# 
# 2. **Box Plots of Feature Interactions:**
#    - **Top and Bottom 10% feature pairs**: A box plot highlighting the feature pairs that fall within the top 10% and bottom 10% of median SHAP interaction values. This helps identify the most and least significant interactions.
# 
# Each type of plot is generated for both classes, resulting in comprehensive visualizations that facilitate the understanding of how different features interact and contribute to the model's predictions. The plots are saved as TIFF files for further analysis and presentation.

# %%
if not skip_block:
    # Define the classes
    classes = [False, True]

    for current_class in classes:
        # Get the indices for the current class
        class_indices = np.where(y_test == current_class)[0]
        
        # Extract SHAP values for the current class
        shap_values_class = shap_values[class_indices]
        
        # Get the number of features
        num_samples, num_features = shap_values_class.shape

        # Initialize matrices to store the median, min, and max SHAP values for each pair of features
        interaction_matrix_median = np.zeros((num_features, num_features))
        interaction_matrix_min = np.zeros((num_features, num_features))
        interaction_matrix_max = np.zeros((num_features, num_features))

        # Calculate the median, min, and max SHAP values for each pair of features
        for i in range(num_features):
            for j in range(i, num_features):
                pairwise_shap_values = shap_values_class[:, i] + shap_values_class[:, j]
                interaction_matrix_median[i, j] = np.median(pairwise_shap_values)
                interaction_matrix_median[j, i] = interaction_matrix_median[i, j]
                interaction_matrix_min[i, j] = np.min(pairwise_shap_values)
                interaction_matrix_min[j, i] = interaction_matrix_min[i, j]
                interaction_matrix_max[i, j] = np.max(pairwise_shap_values)
                interaction_matrix_max[j, i] = interaction_matrix_max[i, j]

        # Select appropriate test dataset based on the model type
        if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
            df = X_test_OHE.copy()
        else:
            df = X_test.copy()

        # DataFrames from the interaction matrices for easier plotting
        interaction_df_median = pd.DataFrame(interaction_matrix_median, index=df.columns, columns=df.columns)
        interaction_df_min = pd.DataFrame(interaction_matrix_min, index=df.columns, columns=df.columns)
        interaction_df_max = pd.DataFrame(interaction_matrix_max, index=df.columns, columns=df.columns)

        # a mask for the upper triangle excluding the diagonal
        mask = np.triu(np.ones_like(interaction_df_median, dtype=bool))
        height = round(np.max([10, np.log(interaction_df_median.shape[0]**2)])) 
        # Ensure height does not exceed the maximum allowed dimension
        max_height = 20000 / 72  # Convert pixels to inches
        if height > max_height:
            height = max_height

        # Plot heatmaps for median, min, and max interactions
        fig, axs = plt.subplots(3, 1, figsize=(10, height*3))

        sns.heatmap(interaction_df_median, annot=True, mask=mask, cmap='coolwarm', linewidths=0.5, fmt=".1f", annot_kws={"size": 8}, ax=axs[0])
        axs[0].set_title(f'Feature interaction heatmap based on median SHAP values for class {int(current_class)}')

        sns.heatmap(interaction_df_min, annot=True, mask=mask, cmap='coolwarm', linewidths=0.5, fmt=".1f", annot_kws={"size": 8}, ax=axs[1])
        axs[1].set_title(f'Feature interaction heatmap based on minimum SHAP values for class {int(current_class)}')

        sns.heatmap(interaction_df_max, annot=True, mask=mask, cmap='coolwarm', linewidths=0.5, fmt=".1f", annot_kws={"size": 8}, ax=axs[2])
        axs[2].set_title(f'Feature interaction heatmap based on maximum SHAP values for class {int(current_class)}')

        plt.tight_layout()
        plt.savefig(f'f_interaction_heatmap_{current_class}.tif', bbox_inches='tight')
        # plt.show()

        # Create a box plot to show the distribution of interactions for each feature pair
        interaction_values = []
        feature_pairs = []

        for i in range(num_features):
            for j in range(i, num_features):
                pairwise_shap_values = shap_values_class[:, i] + shap_values_class[:, j]
                interaction_values.extend(pairwise_shap_values)
                feature_pairs.extend([f"{df.columns[i]} & {df.columns[j]}"] * num_samples)

        interaction_df = pd.DataFrame({'Feature Pair': feature_pairs, 'Interaction Value': interaction_values})
        
        # Remove feature pairs where a feature interacts with itself
        interaction_df = interaction_df[~interaction_df['Feature Pair'].apply(lambda x: x.split(' & ')[0] == x.split(' & ')[1])]

        # Calculate median SHAP values for each feature pair
        median_values = interaction_df.groupby('Feature Pair')['Interaction Value'].median().sort_values(ascending=False)

        # Determine the cutoff values for the top 10% and lowest 10%
        top_10_percent_threshold = median_values.quantile(0.90)
        bottom_10_percent_threshold = median_values.quantile(0.10)

        # Identify the feature pairs within the top 10% and lowest 10%
        top_10_percent_pairs = median_values[median_values >= top_10_percent_threshold].index
        bottom_10_percent_pairs = median_values[median_values <= bottom_10_percent_threshold].index

        # Combine the top and bottom feature pairs
        selected_pairs = top_10_percent_pairs.append(bottom_10_percent_pairs)

        # Filter the DataFrame to include only the selected feature pairs
        interaction_df_filtered = interaction_df[interaction_df['Feature Pair'].isin(selected_pairs)]

        # Determine figure size based on the number of variables
        height = round(np.max([10, np.log(interaction_df_filtered.shape[0]**2)])) 

        # Ensure height does not exceed the maximum allowed dimension
        max_height = 20000 / 72  # Convert pixels to inches
        if height > max_height:
            height = max_height

        # Plot the box plot with the filtered data
        plt.figure(figsize=(height, 5))
        sns.boxplot(x='Feature Pair', y='Interaction Value', data=interaction_df_filtered, order=selected_pairs)
        plt.xticks(rotation=90)
        plt.title(f'Box plot of feature interaction values for top and bottom 10% median SHAP values for class {int(current_class)}')
        plt.savefig(f'f_interaction_filtered_bplot_{current_class}.tif', dpi = 300)
        # plt.show()


# %% [markdown]
# In this context, "interaction" refers to the combined effect of two features on the model's prediction, as measured by their SHAP values. SHAP (SHapley Additive exPlanations) values provide a way to interpret the contribution of each feature to the prediction of a machine learning model. Here, the interaction between two features is quantified by combining their SHAP values and assessing how these combined values influence the prediction.
# 
# Specifically:
# 
# 1. **Pairwise SHAP Values**: The interaction between two features \( i \) and \( j \) is evaluated by summing their individual SHAP values for each sample. This summed value represents the joint contribution of both features to the prediction for that sample.
# 
# 2. **Interaction Metrics**:
#    - **Median Interaction**: The median of the pairwise SHAP values across all samples for a given class.
#    - **Minimum Interaction**: The minimum of the pairwise SHAP values across all samples for a given class.
#    - **Maximum Interaction**: The maximum of the pairwise SHAP values across all samples for a given class.
# 
# These metrics are calculated for each pair of features, resulting in matrices that summarize the interactions. The script then visualizes these interactions through heatmaps and box plots to provide insights into how pairs of features work together to influence the model's predictions for different classes.
# 
# ### Additional analyses on model interpretation and evaluation:
# 
# 1. **Heatmaps**:
#    - **Median SHAP Interaction Heatmap**: Shows the median combined effect of feature pairs.
#    - **Minimum SHAP Interaction Heatmap**: Shows the smallest combined effect of feature pairs.
#    - **Maximum SHAP Interaction Heatmap**: Shows the largest combined effect of feature pairs.
# 
# 2. **Box Plots**:
#    - **Top and Bottom 10% Feature Pairs Box Plot**: Highlights the feature pairs with the most and least significant interactions, based on the median SHAP values.

# %% [markdown]
# ##### Feature interactions based on feature permutation method for feature pairs
# 
# this code provides insight into the interaction effects between pairs of features in the machine learning model, helping identify which combinations of features contribute significantly to the model's performance.
# 
# - The `permute_feature_pairs` function calculates the permutation importances for pairs of features.
# - It converts the binary target variable to numeric format and calculates the baseline score using ROC AUC.
# - For each pair of features, it shuffles their values multiple times and computes the change in ROC AUC compared to the baseline. The average change in ROC AUC is stored as the importance score for that feature pair.
# 
# - It generates all possible pairs of features from the input feature set.
# - It computes the permutation importances for pairs of features using the defined function.
# - The results are stored in a DataFrame, where each row represents a feature pair along with its importance score.
# - The DataFrame is sorted based on importance in descending order and printed to display the importance of feature pairs.

# %%
if not skip_block:
    # Function to permute pairs of features
    if find_interacting_feature_permutation:
        def permute_feature_pairs(model, X, y, pairs, n_repeats, random_state, scoring, n_jobs):
            """
            Computes the permutation importance of pairs of features in a given model.This function computes the permutation importance of pairs of features in a given machine learning model. It takes as input:

            *   The model to be used
            *   The input data `X`
            *   The target variable `y`
            *   A list of tuples, where each tuple contains two feature indices `pairs`
            *   The number of times to repeat the permutation for each pair `n_repeats`
            *   The random seed for reproducibility `random_state`
            *   A function to evaluate the model's performance on a given split of data `scoring`
            *   The number of CPU cores to use for parallel computation `n_jobs`

            The function returns a dictionary where the keys are tuples of feature indices and the values are the average permutation importances for each pair. This allows users to easily identify which pairs of features have the most significant impact on the model's performance.

            In this specific implementation, the function is used to compute the permutation importance of all pairs of features in the test data `X_test_OHE` (or `X_test` if the model can handle missing values). The results are then stored in a pandas DataFrame for easy visualization and analysis.

            Parameters:
                model (object): The machine learning model to be used.
                X (pandas DataFrame or numpy array): The input data.
                y (pandas Series): The target variable.
                pairs (list): A list of tuples, where each tuple contains two feature indices.
                n_repeats (int): The number of times to repeat the permutation for each pair.
                random_state (int): The random seed for reproducibility.
                scoring (function): A function to evaluate the model's performance on a given split of data.
                n_jobs (int): The number of CPU cores to use for parallel computation.

            ## Returns
                dict: A dictionary where the keys are tuples of feature indices and the values are the average permutation importances for each pair.
            """
            y = y.replace({class_1: 1, class_0: 0})
            baseline_score = roc_auc_score(y, model.predict_proba(X)[:, 1])
            importance_dict = {}

            for feature1, feature2 in pairs:
                importances = []
                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    X_permuted[[feature1, feature2]] = X_permuted[[feature1, feature2]].sample(frac=1, random_state=random_state).values
                    permuted_score = roc_auc_score(y, model.predict_proba(X_permuted)[:, 1])
                    importances.append(baseline_score - permuted_score)
                
                importance_dict[(feature1, feature2)] = sum(importances) / n_repeats  # Average importance over n_repeats
            
            return importance_dict

        if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier,LogisticRegression, GaussianNB)):
            df = X_test_OHE.copy()
        else:
            df = X_test.copy()

        # Define the features
        features = df.columns.tolist()
        pairs = list(itertools.combinations(features, 2))

        # Calculate the permutation importances for pairs of features
        pair_importances = permute_feature_pairs(
            selected_model, df, y_test, pairs, n_repeats = n_rep_feature_permutation, random_state = SEED, scoring = custom_scorer, n_jobs=n_cpu_model_training
        )

        # a DataFrame to store the results
        pair_importance_df = pd.DataFrame(
            {"feature pair": [f"{pair[0]} + {pair[1]}" for pair in pair_importances.keys()],
            "importance": pair_importances.values()}
        )

        # Sort by importance
        pair_importance_df = pair_importance_df.sort_values(by="importance", ascending=False)
        print(pair_importance_df)


# %% [markdown]
# ##### SHAP decision plot 
# 
# The SHAP decision plot centers around the `explainer.expected_value` on the x-axis, with colored lines representing predictions for each observation. Moving upwards, these lines intersect the x-axis at the prediction specific to each observation, depicted in varying colors on a gradient scale. The plot integrates SHAP values for each feature, illustrating their contributions to the overall prediction relative to the model's baseline value. At the plot's bottom, observations converge at `explainer.expected_value`.
# 
# 1. **Demonstrating feature effects:**
#    - Visualizes the impact of multiple features on predictions and their individual contributions.
# 
# 2. **Revealing interaction effects:**
#    - shows how interactions between features influence predictions by incorporating SHAP values.
# 
# 3. **Exploring feature effects across values:**
#    - Enables exploration of feature effects by showcasing prediction variations across different feature values.
# 
# 4. **Identifying outliers:**
#    - Enables outlier detection by pinpointing observations deviating significantly from expected values or prediction trends.
# 
# 5. **Understanding prediction paths:**
#    - Facilitates the identification of common prediction patterns, offering insight into model behavior.
# 
# 6. **Model comparison:**
#    - Allows comparing predictions across multiple models.

# %%
if not skip_block:
    # Plot the SHAP decision plot with only significant features
    if isinstance(selected_model, (HistGradientBoostingClassifier)):
        shap.decision_plot(explainer.expected_value, 
                        shap_values,
                        X_test_OHE,
                        alpha=0.5, 
                        feature_names=feature_names_with_shapvalues,
                        link='logit',
                        highlight=misclassified,
                        ignore_warnings=True,
                        feature_order = None)
    elif isinstance(selected_model, (RandomForestClassifier)):
        shap.decision_plot(explainer.expected_value[positive_class_index], 
                        shap_values,
                        X_test_OHE,
                        alpha=0.5, 
                        feature_names=feature_names_with_shapvalues,
                        link='logit',
                        highlight=misclassified,
                        ignore_warnings=True,
                        feature_order = None)
    elif isinstance(selected_model, (LogisticRegression)):
        shap.decision_plot(explainer.expected_value, 
                        shap_values,
                        X_test_OHE,
                        alpha=0.5, 
                        feature_names=feature_names_with_shapvalues,
                        link='logit',
                        highlight=misclassified,
                        ignore_warnings=True,
                        feature_order = None)
    elif isinstance(selected_model, (cb.CatBoostClassifier)):
        shap.decision_plot(explainer.expected_value, 
                        shap_values,
                        X_test,
                        alpha=0.5, 
                        feature_names=feature_names_with_shapvalues,
                        link='logit',
                        highlight=misclassified,
                        ignore_warnings=True,
                        feature_order = None)
    elif isinstance(selected_model, (GaussianNB)):
        shap_values_NB = explainer(X_test_OHE)
        shap.decision_plot(base_value = shap_values_NB.base_values[1][1],
                        shap_values = shap_values,
                        features = X_test_OHE,
                        alpha=0.5, 
                        feature_names=feature_names_with_shapvalues,
                        link='logit',
                        highlight=misclassified, 
                        ignore_warnings=True,
                        feature_order = None)
    else:
        shap.decision_plot(explainer.expected_value, 
                        shap_values,
                        X_test,
                        alpha=0.5, 
                        feature_names=feature_names_with_shapvalues,
                        link='logit',
                        highlight=misclassified, 
                        ignore_warnings=True,
                        feature_order = None)

    plt.gca().set_facecolor('white')  
    # display grid lines
    plt.grid(which='both', color="grey")
    # modify grid lines
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.3)
    plt.savefig('shap_decision_allfeats_plot.tif', bbox_inches='tight')

    # Close the extra figure
    plt.close()
    # show the plot
    plt.show()


# %% [markdown]
# ##### SHAP dependence plots

# %%
if not skip_block:
    sns.set(style="ticks")

    # Set the background color to white
    sns.set_style("white")

    plt.rcParams["figure.figsize"] = (10, 5)

    # Compute median absolute SHAP values for each feature
    median_abs_shap_values = np.median(np.abs(shap_values), axis=0)

    # Sort features by median absolute SHAP values in descending order
    sorted_features = np.argsort(median_abs_shap_values)[::-1]

    # Calculate the number of features to plot
    # num_features_to_plot = min(np.sum(median_abs_shap_values > 0), top_n_f)
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
        num_features_to_plot = min(top_n_f,X_test_OHE.shape[1])
    else:
        num_features_to_plot = min(top_n_f,X_test.shape[1])

    # Set the number of columns for subplots
    num_cols = 4

    # Calculate the number of rows for subplots
    num_rows = int(np.ceil(num_features_to_plot / num_cols))

    # Initialize a subplot
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    axs = axs.ravel()

    # Track the current subplot index
    current_subplot = 0

    # Iterate over the top features
    for feature in sorted_features[:num_features_to_plot]:
        # Get feature name
        if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
            feature_name = X_test_OHE.columns[feature]
            x = X_test_OHE.iloc[:, feature]
        else:
            feature_name = X_test.columns[feature]
            if X_test[feature_name].dtype.name == 'category':
                # Convert categorical feature to numerical using LabelEncoder
                encoder = LabelEncoder()
                X_test_encoded = X_test.copy()
                X_test_encoded[feature_name] = encoder.fit_transform(X_test[feature_name])
                x = X_test_encoded.iloc[:, feature].astype(float)
            else:
                x = X_test.iloc[:, feature].astype(float)
        
        # Handle missing values in feature values and SHAP values
        mask_x = ~pd.isnull(x)
        mask_shap = ~np.isnan(shap_values[:, feature])
        mask = mask_x & mask_shap
        
        x_filtered = x[mask]
        shap_values_filtered = shap_values[:, feature][mask]
        predictions_filtered = predictions[mask]
        misclassified_filtered = misclassified[mask]
        
        # Check if all x values are identical
        if len(np.unique(x_filtered)) == 1:
            print(f"Skipped feature {feature_name} because all x values are identical.")
            continue
        
        # Calculate Spearman correlation coefficient and p-value
        correlation, p_value = spearmanr(x_filtered, shap_values_filtered, nan_policy='omit')
        
        # Create scatter plot in the current subplot
        scatter = axs[current_subplot].scatter(x_filtered, shap_values_filtered, c=predictions_filtered, cmap='viridis', alpha=0.7, s=50)
        axs[current_subplot].set_xlabel(feature_name)
        axs[current_subplot].set_ylabel("SHAP Value")
        
        # Add correlation line
        slope, intercept, r_value, p_value_corr, std_err = linregress(x_filtered, shap_values_filtered)
        axs[current_subplot].plot(x_filtered, slope * x_filtered + intercept, color='red')
        
        # Mark misclassified samples with 'x'
        axs[current_subplot].scatter(x_filtered[misclassified_filtered], shap_values_filtered[misclassified_filtered], marker="X", color='red', alpha=0.5, s=50)
        
        # Customize colorbar
        cbar = plt.colorbar(scatter, ax=axs[current_subplot])
        cbar.set_label("Predicted Probability")
        
        # Check if correlation is statistically significant
        if not np.isnan(correlation) and not np.isnan(p_value_corr):
            _, p_value_corr_test = ttest_rel(x_filtered, shap_values_filtered)
            p_value_text = f"p < 0.05" if p_value_corr_test < 0.05 else f"p = {p_value_corr_test:.2f}"
            axs[current_subplot].set_title(f"{feature_name} vs. SHAP Value\nSpearman Correlation: {correlation:.2f}, {p_value_text}")
        else:
            axs[current_subplot].set_title(f"{feature_name} vs. SHAP Value\nCorrelation: N/A")
        
        # Increment the current subplot index
        current_subplot += 1

    # Hide any remaining empty subplots
    for i in range(current_subplot, num_rows * num_cols):
        axs[i].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig('shap_dependence_plot.tif', bbox_inches='tight')

    # Show the plot
    plt.show()

# %% [markdown]
# ##### SHAP clustering
# 
# In the context of precision medicine, SHAP clustering serves to uncover patient subgroups with distinct patterns, leading to differential model behavior. These subgroups often manifest as low or high-risk clusters, with a potential third cluster exhibiting less decisive model behavior. Identifying these clusters aids in profiling patient subgroups, offering valuable insights for model application to new patients.

# %%
if not skip_block:
    def find_feature_clusters(X, shap_values, selected_model, data_dictionary, top_n_f):
        """
        This function analyzes SHAP values to identify feature clusters and instance clusters using hierarchical clustering. The goal is to visualize the relationships between features and instances in the dataset.

        ### Parameters

        *   `X`: A pandas DataFrame containing the feature data.
        *   `shap_values`: A numpy array of shape `(n_samples, n_features)` containing SHAP values for each instance and feature.
        *   `selected_model`: The model used to select features (not used in this function).
        *   `data_dictionary`: Not used in this function (possibly used elsewhere in the codebase).
        *   `top_n_f`: An integer specifying the number of top clusters to display for features.

        ### Returns

        A tuple containing three elements:

        1.  A dictionary (`top_n_col_clusters_info`) where keys are cluster indices and values are lists of feature names within each cluster, ordered by the number of features in descending order.
        2.  A dictionary (`row_clusters_info`) where keys are cluster indices and values are lists of instance indices within each cluster, ordered by the number of instances in descending order.
        3.  A DataFrame (`shap_df`) containing SHAP values with feature names as columns.

        ### Functionality

        1.  The function first creates a clustermap plot showing relationships between features and instances using SHAP values.
        2.  It then performs hierarchical clustering on the column (feature) data to identify clusters and groups them into `top_n_f` top clusters based on the number of features in each cluster.
        3.  Next, it determines the best number of clusters for row (instance) data using the silhouette score and plots a graph showing the relationship between the number of clusters and the silhouette scores.
        4.  After finding the optimal number of clusters for rows, it performs hierarchical clustering on the row data with the chosen number of clusters and groups instances into `best_num_clusters` top clusters based on the number of instances in each cluster.
        5.  Finally, the function returns three dictionaries and a DataFrame containing SHAP values: one for feature clusters, one for instance clusters, and one for the original SHAP value DataFrame.

        ### Notes

        *   This function assumes that the input data is properly formatted and does not contain any errors or inconsistencies.
        *   The best number of clusters for row data can be manually adjusted by modifying the range in the `silhouette_scores` loop (currently set to 3-5).
        """
        # a DataFrame for SHAP values with feature names as columns
        shap_df = pd.DataFrame(shap_values, columns=X.columns)

        # Plot clustermap for both rows and columns
        cluster_grid = sns.clustermap(shap_df)
        plt.title('Clustermap for Features and Instances')
        plt.show()
        
        # Perform hierarchical clustering on columns (features)
        col_clusters = AgglomerativeClustering(n_clusters=3).fit_predict(shap_values.T)

        # Group columns into clusters
        features_in_clusters = {i: [] for i in range(3)}

        for i, cluster_idx in enumerate(col_clusters):
            features_in_clusters[cluster_idx].append(X.columns[i])

        # Get top N clusters for features
        top_n_col_clusters = sorted(features_in_clusters.keys(), key=lambda x: len(features_in_clusters[x]), reverse=True)[:top_n_f]
        top_n_col_clusters_info = {cluster: features_in_clusters[cluster] for cluster in top_n_col_clusters}

        # Determine the best number of clusters for rows (instances) using silhouette score
        best_num_clusters = 3
        best_silhouette_score = -1
        silhouette_scores = []

        for n_clusters in range(3, 6):
            row_clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(shap_values)
            silhouette_avg = silhouette_score(shap_values, row_clusters)
            silhouette_scores.append((n_clusters, silhouette_avg))
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_num_clusters = n_clusters

        # Plot silhouette scores
        cluster_counts, scores = zip(*silhouette_scores)
        plt.figure()
        plt.plot(cluster_counts, scores, marker='o')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.show()

        # Perform hierarchical clustering on rows (instances) with the best number of clusters
        row_clusters = AgglomerativeClustering(n_clusters=best_num_clusters).fit_predict(shap_values)

        # Group rows into clusters
        instances_in_clusters = {i: [] for i in range(best_num_clusters)}

        for i, cluster_idx in enumerate(row_clusters):
            instances_in_clusters[cluster_idx].append(i)

        # Get top N clusters for instances
        top_N_row_clusters = sorted(instances_in_clusters.keys(), key=lambda x: len(instances_in_clusters[x]), reverse=True)[:best_num_clusters]
        row_clusters_info = {cluster: instances_in_clusters[cluster] for cluster in top_N_row_clusters}

        return top_n_col_clusters_info, row_clusters_info, shap_df


    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier,LogisticRegression, GaussianNB)):
        top_n_col_clusters_info, row_clusters_info, shap_df = find_feature_clusters(X=X_test_OHE, 
                                                                                    shap_values=shap_values,
                                                                                    selected_model=selected_model, 
                                                                                    data_dictionary=data_dictionary, 
                                                                                    top_n_f=top_n_f)
    elif isinstance(selected_model, (cb.CatBoostClassifier, lgb.LGBMClassifier)):
        top_n_col_clusters_info, row_clusters_info, shap_df = find_feature_clusters(X=X_test, 
                                                                                    shap_values=shap_values,
                                                                                    selected_model=selected_model, 
                                                                                    data_dictionary=data_dictionary, 
                                                                                    top_n_f=top_n_f)
        
      
    with open('cluster_info.txt', 'w') as f:
        f.write("Top N clusters for features:\n")
        for cluster, features in top_n_col_clusters_info.items():
            f.write(f"Cluster {cluster}: {features}\n")
        
        f.write("\nTop N clusters for instances:\n")
        for cluster, instances in row_clusters_info.items():
            f.write(f"Cluster {cluster}: {instances}\n")

    print("Cluster information has been saved to 'cluster_info.txt'")

# %%
if not skip_block:
    def plot_confusion_matrix_for_clusters(X, y, cluster_info, model, shap_values, top_n):
        """
        Plots a confusion matrix for each cluster in the test data and visualizes the feature importance using SHAP values.

        Parameters:
            X (pandas DataFrame or numpy array): The input data.
            y (pandas Series): The target variable.
            cluster_info (dict): A dictionary indicating the instances in each cluster.
            model (object): The machine learning model to be used.
            shap_values (numpy array): The SHAP values for the given model and data.
            top_n (int): The number of features to display.

        This function:
        1.  Iterates over each unique cluster.
        2.  Subset the test data and target variable for the current cluster.
        3.  Computes the predicted labels using the selected model on the subsetted data.
        4.  Calculates the confusion matrix between the actual labels and predicted labels.
        5.  Plots a heatmap of the confusion matrix with actual label names.
        6.  Displays the feature importance for the current cluster using SHAP values.

        """
        # Get unique class labels from your data
        unique_labels = np.unique(y)

        # Get unique cluster labels
        unique_clusters = list(cluster_info.keys())

        # Iterate over each cluster
        for cluster in unique_clusters:
            # Subset the test data for the current cluster
            cluster_indices = cluster_info[cluster]
            X_cluster = X.iloc[cluster_indices]
            y_cluster = y.iloc[cluster_indices]
            shap_values_cluster = shap_values[cluster_indices]

            if isinstance(model, (RandomForestClassifier)):
                predictions_proba_cluster = model.predict_proba(X_cluster)[:, 1]
                model_opt_threshold = opt_threshold_rf
            elif model == 'QLattice':
                predictions_proba_cluster = model.predict(X_cluster)[:, 1]
            elif isinstance(model, (HistGradientBoostingClassifier)):
                predictions_proba_cluster = model.predict_proba(X_cluster)[:, 1]
                model_opt_threshold = opt_threshold_HGBC
            elif isinstance(model, (lgb.LGBMClassifier)):
                predictions_proba_cluster = model.predict_proba(X_cluster)[:, 1]
                model_opt_threshold = opt_threshold_LGBM
            elif isinstance(model, (cb.CatBoostClassifier)):
                predictions_proba_cluster = model.predict_proba(X_cluster)[:, 1]
                model_opt_threshold = opt_threshold_CB
            elif isinstance(model, (LogisticRegression)):
                predictions_proba_cluster = model.predict_proba(X_cluster)[:, 1]
                model_opt_threshold = opt_threshold_LR
            elif isinstance(model, (GaussianNB)):
                predictions_proba_cluster = model.predict_proba(X_cluster)[:, 1]
                model_opt_threshold = opt_threshold_NB
            
            predictions_class_cluster = np.where(predictions_proba_cluster >= model_opt_threshold, True, False)

            # Compute confusion matrix
            cm = confusion_matrix(y_cluster, predictions_class_cluster, labels=[False, True])

            # Plot confusion matrix with actual label names
            plt.figure(figsize=(4, 3))
            myheatmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_labels_display, yticklabels=class_labels_display, annot_kws={"size": 8})  # Adjust annot_kws to change annotation font size
            myheatmap.invert_yaxis()
            plt.title(f"Confusion Matrix for Cluster {cluster}", fontsize=10)
            plt.xlabel("Predicted Label", fontsize=8)
            plt.ylabel("True Label", fontsize=8)
            plt.xticks(fontsize=8) 
            plt.yticks(fontsize=8)  
            plt.show()

            # Call categorical_shap_plot for the current cluster
            if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
                shap.summary_plot(shap_values_cluster, X_cluster, show=False, alpha = 0.8, max_display=top_n_f)
            elif any(X_test[column].dtype == 'category' for column in X_test.columns):
                categorical_shap_plot(shap_values=shap_values_cluster, 
                                            data=X_cluster,
                                            top_n=min(top_n, X_cluster.shape[1]),
                                            jitter=0.1)
            else: # it could be a LGBM or CATBoost model based on only numerical features 
                shap.summary_plot(shap_values_cluster, X_cluster, show=False, alpha = 0.8, max_display=top_n_f)

    # based on the assumption that cluster_info is the dictionary indicating the instances in each cluster
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier,LogisticRegression, GaussianNB)):
        plot_confusion_matrix_for_clusters(X=X_test_OHE, y=y_test, cluster_info=row_clusters_info, model=selected_model, shap_values=shap_values, top_n=top_n_f)

    elif isinstance(selected_model, (cb.CatBoostClassifier, lgb.LGBMClassifier)):
        plot_confusion_matrix_for_clusters(X=X_test, y=y_test, cluster_info=row_clusters_info, model=selected_model, shap_values=shap_values, top_n=top_n_f)



# %% [markdown]
# #### SHAP force plot for individuals (e.g., one patient)

# %%
if not skip_block:
    # load JS visualization code to notebook
    shap.initjs()
    # Function to get a sample from each class
    def get_samples_per_class(X, y):
        samples = {}
        for class_value in np.unique(y):
            # Select a sample from each class
            class_samples = X[y == class_value]
            if class_samples.shape[0] > 0:
                samples[class_value] = class_samples.sample(n=1, random_state=SEED)  # Random sample
        return samples
    if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
        # Get samples from each class
        mysamples = get_samples_per_class(X_test_OHE, y_test)
    else:
        mysamples = get_samples_per_class(X_test, y_test)
        
    try:
        # Generate SHAP force plots for each sample
        for class_value, sample in mysamples.items():
            
            # Convert sample to appropriate format for explainer
            if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
                sample_predicted_prob = selected_model.predict_proba(sample)
                print(f"predicted probability: {np.round(sample_predicted_prob,2)}")
                print(f"class: {class_value}")
                # Get the SHAP values for the sample
                sample_shap_values = explainer(sample)
                # Generate the force plot
                base_value = sample_shap_values.base_values
                # shap.plots.force(base_value, sample_shap_values.values[:,:,1])
                display(shap.plots.force(base_value, sample_shap_values.values, plot_cmap="RdBu", feature_names=X_train_OHE_nocv.columns))
            
            elif isinstance(selected_model, (cb.CatBoostClassifier, lgb.LGBMClassifier)):
                sample_predicted_prob = selected_model.predict_proba(sample)
                print(f"predicted probability: {sample_predicted_prob[:,1]}")
                # Get the SHAP values for the sample
                sample_shap_values = explainer(sample)
                # Generate the force plot
                display(shap.plots.force(sample_shap_values, plot_cmap="RdBu", feature_names=feature_names))
    except Exception as e:
        print(f"An error occurred: {e}. Skipping to the next block.")

# %% [markdown]
# #### Decision curve analysis
# Net benefit of the model compared to random guessing, extreme cases, and an alternative method or model. Read more here: https://en.wikipedia.org/wiki/Decision_curve_analysis#:~:text=Decision%20curve%20analysis%20evaluates%20a,are%20positive%20are%20also%20plotted.
# As an alternative model we here use logistic regression model but you can modify this or import prediction probabilities for the test samples from elsewhere.

# %%
if not skip_block:
    if do_decision_curve_analysis:

        # Calculate necessary information for adjusting hyperparameters
        n_rows = X_train_OHE_nocv.shape[0]
        n_cols = X_train_OHE_nocv.shape[1]
        class_proportion = y_train.mean()  # binary classification
        rf_params, lgbm_params, hgbc_params, cb_params, lr_params = set_parameters(n_rows, n_cols, class_proportion)
        # Create a Logistic Regression instance
        lr = LogisticRegression(penalty='l1',random_state=SEED, solver="liblinear", **lr_params)
        # lr = LogisticRegression(penalty='elasticnet',random_state=SEED, solver="saga")
        if hp_tuning:
        
            # Adjust hyperparameters based on the training data in this fold
            rf_param_dist, lgbm_param_dist, hgbc_param_dist, cb_param_dist, lr_param_dist = adjust_hyperparameters(n_rows, n_cols)
            # Create a RandomizedSearchCV instance
            random_search = RandomizedSearchCV(
                estimator=lr,
                param_distributions=lr_param_dist,
                n_iter=n_iter_hptuning,
                scoring= custom_scorer, 
                cv=cv_folds_hptuning,
                refit=True, 
                random_state=SEED,
                verbose=0,
                n_jobs=n_cpu_for_tuning
            )

            # Perform the random search on the training data
            random_search.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)

            # Get the best parameters and best estimator
            best_params = random_search.best_params_
            lr = LogisticRegression(penalty='l1',random_state=SEED, solver="liblinear", **best_params)
            # lr = LogisticRegression(penalty='elasticnet',random_state=SEED, solver="saga", **best_params)
        else:
            lr = LogisticRegression(penalty='l1',random_state=SEED, solver="liblinear", **lr_params)

        # Fit the best estimator on the entire training data
        lr.fit(X_train_OHE_nocv, y_train, sample_weight=sample_weights)

# %% [markdown]
# #### Cost-sensitive model evaluation
# 
# Here we introduce cost-sensitive net benefit where we introduce weights as coefficients for the number of TP and FP cases.
# The weights are supposed to be positive values >0. Use w_tp=1, w_fp=1 for true positives and false positives respectively when they are equally important (equal cost).

# %%
if not skip_block:
    if do_decision_curve_analysis:
        def cost_sensitive_net_benefit(tp, fp, threshold, N, w_tp=1, w_fp=1):
            """
            Calculates the net benefit of a classification model under a given probability threshold.
            
            The net benefit is calculated as the weighted sum of the positive and negative classes, 
            where the weights are based on the prior probabilities of the classes. This metric allows for 
            cost-sensitive comparison between different models, taking into account the costs associated with 
            false positives and false negatives.
            
            ## Parameters
                tp (int): True positives in the current test set.
                fp (int): False positives in the current test set.
                threshold (float): Probability threshold used to determine predicted positive classes.
                N (int): Total number of samples in the test set.
                w_tp (float, optional): Weight assigned to true positives. Defaults to 1.
                w_fp (float, optional): Weight assigned to false positives. Defaults to 1.

            ## Returns
                float: Net benefit of the model under the given threshold.
            """
            # Calculate cost-sensitive net benefit
            if N == 0:
                return 0  # Prevent division by zero
            net_benefit = (tp * w_tp / N) - (fp * w_fp / N) * (threshold / (1 - threshold))
            return net_benefit
        def decision_curve_analysis(pred_probs_selected_model, pred_probs_alternative_model, rand_pred_probs, w_tp=1, w_fp=1):
            """
            Performs cost-sensitive decision curve analysis on two models.
            
            The analysis plots the net benefit of each model under different probability thresholds 
            and allows for comparison between the two models. The plot also includes a reference line 
            representing the default threshold used in many machine learning applications.
            
            ## Parameters
                pred_probs_selected_model (array-like): Predicted probabilities of positive classes for the first model.
                pred_probs_alternative_model (array-like): Predicted probabilities of positive classes for the second model.
                rand_pred_probs (array-like): Randomly generated predicted probabilities for a third set of models.

            ## Returns
                None
            """
            N = len(y_test)  # Total number of observations
            
            # Precompute TP and FP for extreme cases
            tp_all_positive = np.sum(y_test.values == True)
            fp_all_positive = np.sum(y_test.values == False)
            
            # Extreme cases do not vary by threshold
            net_benefit_all_positive = [cost_sensitive_net_benefit(tp_all_positive, fp_all_positive, threshold, N, w_tp, w_fp) for threshold in np.linspace(0, 1, 100)]
            net_benefit_all_negative = [cost_sensitive_net_benefit(0, 0, threshold, N, w_tp, w_fp) for threshold in np.linspace(0, 1, 100)]

            # Initialize lists for storing net benefits
            net_benefit_selected_model = []
            net_benefit_alternative_model = []
            net_benefit_rand = []

            threshold_range = np.linspace(0, 1, 100)
            
            for threshold in threshold_range:
                # Calculate TP and FP for selected model
                tp_selected_model = np.sum((pred_probs_selected_model > threshold) & (y_test.values == True))
                fp_selected_model = np.sum((pred_probs_selected_model > threshold) & (y_test.values == False))
                
                # Calculate TP and FP for alternative model
                tp_alternative_model = np.sum((pred_probs_alternative_model > threshold) & (y_test.values == True))
                fp_alternative_model = np.sum((pred_probs_alternative_model > threshold) & (y_test.values == False))
                
                # Calculate TP and FP for random predictions
                tp_rand = np.sum((rand_pred_probs > threshold) & (y_test.values == True))
                fp_rand = np.sum((rand_pred_probs > threshold) & (y_test.values == False))
                
                # Calculate net benefits
                net_benefit_selected_model.append(cost_sensitive_net_benefit(tp_selected_model, fp_selected_model, threshold, N, w_tp, w_fp))
                net_benefit_alternative_model.append(cost_sensitive_net_benefit(tp_alternative_model, fp_alternative_model, threshold, N, w_tp, w_fp))
                net_benefit_rand.append(cost_sensitive_net_benefit(tp_rand, fp_rand, threshold, N, w_tp, w_fp))

            # Find the maximum net benefit for y-axis limit
            max_net_benefit = max(
                max(net_benefit_selected_model),
                max(net_benefit_alternative_model),
                max(net_benefit_rand),
                max(net_benefit_all_positive),
                max(net_benefit_all_negative)
            )

            # Plot decision curve
            plt.plot(threshold_range, net_benefit_selected_model, label='Selected model')
            plt.plot(threshold_range, net_benefit_alternative_model, label='Alternative model')
            plt.plot(threshold_range, net_benefit_rand, label='Random predictions')
            plt.plot(threshold_range, net_benefit_all_positive, label='All positive')
            plt.plot(threshold_range, net_benefit_all_negative, label='All negative')

            plt.axvline(x=0.5, color='k', linestyle='--', label='Default threshold (0.5)')
            
            plt.xlabel('Probability threshold')
            plt.ylabel('Net benefit')
            plt.title('Cost-sensitive decision curve analysis')
            plt.ylim(bottom=-0.01, top=max_net_benefit + 0.01)
            plt.legend()
            plt.show()
            
        if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
            np.random.seed(SEED)
            rand_pred_probs = np.random.rand(len(X_test_OHE))
            predictions = selected_model.predict_proba(X_test_OHE)
            pred_probs_selected_model = predictions[:, 1]
        elif isinstance(selected_model, (cb.CatBoostClassifier, lgb.LGBMClassifier)):
            np.random.seed(SEED)
            rand_pred_probs = np.random.rand(len(X_test))
            predictions = selected_model.predict_proba(X_test)
            pred_probs_selected_model = predictions[:, 1]

        # pred_probs_selected_model = selected_model.predict_proba(X_test)[:, 1] 
        pred_probs_alternative_model = lr.predict_proba(X_test_OHE)[:, 1]

        decision_curve_analysis(pred_probs_selected_model=pred_probs_selected_model,
                                pred_probs_alternative_model= pred_probs_alternative_model,
                                rand_pred_probs=rand_pred_probs)

# %% [markdown]
# #### Model calibration and conformal predictions (optional)
# 
# Here we applied isotonic regression as the model calibration method. Isotonic regression is a non-parametric approach used to calibrate the predicted probabilities of a classifier. Note that the calibration should be preferrebly done based on an unseen dataset (not the dataset the model is already trained).
# 
# The following steps are followed: 
# 
# 1) Test Set Split:
# 
# We split the test set into a calibration set (X_calibration, y_calibration) and a new test set (X_new_test, y_new_test). The calibration set is used to compute the nonconformity scores for Conformal Prediction.
# 
# 2) Isotonic Regression:
# 
# We calibrate the predicted probabilities using Isotonic Regression to make the predicted probabilities more reliable.
# 
# 3) Conformal Prediction:
# 
# To understand conformal prediction you can refer to Shafer and Vovk, 2008. Below is the steps performed in the following code:
# 
# **conformal prediction** for binary classification is based on a split-conformal approach. The goal is to provide prediction sets for each test instance, ensuring 95% coverage (i.e., that the true label is included in the prediction set for approximately 95% of instances). 
# 
# **Non-conformity Scores**: These scores are calculated for the calibration set based on the predicted probabilities for the true class: \( s_i = 1 - p_i \), where \( p_i \) is the predicted probability for the true class.
# 
# **Threshold Calculation**: The 95th percentile of the non-conformity scores from the calibration set is used to determine the threshold for prediction sets.
# 
# **Prediction Sets**: For each test instance, the non-conformity scores for both classes (class 0 and class 1) are compared to the threshold. The class(es) whose non-conformity scores fall below the threshold are included in the prediction set.
# 
# **Coverage and Metrics**: The coverage, or proportion of test instances where the true label is in the prediction set, is reported. Additional metrics like Brier Score, MCC, and ROC AUC are also evaluated for confident predictions.
# 
# Coverage is the proportion of test instances for which the true label is included in the prediction set. In this analysis, coverage was **calculated as the fraction of confident predictions** made by the model:
# 
# The percentage of confident predictions was calculated as the fraction of predictions where the model was able to predict a single class with confidence.
# 
# 4) Filtering Confident Predictions:
# 
# We filter out the predictions where the p-value is less than alpha (indicating less confidence).
# Only single-class prediction sets are retained, which means the model is confident enough to assign a label with a clear margin.
# 
# 5) Evaluation:
# 
# Various metrics like Brier Score, Matthews Correlation Coefficient (MCC), ROC AUC, and PR AUC are computed for the confident predictions only.
# We also report the percentage of confident predictions, giving insight into how often the model is making confident predictions.
# 
# 
# 

# %%
if not skip_block:
    if calibration_and_conformal_predictions:

        # Step 1: Split the test data into calibration and new test sets
        if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
            X_calibration, X_new_test, y_calibration, y_new_test = train_test_split(
                X_test_OHE, y_test, test_size=0.5, random_state=SEED, stratify=y_test
            )
        else:
            X_calibration, X_new_test, y_calibration, y_new_test = train_test_split(
                X_test, y_test, test_size=0.5, random_state=SEED, stratify=y_test
            )

        # Convert y_calibration and y_new_test from True/False to 1/0
        y_calibration = y_calibration.astype(int)
        y_new_test = y_new_test.astype(int)

        # Step 2: Get predicted probabilities for the calibration set
        if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
            calibration_probs = selected_model.predict_proba(X_calibration)[:, 1]
        elif isinstance(selected_model, (cb.CatBoostClassifier, lgb.LGBMClassifier)):
            calibration_probs = selected_model.predict_proba(X_calibration)[:, 1]

        # Step 3: Compute nonconformity scores using the calibration set
        # Nonconformity scores for class 1: (1 - probability), for class 0: probability
        nonconformity_scores = np.where(y_calibration == 1, 1 - calibration_probs, calibration_probs)

        # Step 4: Get predicted probabilities for the new test set
        if isinstance(selected_model, (HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, GaussianNB)):
            test_probs = selected_model.predict_proba(X_new_test)[:, 1]
        elif isinstance(selected_model, (cb.CatBoostClassifier, lgb.LGBMClassifier)):
            test_probs = selected_model.predict_proba(X_new_test)[:, 1]

        # Step 5: Define the p-value calculation for conformal prediction
        def conformal_p_value(prob, calibration_scores, true_label):
            """
            Calculates the p-value for conformal prediction using the calibration scores.
            
            This function estimates the proportion of calibration nonconformity scores that are greater than or equal to 
            the test sample's nonconformity score. The p-value represents the probability that the null hypothesis of 
            no difference between the distribution of labels and the predicted probabilities is rejected.
            
            ## Parameters
                prob (float): Predicted probability for the test sample.
                calibration_scores (array-like): Calibration scores from a conformal prediction model.
                true_label (int): True label of the test sample.

            ## Returns
                float: The p-value, representing the proportion of calibration nonconformity scores greater than or equal to 
                    the test sample's nonconformity score.
            """
            # Nonconformity score for the test sample
            nonconformity = 1 - prob if true_label == 1 else prob
            # p-value is the proportion of calibration nonconformity scores that are greater than or equal to the test nonconformity
            p_value = np.mean(calibration_scores >= nonconformity)
            return p_value

        # Step 6: Conformal prediction for binary classification
        alpha = 0.05  # 95% coverage level
        conformal_predictions = []
        filtered_indices = []

        # Calculate conformal p-values and make predictions
        for i in range(len(X_new_test)):
            p_value_class_0 = conformal_p_value(test_probs[i], nonconformity_scores, true_label=0)
            p_value_class_1 = conformal_p_value(test_probs[i], nonconformity_scores, true_label=1)

            # Make confident prediction if one p-value is >= alpha and the other is < alpha
            if p_value_class_1 >= alpha and p_value_class_0 < alpha:
                conformal_predictions.append(1)
                filtered_indices.append(i)
            elif p_value_class_0 >= alpha and p_value_class_1 < alpha:
                conformal_predictions.append(0)
                filtered_indices.append(i)
                
        # Open a file for writing
        with open('prediction_sets.txt', 'w') as file:
            # Write the prediction sets to the file
            for i in range(len(X_new_test)):
                p_value_class_0 = conformal_p_value(test_probs[i], nonconformity_scores, true_label=0)
                p_value_class_1 = conformal_p_value(test_probs[i], nonconformity_scores, true_label=1)
                
                prediction_set = []
                if p_value_class_0 >= alpha:
                    prediction_set.append(0)  # Add class 0 to the prediction set
                if p_value_class_1 >= alpha:
                    prediction_set.append(1)  # Add class 1 to the prediction set
                
                # Write to file instead of printing
                file.write(f"Test sample {X_new_test.index[i]}: Prediction set = {prediction_set}\n")

        # Step 7: Filter the test set and report results
        y_confident = np.array(y_new_test)[filtered_indices]
        confident_predictions = np.array(conformal_predictions)

        # Evaluate Brier Score, MCC, ROC AUC, and PR AUC for both calibrated and uncalibrated probabilities
        if len(confident_predictions) > 0:
            brier_score_uncalibrated = brier_score_loss(y_new_test, test_probs)
            brier_score_calibrated = brier_score_loss(y_confident, confident_predictions)
            print("Brier Score (Uncalibrated):", brier_score_uncalibrated)
            print("Brier Score (Calibrated for confident predictions):", brier_score_calibrated)

            mcc_uncalibrated = matthews_corrcoef(y_new_test, (test_probs > 0.5).astype(int))
            mcc_calibrated = matthews_corrcoef(y_confident, confident_predictions)
            print("MCC (Uncalibrated):", mcc_uncalibrated)
            print("MCC (Calibrated for confident predictions):", mcc_calibrated)

            roc_auc_uncalibrated = roc_auc_score(y_new_test, test_probs)
            roc_auc_calibrated = roc_auc_score(y_confident, confident_predictions)
            print("ROC AUC (Uncalibrated):", roc_auc_uncalibrated)
            print("ROC AUC (Calibrated for confident predictions):", roc_auc_calibrated)

            pr_auc_uncalibrated = average_precision_score(y_new_test, test_probs)
            pr_auc_calibrated = average_precision_score(y_confident, confident_predictions)
            print("PR AUC (Uncalibrated):", pr_auc_uncalibrated)
            print("PR AUC (Calibrated for confident predictions):", pr_auc_calibrated)

        # Report the percentage of confident predictions
        confident_ratio = len(confident_predictions) / len(y_new_test)
        print(f"Percentage of confident predictions (coverage): {confident_ratio * 100:.2f}%")


# %% [markdown]
# #### Export the selected model to deploy
# The best performing model is exported (saved) on disk to be deployed.

# %%
if not skip_block:
    # Export the model
    joblib.dump(selected_model, 'selected_model.pkl')

    # Load the model
    loaded_model = joblib.load('selected_model.pkl')

# %% [markdown]
# ## Survival models
# 
# This part of the pipeline is intended to be used in case the data contains a column for time-to-event information as a survival outcome variable. If so it is possible to use the following code chunks to develop a random survival forest model and a Cox proportional hazard model and compare their performance in prediction performance. For survival models we use scikit-survival package and you can read about here: https://scikit-survival.readthedocs.io/en/stable/#
# This part may require minimal modifications according to the names used for the target column and whether Cox model outperforms the survival random forest model. By default, the assumption is that the outcome variable requires formatting as follows and the random survival forest outperforms its linear alternative that is the Cox model. It is of course possible to include more models from scikit-survival package, however it is expected that the random survival model to have similar performance to its alternative ensemble models.
# 
# It should be noted that the survival models can work with one-hot encoded data with no missingness. So X_train_OHE and X_test_OHE are suitable for the analyses. Another thing to note is that the time-to-event column is not in X_train_OHE and X_test_OHE and so we get that column from the copy of the dataset that was initially made in the beginning of the pipeline as a back up to extract that information. In the following chunk you can see the column is called "max_time_difference_days", and so if that is different in your dataset, you should modify it.

# %%
if survival_analysis:
    # X_train_OHE
    X_train_surv = pd.merge(X_train_OHE_nocv, mydata_copy_survival[time_to_event_column], left_index=True, right_index=True, how='left')
    X_test_surv = pd.merge(X_test_OHE, mydata_copy_survival[time_to_event_column], left_index=True, right_index=True, how='left')

    y_train_surv = X_train_surv[time_to_event_column]
    y_test_surv = X_test_surv[time_to_event_column]
    X_train_surv.drop(columns=[time_to_event_column], inplace=True)
    X_test_surv.drop(columns=[time_to_event_column], inplace=True)
    
    
    if external_val:
        X_extval_surv = pd.merge(X_extval_data_OHE, extval_data_survival[time_to_event_column], left_index=True, right_index=True, how='left')
        y_extval_surv = X_extval_surv[time_to_event_column]
        X_extval_surv.drop(columns=[time_to_event_column], inplace=True)
        
    

# %%
if survival_analysis:
    contains_nan = np.isnan(y_train_surv.values).sum()
    contains_nan = np.isnan(y_test_surv.values).sum()

    # Check for NaN values
    nan_indices = np.isnan(y_train_surv.values)

    # Replace NaN values with 0
    y_train_surv[nan_indices] = 0

    # Check for NaN values
    nan_indices = np.isnan(y_test_surv.values)

    # Replace NaN values with 0
    y_test_surv[nan_indices] = 0


# %%
if survival_analysis:

    y_train_surv_transformed = Surv().from_arrays(y_train.values, y_train_surv.values)
    y_test_surv_transformed = Surv().from_arrays(y_test.values, y_test_surv.values)
    
    if external_val:
        y_extval_surv_transformed = Surv().from_arrays(y_extval_data.values, y_extval_surv.values)

# %% [markdown]
# This is how the outcome column has to be formatted for survival models. In each array the first entry determines if there is any event or not and the second entry determines the last follow up time within a specific observation period. For example, when there is an event (e.g. daignosed disease) the first entry becomes True and the second entry show when it was recorded with respect to the baseline time (e.g. time of transplantation). If there was no event, then the last recorded sample of a patient is considered for the time and the event entry is False that clarifies that there was no event up to that time.

# %%
if survival_analysis:
    print("10 samples from the test set:")
    print(y_train_surv_transformed[:10])

# %% [markdown]
# ### Training and evaluation of the survival models
# 
# First we do corss validation using the traing set (development set) to assess the prediction performance of RSF and CPH models. The cross validation follows the same folding setting (i.e., number of folds) of the binary classification models (except for the survival models it is not stratified by the biary outcome variable). After we do the assessment of the models based on cross validation, we train the models on the whole trainig set and evaluate them on the test set. Two metrics are used to evaluate the models: (1) concordance index (CI), and (2) Integrated Brier Score (IBS). These scores are explained here: https://scikit-survival.readthedocs.io/en/v0.23.0/api/metrics.html.
# 
# #### Concordance Index (CI) and Integrated Brier Score (IBS)
# 
# ##### Concordance Index (CI)
# The **Concordance Index (CI)** is a performance measure for survival models. It evaluates how well the model can correctly rank survival times. The CI measures the proportion of all usable pairs of individuals where the model correctly predicts the order of survival times. A CI of `1.0` indicates perfect predictions, while `0.5` represents random guessing.
# 
# - **Interpretation**: 
#   - **CI = 1**: Perfect prediction, the model correctly ranks all pairs of individuals.
#   - **CI = 0.5**: Random prediction, no better than chance.
#   - **CI < 0.5**: Worse than random, model is predicting the reverse order of survival times.
# 
# For more details: [Concordance Index in scikit-survival](https://scikit-survival.readthedocs.io/en/v0.23.0/api/generated/sksurv.metrics.concordance_index_censored.html#sksurv.metrics.concordance_index_censored).
# 
# ##### Integrated Brier Score (IBS)
# The **Integrated Brier Score (IBS)** is a measure of the accuracy of predicted survival probabilities over time. It is the average Brier score, which measures the difference between the predicted survival probability and the actual outcome (whether the event occurred or not), across a range of time points. A lower IBS indicates better performance.
# 
# - **Interpretation**:
#   - **IBS = 0**: Perfect prediction, the model’s predicted probabilities match the true outcomes.
#   - **Higher IBS values**: Less accurate predictions.
# 
# For more details: [Integrated Brier Score in scikit-survival](https://scikit-survival.readthedocs.io/en/v0.23.0/api/generated/sksurv.metrics.integrated_brier_score.html#sksurv.metrics.integrated_brier_score).
# 

# %% [markdown]
# ### K-fold cross validation of survival models

# %%
if survival_analysis:
    n_rows = X_train_surv.shape[0]
    #########
    def adjust_hyperparameters_surv_models(n_rows):
        """
        Returns a dictionary of hyperparameter distributions for various survival models.

        ## Parameters
            n_rows (int): The number of rows in the dataset.

        ## Returns
            dict: A dictionary containing the following keys:
                - 'RSF_param_dist': Hyperparameters for Random Forest Classifier.
                - 'CoxPH_param_dist': Hyperparameters for Cox Proportional Hazards Model.
                - 'Coxnet_param_dist': Hyperparameters for CoxnetSurvivalAnalysis.

        """
        RSF_param_dist = {
            'n_estimators': np.linspace(50, max(1000, int(np.sqrt(n_rows))), num=10, dtype=int),  # Number of trees in the forest
            'min_samples_split': np.linspace(2, 20, num=10, dtype=int),  # Minimum number of samples required to split an internal node
            'min_samples_leaf': np.linspace(1, 20, num=10, dtype=int)  # Minimum number of samples required to be at a leaf node
        }

        # Define the parameter grid for CPH model
        CoxPH_param_dist = {
            'alpha': [1e-4, 0.01, 0.1, 1.0],  # Regularization parameter for ridge regression penalty
            'ties': ['breslow', 'efron'],  # Method to handle tied event times
            'n_iter': np.linspace(50, max(1000, int(np.sqrt(n_rows))), num=10, dtype=int),  # Maximum number of iterations
            'tol': [1e-6, 1e-7, 1e-8],  # Convergence criteria
            }
        # Define the parameter grid for CoxnetSurvivalAnalysis
        Coxnet_param_dist = {
            # 'alphas': [0.001, 0.01, 0.1, 0.5],  # Range of alpha values for elastic net regularization
            'l1_ratio': [0.1, 0.5, 0.9],  # Mix between L1 and L2 regularization
            'max_iter': np.linspace(50, max(1000, int(np.sqrt(n_rows))), num=10, dtype=int),  # Maximum number of iterations
            'tol': [1e-6, 1e-7, 1e-8],  # Convergence tolerance
        }

        return RSF_param_dist, CoxPH_param_dist, Coxnet_param_dist
    def set_params_surv_models(n_rows):
        """
        Sets the parameters for different survival models based on the given dataset characteristics.

        Parameters:
            n_rows (int): The number of rows in the dataset.

        ## Returns
            A dictionary containing the parameters for each survival model, including:
                - Random Survival Forest (RSF)
                - Cox proportional hazards regression (CoxPH)
                - Cox proportional hazards regression with L1 regularization (Coxnet)

        Note that these functions assume hyperparameter tuning is not done and set default values based on common practices in survival analysis.
        """
        RSF_params = {
            'n_estimators': max(1000, int(np.sqrt(n_rows))),  # Number of trees in the forest
            'min_samples_split': 2,  # Minimum number of samples required to split an internal node
            'min_samples_leaf': 1  # Minimum number of samples required to be at a leaf node
        }
        CoxPH_param = {
            'alpha': 1e-4,  # Regularization parameter for ridge regression penalty
            'ties': 'efron',  # Method to handle tied event times
            'n_iter': max(1000, int(np.sqrt(n_rows))),  # Maximum number of iterations
            'tol': 1e-6,  # Convergence criteria
            }
        Coxnet_param = {
            # 'alphas': [0.01],  # Regularization parameter for elastic net (similar to Ridge regression)
            'l1_ratio': 0.5,  # Mix between L1 and L2 regularization (0.5 means equal mix of L1 and L2)
            'max_iter': max(1000, int(np.sqrt(n_rows))),  # Maximum number of iterations
            'tol': 1e-6,  # Convergence tolerance
            "fit_baseline_model": True
        }

        return RSF_params, CoxPH_param, Coxnet_param
    #########
    def calculate_surv_metrics(y_true, model, X_test, survival_train, survival_test):
        """
        Calculate and compute key metrics for survival prediction models.

        This function takes in several inputs:
            - y_true: The true event labels (0 or 1) and corresponding times.
            - model: A trained survival prediction model.
            - X_test: Test feature data used to evaluate the model's performance.
            - survival_train: Training dataset containing event indicators, times, and predicted survival functions.
            - survival_test: Testing dataset containing event indicators, times, and predicted survival functions.

        The function computes the following metrics:
            - Concordance Index (CI): A measure of model accuracy that ranks predictions based on their performance.
            - Integrated Brier Score (IBS): An alternative measure of predictive performance that takes into account both false positives and false negatives.

        Parameters:
            y_true (pandas DataFrame): True event labels and corresponding times.
            model (scikit-learn estimator): Trained survival prediction model.
            X_test (numpy array or pandas DataFrame): Test feature data used to evaluate the model's performance.
            survival_train (pandas DataFrame): Training dataset containing event indicators, times, and predicted survival functions.
            survival_test (pandas DataFrame): Testing dataset containing event indicators, times, and predicted survival functions.

        ## Returns
            dict: A dictionary containing the Concordance Index (CI) and Integrated Brier Score (IBS).

        Raises:
            ValueError: If predictions array is empty or invalid.
        """
        event_indicator = y_true['event']
        event_time = y_true['time']
        
        # Get the concordance index (CI)
        CI = concordance_index_censored(event_indicator, event_time, model.predict(X_test))[0]

        # Extract event times from survival_train and survival_test
        train_times = survival_train['time']  # Training event times
        test_times = survival_test['time']    # Test event times
        
        # Obtain predicted survival functions for each test instance
        survival_preds = model.predict_survival_function(X_test)

        # Get the follow-up time interval from both training and test datasets
        min_followup_time = max(train_times.min(), test_times.min())  # The later start
        max_followup_time = min(train_times.max(), test_times.max())  # The earlier end

        # Define valid times based on the overlap of the follow-up times from both datasets
        valid_times = np.arange(min_followup_time, max_followup_time)

        # Ensure valid_times does not include the maximum observed time
        valid_times = valid_times[valid_times < max_followup_time]
        
        # Generate predictions for each survival function at the valid time points
        preds = np.asarray([[fn(t) for t in valid_times] for fn in survival_preds])

        # Check for empty or invalid predictions before calculating IBS
        if preds.size == 0:
            raise ValueError("Predictions array is empty. Check the time points and survival functions.")

        # Replace NaN and infinity values in preds with finite numbers
        preds = np.nan_to_num(preds, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

        # Calculate Integrated Brier Score (IBS) ensuring times are within the valid follow-up period
        IBS = integrated_brier_score(survival_train=survival_train, survival_test=survival_test, estimate=preds, times=valid_times)
        
        return {'CI': CI, 'IBS': IBS}


    #########
    # Function to cross-validate the survival model
    def cross_validate_surv_model(model_class, X, y, n_splits=cv_folds, random_state=SEED, measures=None, hp_tuning=False):
        '''
        This function performs cross-validation on a specified survival model class, evaluating its performance on a dataset. It allows users to tune hyperparameters using random search or grid search.

        ### Function Signature

        `cross_validate_surv_model(model_class: str, X: pd.DataFrame, y: np.ndarray, n_splits: int = cv_folds, random_state: int = SEED, measures: list[str] = ['CI', 'IBS'], hp_tuning: bool = False) -> pd.DataFrame`

        ### Parameters

        - `model_class`: The class of the survival model to be used for cross-validation. Supported models include `RandomSurvivalForest`, `CoxPHSurvivalAnalysis`, and `CoxnetSurvivalAnalysis`.
        - `X`: The feature data.
        - `y`: The event data, where each row contains an 'event' and a 'time'.
        - `n_splits`: The number of folds for cross-validation. Defaults to `cv_folds` if not specified.
        - `random_state`: The random seed used for shuffling and splitting the data. Defaults to `SEED` if not specified.
        - `measures`: A list of metrics to be evaluated during cross-validation. Defaults to ['CI', 'IBS'] if not specified.
        - `hp_tuning`: Whether to perform hyperparameter tuning using random search or grid search.

        ### Function Behavior

        1. The function performs KFold cross-validation on the dataset, creating a new DataFrame for each fold to store the results of the model evaluation.
        2. For each fold, it converts the data back to its original structured format and prepares the survival data for calculating metrics.
        3. If hyperparameter tuning is enabled, it uses RandomizedSearchCV to perform random search or grid search on the specified model class and parameter distribution.
        4. It then fits the model with the trained parameters and predicts the event times for the test set.
        5. After obtaining the predicted event times, it calculates the evaluation metrics using `calculate_surv_metrics`.
        6. The function aggregates the results from each fold by calculating the mean and standard deviation of each metric.
        7. Finally, it displays the aggregated results in a table format.

        ### Returns

        The function returns a DataFrame containing the aggregated results of the model evaluations for each metric specified in the `measures` list.
        '''
        if measures is None:
            measures = ['CI', 'IBS']
        
        fold_results = pd.DataFrame()
        
        # Convert survival data to DataFrame for indexing purposes
        y_df = pd.DataFrame(y, columns=['event', 'time'])
        
        # Use KFold instead of StratifiedKFold (since this is survival data, not classification)
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

        for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold_df, y_test_fold_df = y_df.iloc[train_index], y_df.iloc[test_index]
            
            # Convert y_train_fold and y_test_fold back to original structured format
            y_train_fold = np.array([(row['event'], row['time']) for _, row in y_train_fold_df.iterrows()],
                                    dtype=[('event', '?'), ('time', '<f8')])
            y_test_fold = np.array([(row['event'], row['time']) for _, row in y_test_fold_df.iterrows()],
                                dtype=[('event', '?'), ('time', '<f8')])
            
            n_rows = X_train_fold.shape[0]
            # Adjust hyperparameters based on the training data in this fold
            RSF_param_dist, CoxPH_param_dist, Coxnet_param_dist = adjust_hyperparameters_surv_models(n_rows)
            RSF_params, CoxPH_params, Coxnet_params = set_params_surv_models(n_rows)
            # Prepare survival data for calculating metrics
            survival_train = Surv.from_dataframe('event', 'time', y_train_fold_df)
            survival_test = Surv.from_dataframe('event', 'time', y_test_fold_df)
            
            # Initialize and fit the model
            if model_class == RandomSurvivalForest:
                RSF_model = RandomSurvivalForest(random_state=random_state, **RSF_params)
                if hp_tuning:
                    random_search = RandomizedSearchCV(
                        estimator=RSF_model, 
                        param_distributions=RSF_param_dist, 
                        n_iter=n_iter_hptuning, 
                        cv=cv_folds_hptuning, 
                        refit=True, 
                        random_state=random_state)
                    random_search.fit(X_train_fold, y_train_fold)
                    # RSF_model = random_search.best_estimator_
                    RSF_model = RandomSurvivalForest(**random_search.best_params_)
                
                RSF_model.fit(X_train_fold, y_train_fold)
                y_pred = RSF_model.predict(X_test_fold)
                
                # Calculate evaluation metrics
                metrics = calculate_surv_metrics(y_true=y_test_fold_df, model=RSF_model, X_test=X_test_fold, survival_train=survival_train, survival_test=survival_test)
                fold_results = fold_results.append(metrics, ignore_index=True)
                
            elif model_class == CoxPHSurvivalAnalysis:
                CPH_model = CoxPHSurvivalAnalysis(**CoxPH_params)
                if hp_tuning:
                    random_search = RandomizedSearchCV(
                        estimator=CPH_model, 
                        param_distributions=CoxPH_param_dist, 
                        n_iter=n_iter_hptuning, 
                        cv=cv_folds_hptuning, 
                        refit=True, 
                        random_state=random_state)
                    random_search.fit(X_train_fold, y_train_fold)
                    # CPH_model = random_search.best_estimator_
                    CPH_model = CoxPHSurvivalAnalysis(**random_search.best_params_)
                    
                CPH_model.fit(X_train_fold, y_train_fold)
                y_pred = CPH_model.predict(X_test_fold)
                # Calculate evaluation metrics
                metrics = calculate_surv_metrics(y_true=y_test_fold_df, model=CPH_model, X_test=X_test_fold, survival_train=survival_train, survival_test=survival_test)
                fold_results = fold_results.append(metrics, ignore_index=True)
            elif model_class == CoxnetSurvivalAnalysis:
                Coxnet_model = CoxnetSurvivalAnalysis(**Coxnet_params)
                
                if hp_tuning:
                    random_search = RandomizedSearchCV(
                        estimator=Coxnet_model, 
                        param_distributions=Coxnet_param_dist, 
                        n_iter=n_iter_hptuning, 
                        cv=cv_folds_hptuning, 
                        refit=True, 
                        random_state=random_state
                    )
                    random_search.fit(X_train_fold, y_train_fold)
                    # Coxnet_model = random_search.best_estimator_
                    Coxnet_model = CoxnetSurvivalAnalysis(fit_baseline_model=True, **random_search.best_params_)
                
                Coxnet_model.fit(X_train_fold, y_train_fold)
                y_pred = Coxnet_model.predict(X_test_fold)
                
                # Calculate evaluation metrics
                metrics = calculate_surv_metrics(
                    y_true=y_test_fold_df, 
                    model=Coxnet_model, 
                    X_test=X_test_fold, 
                    survival_train=survival_train, 
                    survival_test=survival_test
                )
                fold_results = fold_results.append(metrics, ignore_index=True)

            else:
                raise ValueError(f"Unsupported model class: {model_class}")
            

        # Aggregating the results
        aggregated_results = {metric: np.nanmean(fold_results[metric].values).round(2) for metric in measures}
        aggregated_results_sd = {metric: np.nanstd(fold_results[metric].values).round(2) for metric in measures}
        
        # Combine mean and standard deviation
        combined_results = {metric: f"{mean} ± {sd}" for metric, mean in aggregated_results.items() for _, sd in aggregated_results_sd.items() if metric == _}
        
        # Create a DataFrame for displaying the results
        results_table = pd.DataFrame(list(combined_results.items()), columns=['Metric', 'Result'])
        
        
        print("Aggregated Results:")
        print(results_table.to_string(index=False))
        
        return results_table


# %%
def integrated_brier_score(survival_train, survival_test, estimate, times):
    """
    Compute the Integrated Brier Score (IBS) using scikit-learn's Brier score loss.
    
    Parameters
    ----------
    survival_train : structured array
        Survival times for training data (event indicator, time).
    survival_test : structured array
        Survival times for test data (event indicator, time).
    estimate : array-like, shape = (n_samples, n_times)
        Estimated survival probabilities.
    times : array-like, shape = (n_times,)
        Time points for the calculation.
    
    Returns
    -------
    ibs : float
        The Integrated Brier Score (IBS).
    """
    # Initialize list to hold Brier scores at each time point
    brier_scores = []

    event_indicator = survival_test['event']
    time_of_event_or_censoring = survival_test['time']

    # Iterate through all the times to calculate the Brier score at each time point
    for i, t in enumerate(times):
        # Calculate the true survival status at time t
        y_true = (time_of_event_or_censoring > t).astype(int)
        
        # Calculate the predicted survival probabilities at time t
        pred_prob = estimate[:, i]  # Use the i-th column directly
        
        # Identify samples that are not censored before time t
        mask = (time_of_event_or_censoring >= t) | event_indicator
        
        # Brier score is calculated based on the true survival status and the predicted probability
        brier_score = brier_score_loss(y_true[mask], pred_prob[mask])
        brier_scores.append(brier_score)

    # Convert to numpy array for easier processing
    brier_scores = np.array(brier_scores)
    
    # Compute the Integrated Brier Score using the trapezoidal rule
    if len(times) < 2:
        raise ValueError("At least two time points must be provided")

    # Apply the trapezoidal rule to integrate the Brier scores
    ibs_value = np.trapz(brier_scores, times) / (times[-1] - times[0])

    return ibs_value

# %%
if survival_analysis:
    # getting the cross validation results (performance) for RSF - mean and standard deviation of the metrics
    results_table_RSF = cross_validate_surv_model(
        model_class=RandomSurvivalForest,
        X=X_train_surv, 
        y=y_train_surv_transformed,
        n_splits=cv_folds,
        random_state=SEED
    )


# %%
from sksurv.linear_model import CoxnetSurvivalAnalysis
if survival_analysis:
    # getting the cross validation results (performance) for CPH - mean and standard deviation of the metrics
    results_table_CPH = cross_validate_surv_model(
        model_class=CoxnetSurvivalAnalysis,
        X=X_train_surv, 
        y=y_train_surv_transformed,
        n_splits=cv_folds,
        random_state=SEED
    )

# %%
# if survival_analysis:
#     # getting the cross validation results (performance) for CPH - mean and standard deviation of the metrics
#     results_table_CPH = cross_validate_surv_model(
#         model_class=CoxPHSurvivalAnalysis,
#         X=X_train_surv, 
#         y=y_train_surv_transformed,
#         n_splits=cv_folds,
#         random_state=SEED
#     )

# %%
if survival_analysis:
    # aggragation of the CV results for RSF and CPH models to make them into one table

    results_table_RSF.rename(columns={'Result': 'RSF'}, inplace=True)
    results_table_CPH.rename(columns={'Result': 'CPH'}, inplace=True)

    # Merging the two tables on the 'Metric' column
    merged_table = pd.merge(results_table_RSF, results_table_CPH, on='Metric')

    # Display the merged table
    print(merged_table.to_string(index=False))
    merged_table.to_excel('CV_surv.xlsx', index=False)

# %%
# now we do the survival analysis using the whole trainig set for trainig the survival models and evaluating them on the test set
if survival_analysis:
    # Define the parameter grid for RSF model
    n_rows = X_train_surv.shape[0]
    RSF_param_dist, CoxPH_param_dist, Coxnet_param_dist = adjust_hyperparameters_surv_models(n_rows)
    
    ############# RSF
    # RandomizedSearchCV for Random Survival Forest (RSF)
    rsf = RandomSurvivalForest(n_jobs=n_cpu_model_training, random_state=SEED)
    random_search = RandomizedSearchCV(rsf, param_distributions=RSF_param_dist, n_iter=n_iter_hptuning, cv=cv_folds, random_state=SEED)
    
    # Fit RandomizedSearchCV on training data
    random_search.fit(X_train_surv, y_train_surv_transformed)
    print("Best Parameters for RSF:", random_search.best_params_)
    
    # Train the RSF model with the best hyperparameters
    rsf = RandomSurvivalForest(n_jobs=n_cpu_model_training, random_state=SEED, **random_search.best_params_)
    rsf.fit(X_train_surv, y_train_surv_transformed)
    
    # Save the RSF model to disk
    dump(rsf, 'RSF_model.pkl')
    
    # Predict survival function for test data
    survival_preds_rsf = rsf.predict_survival_function(X_test_surv)

    # Extract event times from survival_train and survival_test
    train_times = y_train_surv.values  # Training event times
    test_times = y_test_surv.values    # Test event times

    # Get the follow-up time interval from both training and test datasets
    min_followup_time = max(train_times.min(), test_times.min())  # The later start
    max_followup_time = min(train_times.max(), test_times.max())  # The earlier end

    # Define valid times based on the overlap of the follow-up times from both datasets
    valid_times = np.arange(min_followup_time, max_followup_time)

    # Ensure valid_times does not include the maximum observed time
    valid_times = valid_times[valid_times < max_followup_time]

    # Generate predictions for each survival function at the valid time points
    preds = np.asarray([[fn(t) for t in valid_times] for fn in survival_preds_rsf])

    # Replace NaN and infinity values in preds with finite numbers
    estimate_rsf = np.nan_to_num(preds, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    # Calculate IBS for RSF
    IBS_rsf = integrated_brier_score(survival_train=y_train_surv_transformed, survival_test=y_test_surv_transformed, estimate=estimate_rsf, times=valid_times)
    print(f"Integrated Brier Score for RSF on the test set: {IBS_rsf:.2f}")
    
    # Get the concordance index (CI)
    CI_rsf = concordance_index_censored(event_indicator= y_test, event_time = y_test_surv.values, estimate = rsf.predict(X_test_surv))[0]
    print(f"C-index for RSF on the test set: {CI_rsf:.2f}")
    
    ############## CPH
    # RandomizedSearchCV for Cox Proportional Hazards (CPH)
    # coxph = CoxPHSurvivalAnalysis()
    coxph = CoxnetSurvivalAnalysis(fit_baseline_model=True)
    random_search_coxph = RandomizedSearchCV(coxph, param_distributions=Coxnet_param_dist, n_iter=n_iter_hptuning, cv=cv_folds, random_state=SEED)

    # random_search_coxph = RandomizedSearchCV(coxph, param_distributions=CoxPH_param_dist, n_iter=n_iter_hptuning, cv=cv_folds, random_state=SEED)
    
    # Fit RandomizedSearchCV on training data
    random_search_coxph.fit(X_train_surv, y_train_surv_transformed)
    print("Best Parameters for CPH:", random_search_coxph.best_params_)

    # Train the CPH model with the best hyperparameters
    coxph = CoxnetSurvivalAnalysis(fit_baseline_model=True, **random_search_coxph.best_params_)
    # coxph = CoxPHSurvivalAnalysis(**random_search_coxph.best_params_)
    coxph.fit(X_train_surv, y_train_surv_transformed)
    
    # Save the CPH model to disk
    dump(coxph, 'CPH_model.pkl')

    # Calculate Integrated Brier Score (IBS) for CPH
    survival_preds_cph = coxph.predict_survival_function(X_test_surv)
    
    # Generate predictions for each survival function at the valid time points
    preds = np.asarray([[fn(t) for t in valid_times] for fn in survival_preds_cph])

    # Replace NaN and infinity values in preds with finite numbers
    estimate_cph = np.nan_to_num(preds, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    # Calculate IBS for CPH
    IBS_cph = integrated_brier_score(survival_train=y_train_surv_transformed, survival_test=y_test_surv_transformed, estimate=estimate_cph, times=valid_times)
    print(f"Integrated Brier Score for CPH on the test set: {IBS_cph:.2f}")
    
    # Get the concordance index (CI)
    CI_cph = concordance_index_censored(event_indicator= y_test, event_time = y_test_surv.values, estimate = coxph.predict(X_test_surv))[0]
    print(f"C-index for CPH on the test set: {CI_cph:.2f}")

# %%
if survival_analysis and external_val:
    # RSF model evaluation on the external validation set
    # Predict survival function for test data
    survival_preds_rsf = rsf.predict_survival_function(X_extval_surv)

    # Extract event times from survival_train and survival_test
    train_times = y_train_surv.values  # Training event times
    test_times = y_extval_surv.values    # Test event times

    # Get the follow-up time interval from both training and test datasets
    min_followup_time = max(train_times.min(), test_times.min())  # The later start
    max_followup_time = min(train_times.max(), test_times.max())  # The earlier end

    # Define valid times based on the overlap of the follow-up times from both datasets
    valid_times = np.arange(min_followup_time, max_followup_time)

    # Ensure valid_times does not include the maximum observed time
    valid_times = valid_times[valid_times < max_followup_time]

    # Generate predictions for each survival function at the valid time points
    preds = np.asarray([[fn(t) for t in valid_times] for fn in survival_preds_rsf])

    # Replace NaN and infinity values in preds with finite numbers
    estimate_rsf = np.nan_to_num(preds, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    # # Define a time range for the Brier score calculation based on the test data
    # extval_times_rsf = y_extval_surv.values
    # valid_times_rsf = np.linspace(extval_times_rsf.min(), extval_times_rsf.max() - 1e-5, 100)
    
    # # Create an estimate array for the predicted survival probabilities at each valid time point
    # estimate_rsf = np.zeros((len(X_extval_surv), len(valid_times_rsf)))
    # for i, fn in enumerate(survival_preds_rsf):
    #     estimate_rsf[i, :] = fn(valid_times_rsf)

    # Calculate IBS for RSF
    IBS_rsf = integrated_brier_score(survival_train=y_train_surv_transformed, survival_test=y_extval_surv_transformed, estimate=estimate_rsf, times=valid_times)
    print(f"Integrated Brier Score for RSF on the external validation set: {IBS_rsf:.2f}")
    
    # Get the concordance index (CI)
    CI_rsf = concordance_index_censored(event_indicator= y_extval_data, event_time = y_extval_surv.values, estimate = rsf.predict(X_extval_surv))[0]
    print(f"C-index for RSF on the external validation set: {CI_rsf:.2f}")

# %% [markdown]
# ### feature importance for the survival model
# For each feature it shows if its relationship to survival time is removed (by random shuffling), the concordance index on the test data drops on average by mentioned values displayed on the table.

# %%
if survival_analysis:
    # RSF model
    perm_result = permutation_importance(rsf, X_test_surv, y_test_surv_transformed, n_repeats=15, random_state=SEED)

    RSF_perm_fi = pd.DataFrame(
        {
            k: perm_result[k]
            for k in (
                "importances_mean",
                "importances_std",
            )
        },
        index=X_test_surv.columns,
    ).sort_values(by="importances_mean", ascending=False)
    
    print(RSF_perm_fi)

# %% [markdown]
# One of the methods based on SHAP that is implemented for survival ML models like RSF is to use SurvSHAP(t). It provides time-dependent explanations for the survival machine learning models. The impact of a variable over the observation period may change and that is the information that SurvSHAP(t) can reveal. SurvSHAP(t) is explained in details in https://www.sciencedirect.com/science/article/pii/S0950705122013302?via%3Dihub .

# %%
# here we can see the impact of variables over time on the predictions of the survival model for one sample (patient)
if survival_analysis:
    # adopted from https://pypi.org/project/survshap/

    # create explainer(X_train_surv, y_train_surv_transformed)
    rsf_exp = SurvivalModelExplainer(model = rsf, data = X_train_surv, y = y_train_surv_transformed)

    # compute SHAP values for a single instance
    observation_A = X_train_surv.iloc[[0]]
    survshap_A = PredictSurvSHAP(
        random_state=SEED,        # Set the random seed for reproducibility
        function_type = "chf", # Either "sf" representing survival function or "chf" representing cumulative hazard function (use chf if you want to see the direction of the feature impact aligned with binary classification models: positive SHAP equivalent to increase risk)
        calculation_method="sampling"  # "shap_kernel" for shap.KernelExplainer, "kernel" for exact KernelSHAP, "sampling" for sampling method, or "treeshap" for shap.TreeExplainer
    )
    survshap_A.fit(explainer = rsf_exp, new_observation = observation_A)

    survshap_A.result 
    survshap_A.plot()

# %%
# now we can get the survival SHAP values for a group of patients (samples) or all samples on the test set for example
if survival_analysis:
    # rsf_exp = SurvivalModelExplainer(rsf, X_train_surv, y_train_surv_transformed)

    # you can set this to smaller numbers for a subset of samples patients if it takes too long
    n_samples = X_test_surv.shape[0]

    def parallel_compute_shap_surv(data, i):
        """
        Computes the SHAP values for a single patient or sample using parallel computing.
        This function is used to compute the SHAP (SHapley Additive exPlanations) values for a single patient or sample in parallel, using the `parallel_compute_shap_surv` approach. It takes a pandas DataFrame `data` and an integer index `i` as input, and returns an object of type `PredictSurvSHAP`, which contains the computed SHAP values.

        The function uses the `Parallel` class from the `loky` library to run multiple iterations of the computation in parallel, taking advantage of multiple CPU cores for speedup. The number of CPU cores used is controlled by the `n_cpu_for_tuning` variable, which can be adjusted depending on the system's capabilities and the size of the dataset.

        The function is then applied to a range of indices using a list comprehension, and the resulting objects are collected into a list called `survshaps`. This approach allows for efficient computation of SHAP values for all samples in the test set.

        Parameters:
            data (pandas DataFrame): The input data.
            i (int): The index of the patient or sample to compute SHAP values for.

        ## Returns
            PredictSurvSHAP: An object containing the computed SHAP values.
        """
        survshap = PredictSurvSHAP(random_state = SEED, function_type = "chf", calculation_method="sampling")
        survshap.fit(rsf_exp, data.iloc[[i]])
        return survshap
    # run it in parallel to speed up the processing
    survshaps = Parallel(n_jobs=n_cpu_for_tuning, backend='loky')(delayed(parallel_compute_shap_surv)(X_test_surv, i) for i in range(n_samples))
    if external_val:
        survshaps_extval = Parallel(n_jobs=n_cpu_for_tuning, backend='loky')(delayed(parallel_compute_shap_surv)(X_extval_surv, i) for i in range(n_samples))


# %%
if survival_analysis:
    def plot_survshap_detailed(shap_results, top_n=10, sample_percentage=100):
        """
        Optimized function to plot SHAP values over time for the top N features on separate subplots,
        with an option to randomly sample a percentage of the data for each feature.

        Parameters:
        shap_results: List of SHAP results for each sample.
        top_n: The number of top features to plot based on mean of max absolute SHAP values.
        sample_percentage: Percentage of samples randomly selected to be displayed on the plots (0 < sample_percentage <= 100).
        """
        # Combine results from all samples into one DataFrame
        shap_df = pd.concat([shap.result for shap in shap_results], axis=0)
        
        # Extract time columns (assuming they start with 't = ')
        time_columns = [col for col in shap_df.columns if col.startswith('t =')]
        
        # Get unique feature names
        feature_names = shap_df['variable_name'].unique()

        # Precompute mean of max absolute SHAP values for each feature
        feature_data_dict = {}
        for feature_name in feature_names:
            feature_data = shap_df[shap_df['variable_name'] == feature_name]
            shap_values = feature_data[time_columns].values

            # Calculate the max absolute SHAP value for each sample and then compute the mean across all samples
            max_abs_shap_per_sample = np.max(np.abs(shap_values), axis=1)
            max_max_abs_shap = np.max(max_abs_shap_per_sample)
            
            feature_data_dict[feature_name] = {
                'data': feature_data,
                'shap_values': shap_values,
                'max_max_abs_shap': max_max_abs_shap
            }

        # Sort features by their max of max absolute SHAP value and take only the top N features
        sorted_features = sorted(feature_data_dict.keys(), key=lambda x: feature_data_dict[x]['max_max_abs_shap'], reverse=True)[:top_n]
        
        # Create subplots
        num_features = len(sorted_features)
        fig, axes = plt.subplots(num_features, 1, figsize=(10, 2 * num_features), sharex=True, constrained_layout=True)
        
        if num_features == 1:
            axes = [axes]  # Ensure axes is iterable if only one subplot

        # Colormap for feature values
        cmap = cm.get_cmap('coolwarm')

        # Process and plot each feature
        for idx, (ax, feature_name) in enumerate(zip(axes, sorted_features)):
            # Get data and SHAP values for the specific feature
            feature_data = feature_data_dict[feature_name]['data']
            shap_values = feature_data_dict[feature_name]['shap_values']
            feature_values = feature_data['variable_value'].values
            
            if len(shap_values) == 0:
                continue

            # Randomly select a subset of samples if sample_percentage < 100
            if sample_percentage < 100:
                num_samples = len(feature_data)
                sample_size = int(num_samples * (sample_percentage / 100))
                sampled_indices = np.random.choice(num_samples, sample_size, replace=False)
                shap_values = shap_values[sampled_indices]
                feature_values = feature_values[sampled_indices]

            # feature value normalization
            normalized_values = QuantileTransformer(output_distribution='uniform').fit_transform(feature_values.reshape(-1, 1)).flatten()
            
            # Plot SHAP values for each sample
            for i in range(len(shap_values)):
                color = cmap(normalized_values[i])
                ax.plot(time_columns, shap_values[i], color=color, alpha=0.6, lw=1)
            
            # Title and Y-axis label
            ax.set_title(f"{feature_name} (grand max |SHAP|: {feature_data_dict[feature_name]['max_max_abs_shap']:.2f})", fontsize=10)
            ax.set_ylabel("SHAP value")
            
            # Horizontal line at y=0
            ax.axhline(0, color='grey', linestyle=':', linewidth=1)
        
        # Simplify x-axis labels
        xticks_interval = max(len(time_columns) // 10, 1)  # Ensure at least some ticks show up
        plt.xticks(time_columns[::xticks_interval], rotation=90)

        # Colorbar for all subplots
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02, label='Feature Value')

        return plt

    survshaps_plt = plot_survshap_detailed(survshaps, top_n=top_n_f, sample_percentage=100) # plot top top_n_f features for 100% of samples from the test set
    survshaps_plt.savefig('shap_surv_curves_testset.tif', bbox_inches='tight')
    survshaps_plt.show()

# %%
if survival_analysis and external_val:
    survshaps_extval_plt = plot_survshap_detailed(survshaps_extval, top_n=top_n_f, sample_percentage=100) # plot top top_n_f features for 100% of samples from the test set
    survshaps_extval_plt.savefig('shap_surv_curves_extval.tif', bbox_inches='tight')
    survshaps_extval_plt.show()

# %%
# here we aggregate the shap values for each feature (variable) and display a SHAP summary plot based on the SHAP values calculated using survSHAP(t) for survival models
# What we see here is the feature values and their relationships with the SHAP values (that could not be shown on the previous plot with SHAP values over time)
if survival_analysis:

    def aggregate_shap_values_with_base(survshaps, X_test_surv, aggregation='mean_abs'):
        """
        Aggregate SHAP values from a survival model.

        ## Parameters
            survshaps (list): A list of Shapely objects containing survival model results.
            X_test_surv (pd.DataFrame): The test set used for feature selection and model evaluation.
            aggregation (str, optional): The method to use for aggregating SHAP values. Can be 'mean_abs', 'sum', or 'mean'. Defaults to 'mean_abs'.

        ## Returns
            tuple: A tuple containing two arrays:
                - shap_values (np.ndarray): An array of shape (n_samples, n_features) where each row represents the aggregated SHAP value for a sample.
                - base_values (np.ndarray): An array of shape (n_samples,) where each element is the base value for a sample.

        """
        n_features = X_test_surv.shape[1]  # Number of features in the test set 
        n_samples = X_test_surv.shape[0]   # Number of samples
        shap_values = np.zeros((n_samples, n_features))  # Initialize shap_values array
        base_values = np.zeros(n_samples)   # Initialize base_values array

        for i, survshap in enumerate(survshaps):
            shap_df = survshap.result

            # Extract columns corresponding to time points (these contain SHAP values)
            time_point_columns = [col for col in shap_df.columns if col.startswith('t =')]
            shap_values_sample = shap_df[time_point_columns].values  # shape: (n_timepoints, n_features_per_timepoint)
            
            # Group SHAP values by feature
            feature_groups = shap_df.groupby('variable_name')

            # Initialize temporary array to store aggregated SHAP values per feature for this sample
            shap_values_aggregated = np.zeros(n_features)
            
            for feature_name, group in feature_groups:
                # Aggregate SHAP values across time points for each feature
                if aggregation == 'mean_abs':
                    shap_agg = group[time_point_columns].abs().mean(axis=1)  # Aggregating by absolute mean
                elif aggregation == 'sum':
                    shap_agg = group[time_point_columns].sum(axis=1)  # Aggregating by sum
                elif aggregation == 'mean':
                    shap_agg = group[time_point_columns].mean(axis=1)  # Aggregating by mean
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation}")

                # Map the aggregated SHAP values to the correct feature in shap_values_aggregated
                feature_idx = X_test_surv.columns.get_loc(feature_name)  # Get the index of the feature
                shap_values_aggregated[feature_idx] = shap_agg.mean()  # Store the aggregated value for this feature
            
            # Store the aggregated SHAP values for this sample
            shap_values[i] = shap_values_aggregated

            # Extract base value if available
            if hasattr(survshap, 'base_value'):
                base_values[i] = survshap.base_value
            else:
                base_values[i] = 0  # Default or adjust as needed
        
        return shap_values, base_values

    # getting the shap summary plot for the survival model (RSF) on the test set
    shap_values, base_values = aggregate_shap_values_with_base(survshaps, X_test_surv, aggregation='mean')

    # Create the Explanation object with per-sample base_values
    explainer_values = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X_test_surv,
        feature_names=X_test_surv.columns
    )

    shap.plots.beeswarm(explainer_values, max_display=top_n_f)
    plt.savefig("shap_beeswarm_testset.tif", bbox_inches='tight')
    



# %%
# getting the shap summary plot for the survival model (RSF) on the external validation set
if survival_analysis and external_val:
    shap_values_extval, base_values_extval = aggregate_shap_values_with_base(survshaps_extval, X_extval_surv, aggregation='mean')

    # Create the Explanation object with per-sample base_values
    explainer_values_extval = shap.Explanation(
        values=shap_values_extval,
        base_values=base_values_extval,
        data=X_extval_surv,
        feature_names=X_extval_surv.columns
    )

    shap.plots.beeswarm(explainer_values_extval, max_display=top_n_f)
    plt.savefig("shap_beeswarm_extval.tif", bbox_inches='tight')

# %%
def f_imp_shapboxplot_surv(shap_values, X_test, num_features=20, num_subsamples=1000, random_seed=None, apply_threshold = False):
    """
    The `f_imp_shapboxplot_surv` function generates an importance plot for SHAP (SHapley Additive exPlanations) values to identify the most influential features in a model. The function takes in SHAP value arrays, test data, and various parameters to customize the plot.

    Functionality:

    1. Compute median, lower quantile, and upper quantile of absolute SHAP values for all features.
    2. Sort features by median absolute SHAP value in descending order and select the top N most important features (default N=20).
    3. Perform an IQR crossing test with subsamples to determine feature significance.
    4. Mark significant features with an asterisk based on the IQR crossing test results.
    5. Plot a boxplot of the distribution of absolute SHAP values for the selected features, using different colors for significant and non-significant features.

    ## Parameters:

    `shap_values`: array of SHAP value arrays
    `X_test`: test data
    `num_features`: number of top features to select (default=20)
    `num_subsamples`: number of subsamples for the IQR crossing test (default=1000)
    `random_seed`: random seed for reproducibility (optional)
    `apply_threshold`: apply threshold to SHAP values (default=False)

    ## Returns

    A pandas DataFrame containing the top N most important features, including their median and quantile SHAP values, subsample proportion crossing zero, and significance status.
    A matplotlib plot of the boxplot showing the distribution of absolute SHAP values for the selected features.
    """
    # Use absolute SHAP values for median and quantiles
    abs_shap_values = np.abs(shap_values)
    median_abs_shap_values = np.median(abs_shap_values, axis=0)
    lower_quantiles = np.percentile(abs_shap_values, 25, axis=0)  # 25th percentile of absolute values
    upper_quantiles = np.percentile(abs_shap_values, 75, axis=0)  # 75th percentile of absolute values

    feature_importance_df = pd.DataFrame({
        'Feature': X_test.columns.tolist(),
        'Median_SHAP': median_abs_shap_values,
        'Lower_Quantile': lower_quantiles,
        'Upper_Quantile': upper_quantiles
    })

    # Sort the features by median absolute SHAP values in descending order
    feature_importance_df = feature_importance_df.sort_values('Median_SHAP', ascending=False)
    if apply_threshold:
        # Compute the sum of SHAP values for instance
        sum_shap_values = np.sum(shap_values, axis=1)
        print(len(sum_shap_values))

        # Define threshold as the 1st percentile of the sum of SHAP values
        shap_threshold = np.percentile(np.abs(sum_shap_values), 1)
        print(shap_threshold)
    else:
        shap_threshold = 0

    # Select the top N most important features
    top_features = feature_importance_df.head(num_features)

    # Initialize lists to store subsample results
    subsample_results = []
    is_significant = []
    
    for i in top_features.index:
        feature_shap_values = np.abs(shap_values[:, i])
        # Perform the IQR crossing test with subsamples
        proportion_crossing_zero = subsample_iqr_test(feature_shap_values, num_subsamples=num_subsamples, threshold=shap_threshold, random_seed=random_seed)
        
        # A feature is significant if less than (1 - confidence_level)% of the subsamples cross zero
        significant = proportion_crossing_zero <= (1 - 0.95)
        is_significant.append(significant)
        
        subsample_results.append(proportion_crossing_zero)

    # Add the subsample results and significance to the DataFrame
    top_features['Subsample_Proportion_Crossing_Zero'] = subsample_results
    top_features['Significant'] = is_significant
    
    # Mark significant features with an asterisk
    top_features['Feature'] = top_features.apply(lambda row: row['Feature'] + ('*' if row['Significant'] else ''), axis=1)
    
    # Prepare colors based on significance: light green for significant, light red for non-significant
    colors = ['lightgreen' if sig else 'lightcoral' for sig in top_features['Significant']]

    plt.figure(figsize=(10, 5+round(np.max([10, np.log(num_features)]))))
    sns.boxplot(data=np.abs(shap_values[:, top_features.index[:num_features]]), orient='h', whis=[25, 75], 
                width=.5, flierprops={"marker": "x", "alpha": 0.5}, palette=colors, linewidth=0.9)

    # Customize the plot
    plt.yticks(np.arange(num_features), top_features['Feature'], size=8)
    plt.grid(linestyle=':', linewidth=0.5, alpha=0.5)
    plt.xlabel('Absolute SHAP value')
    plt.ylabel('Feature')
    plt.title(f'Distribution of absolute SHAP values for all available features', size=10)
    
    return top_features, plt


# %%
if survival_analysis:   
    f_imp_shap_table_testset_surv, f_imp_shapboxplot_testset_surv = f_imp_shapboxplot_surv(shap_values, X_test_surv, num_features=X_test_surv.shape[1], random_seed= SEED, apply_threshold=True)

    f_imp_shapboxplot_testset_surv.tight_layout()
    f_imp_shapboxplot_testset_surv.savefig("f_imp_shapboxplot_testset_surv.tif", bbox_inches='tight') 
    f_imp_shapboxplot_testset_surv.show()
    print(f_imp_shap_table_testset_surv)
    f_imp_shap_table_testset_surv.to_excel('f_imp_shap_table_testset_surv.xlsx', index=False)

# %%
if survival_analysis and external_val:   

    f_imp_shap_table_extval_surv, f_imp_shapboxplot_extval_surv = f_imp_shapboxplot_surv(shap_values_extval, X_extval_surv, num_features=X_extval_surv.shape[1], random_seed= SEED, apply_threshold=False)

    f_imp_shapboxplot_extval_surv.tight_layout()
    f_imp_shapboxplot_extval_surv.savefig("f_imp_shapboxplot_extval_thr0_surv.tif", bbox_inches='tight') 
    f_imp_shapboxplot_extval_surv.show()
    print(f_imp_shap_table_extval_surv)
    f_imp_shap_table_extval_surv.to_excel('f_imp_shap_table_extval_thr0_surv.xlsx', index=False)

# %%
if survival_analysis and external_val:   
    f_imp_shap_table_extval_surv, f_imp_shapboxplot_extval_surv = f_imp_shapboxplot_surv(shap_values_extval, X_extval_surv, num_features=X_extval_surv.shape[1], random_seed= SEED, apply_threshold=True)

    f_imp_shapboxplot_extval_surv.tight_layout()
    f_imp_shapboxplot_extval_surv.savefig("f_imp_shapboxplot_extval_surv.tif", bbox_inches='tight') 
    f_imp_shapboxplot_extval_surv.show()
    print(f_imp_shap_table_extval_surv)
    f_imp_shap_table_extval_surv.to_excel('f_imp_shap_table_extval_surv.xlsx', index=False)

# %% [markdown]
# ### Predicted survival and cumulative hazard 

# %%
if survival_analysis:
    # Predict cumulative hazard function for all training samples
    # The informaiton here will be used to translate the predicted hazard functions for each individuals in the test set (or external validation set) to binary classifcation
    surv_train = rsf.predict_cumulative_hazard_function(X_train_surv, return_array=True)

    class_0_indices = np.where(y_train.values == False)[0] 
    class_1_indices = np.where(y_train.values == True)[0] 

    # Separate predictions into classes
    surv_class_0 = surv_train[class_0_indices]
    surv_class_1 = surv_train[class_1_indices]

    # Calculate median and interquartile range for both classes
    median_hazard_class_0_train = np.median(surv_class_0, axis=0)
    median_hazard_class_1_train = np.median(surv_class_1, axis=0)
    
    rsf_riskscores_train = rsf.predict(X_train_surv)
    
    # Calculate average risk scores for each class in the training set
    predicted_risk_socres_class_0 = rsf_riskscores_train[y_train]
    predicted_risk_socres_class_1 = rsf_riskscores_train[~y_train]
    

# %%
if survival_analysis:
    
    # Predict cumulative hazard function for all test samples
    surv = rsf.predict_cumulative_hazard_function(X_test_surv, return_array=True)

    class_0_indices = np.where(y_test.values == False)[0] 
    class_1_indices = np.where(y_test.values == True)[0]  

    # Separate predictions into classes
    surv_class_0 = surv[class_0_indices]
    surv_class_1 = surv[class_1_indices]

    # Calculate median and interquartile range for both classes
    median_hazard_class_0_test = np.median(surv_class_0, axis=0)
    q1_hazard_class_0_test = np.percentile(surv_class_0, 25, axis=0)
    q3_hazard_class_0_test = np.percentile(surv_class_0, 75, axis=0)
    iqr_hazard_class_0_test = q3_hazard_class_0_test - q1_hazard_class_0_test

    median_hazard_class_1_test = np.median(surv_class_1, axis=0)
    q1_hazard_class_1_test = np.percentile(surv_class_1, 25, axis=0)
    q3_hazard_class_1_test = np.percentile(surv_class_1, 75, axis=0)
    iqr_hazard_class_1_test = q3_hazard_class_1_test - q1_hazard_class_1_test


# %%
if survival_analysis:

    # Define a function to calculate the Euclidean distance
    def euclidean_distance(x, y):
        """
        Calculates the Euclidean distance between two vectors.

        The Euclidean distance is a measure of the straight-line distance between two points in n-dimensional space.
        It is defined as the square root of the sum of the squared differences between corresponding elements in the input vectors.

        ## Parameters
            x (numpy array): The first vector.
            y (numpy array): The second vector.

        ## Returns
            float: The Euclidean distance between the two input vectors.

        Note:
            This function assumes that the input vectors are of equal length. If they are not, an error will be raised.
        """
        return np.sqrt(np.sum((x - y) ** 2))

    # Predict cumulative hazard function for all test samples
    surv = rsf.predict_cumulative_hazard_function(X_test_surv, return_array=True)

    # Calculate distances from median curves for each individual
    distances_class_0 = [euclidean_distance(curve, median_hazard_class_0_train) for curve in surv]
    distances_class_1 = [euclidean_distance(curve, median_hazard_class_1_train) for curve in surv]

    # Determine predicted class based on proximity to median curves
    predicted_classes = []
    for dist_0, dist_1 in zip(distances_class_0, distances_class_1):
        if dist_0 < dist_1:
            predicted_classes.append(0)
        else:
            predicted_classes.append(1)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test.values, predicted_classes, labels=[False, True])

    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    balanced_accuracy = (sensitivity + specificity) / 2

    mcc = matthews_corrcoef(y_test.values, predicted_classes)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("Sensitivity:", round(sensitivity,2))
    print("Specificity:", round(specificity,2))
    print("Positive Predictive Value (PPV):", round(ppv,2))
    print("Negative Predictive Value (NPV):", round(npv,2))
    print("Balanced Accuracy:", round(balanced_accuracy,2))
    print("Matthews Correlation Coefficient (MCC):", round(mcc,2))

    # Plot confusion matrix with actual label names
    # class_labels_display = [class_0, class_1]
    plt.figure(figsize=(4, 3))  
    myheatmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_labels_display, yticklabels=class_labels_display, annot_kws={"size": 8}) 
    myheatmap.invert_yaxis()
    plt.title("Confusion Matrix for the test set", fontsize=10)  
    plt.xlabel("Predicted Label", fontsize=8)  
    plt.ylabel("True Label", fontsize=8)  
    plt.xticks(fontsize=8)  
    plt.yticks(fontsize=8)  
    plt.show()




# %%
if survival_analysis and external_val:

    # Predict cumulative hazard function for all test samples
    surv_extval = rsf.predict_cumulative_hazard_function(X_extval_surv, return_array=True)

    # Calculate distances from median curves for each individual
    distances_class_0 = [euclidean_distance(curve, median_hazard_class_0_train) for curve in surv_extval]
    distances_class_1 = [euclidean_distance(curve, median_hazard_class_1_train) for curve in surv_extval]

    # Determine predicted class based on proximity to median curves
    predicted_classes = []
    for dist_0, dist_1 in zip(distances_class_0, distances_class_1):
        if dist_0 < dist_1:
            predicted_classes.append(0)
        else:
            predicted_classes.append(1)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_extval_data.values, predicted_classes, labels=[False, True])

    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    balanced_accuracy = (sensitivity + specificity) / 2

    mcc = matthews_corrcoef(y_extval_data.values, predicted_classes)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("Sensitivity:", round(sensitivity,2))
    print("Specificity:", round(specificity,2))
    print("Positive Predictive Value (PPV):", round(ppv,2))
    print("Negative Predictive Value (NPV):", round(npv,2))
    print("Balanced Accuracy:", round(balanced_accuracy,2))
    print("Matthews Correlation Coefficient (MCC):", round(mcc,2))

    # Plot confusion matrix with actual label names
    # class_labels_display = [class_0, class_1]
    plt.figure(figsize=(4, 3))  
    myheatmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_labels_display, yticklabels=class_labels_display, annot_kws={"size": 8}) 
    myheatmap.invert_yaxis()
    plt.title("Confusion matrix for the external validation set", fontsize=10)  
    plt.xlabel("Predicted label", fontsize=8)  
    plt.ylabel("True label", fontsize=8)  
    plt.xticks(fontsize=8)  
    plt.yticks(fontsize=8)  
    plt.show()




# %%
if survival_analysis:
    # Predict cumulative hazard function for all test samples
    surv_test = rsf.predict_cumulative_hazard_function(X_test_surv, return_array=True)

    class_0_indices = np.where(y_test.values == False)[0] 
    class_1_indices = np.where(y_test.values == True)[0]  
    
  
    # Separate predictions into classes
    surv_class_0 = surv_test[class_0_indices]
    surv_class_1 = surv_test[class_1_indices]

    # Calculate median and interquartile range for both classes
    median_surv_class_0_test = np.median(surv_class_0, axis=0)
    q1_surv_class_0_test = np.percentile(surv_class_0, 25, axis=0)
    q3_surv_class_0_test = np.percentile(surv_class_0, 75, axis=0)
    iqr_surv_class_0_test = q3_surv_class_0_test - q1_surv_class_0_test

    median_surv_class_1_test = np.median(surv_class_1, axis=0)
    q1_surv_class_1_test = np.percentile(surv_class_1, 25, axis=0)
    q3_surv_class_1_test = np.percentile(surv_class_1, 75, axis=0)
    iqr_surv_class_1_test = q3_surv_class_1_test - q1_surv_class_1_test
    
   
    rsf_riskscores_test = rsf.predict(X_test_surv)
    
    # Calculate average risk scores for each class in the test set
    predicted_risk_socres_class_0 = rsf_riskscores_test[y_test]
    predicted_risk_socres_class_1 = rsf_riskscores_test[~y_test]
    
    # Perform Mann-Whitney U test to compare the medians of the two classes
    statistic, p_value = mannwhitneyu(predicted_risk_socres_class_0, predicted_risk_socres_class_1)
    # Add annotation for statistical test
    if p_value < 0.001:
        p_value_text = "<0.001"
    else:
        p_value_text = f"= {p_value:.4f}"
        
    # Create subplots: one for the survival plot and one for the table
    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))

    # Plot median and interquartile range for class 0 on the top plot
    ax1.step(rsf.unique_times_, median_surv_class_0_test, where="post", label=f"median of predicted cumulative hazard in samples from {class_labels_display[0]} class", color='b')
    ax1.fill_between(rsf.unique_times_, q1_surv_class_0_test, q3_surv_class_0_test, step="post", alpha=0.3, color='b', label="IQR")

    # Plot median and interquartile range for class 1
    ax1.step(rsf.unique_times_, median_surv_class_1_test, where="post", label=f"median of predicted cumulative hazard in samples from {class_labels_display[1]} class", color='r')
    ax1.fill_between(rsf.unique_times_, q1_surv_class_1_test, q3_surv_class_1_test, step="post", alpha=0.3, color='r', label="IQR")
    ax1.set_ylabel("Predicted cumulative hazard (test set)") 
    ax1.set_xlabel("Time in days") # Note: you should manually change it if for example the time is in months or years rather than in days
    ax1.legend()
    ax1.grid(True)
    
    # Divide time into 5 intervals
    num_intervals = 5
    time_intervals = np.linspace(min(rsf.unique_times_), max(rsf.unique_times_), num_intervals + 1)

    # Initialize arrays to store at risk, events, and censored counts for each interval
    at_risk_intervals = np.zeros(num_intervals, dtype=int)
    events_intervals = np.zeros(num_intervals, dtype=int)
    censored_intervals = np.zeros(num_intervals, dtype=int)
    
    # Loop through each interval and calculate metrics
    for i in range(num_intervals):
        start_time = time_intervals[i]
        end_time = time_intervals[i+1]
        
        # Patients at risk at the start of the interval
        at_risk_intervals[i] = np.sum(y_test_surv >= start_time)
        
        # Events within the interval
        events_intervals[i] = np.sum((y_test_surv > start_time) & (y_test_surv <= end_time) & (y_test == True))
        
        # Censored within the interval
        censored_intervals[i] = np.sum((y_test_surv > start_time) & (y_test_surv < end_time) & (y_test == False))
        
    # Create the table data with individual and cumulative counts
    table_data = np.array([
        [f'{int(time_intervals[i])}-{int(time_intervals[i+1])}' for i in range(num_intervals)],
        [f'{at_risk_intervals[i]}' for i in range(num_intervals)],
        [f'{events_intervals[i]}' for i in range(num_intervals)],
        [f'{censored_intervals[i]}' for i in range(num_intervals)]
    ])

    # Create the table in the second subplot
    row_labels = ["time interval",'at risk', 'events', 'censored']

    # Hide the axis for the table
    ax2.axis('tight')
    ax2.axis('off')

    # Add the table to the second subplot
    table_display = table(ax2, cellText=table_data, rowLabels=row_labels,cellLoc='center', loc='center')
    
    # Add annotation for statistical test
    ax1.annotate(f'Mann-Whitney U test \n comparing the predicted risk scores \np {p_value_text}', xy=(0.15, 0.3), xycoords='axes fraction', ha='center', va='center', fontsize = 10)
    
    # Adjust the layout to ensure the plots don't overlap
    plt.gca().set_facecolor('white')
    plt.grid(which='both', color="grey")
    plt.grid(which='minor', alpha=0.1)
    plt.grid(which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig("predicted_hazard_test.tif", bbox_inches='tight')
    # Show the final plot
    plt.show()


# %%
if survival_analysis:
    # conduct a statistical test to see if there is any significant difference between the groups in their survival data
    # This is what it does according to https://scikit-survival.readthedocs.io/en/v0.23.0/api/generated/sksurv.compare.compare_survival.html#sksurv.compare.compare_survival
    # K-sample log-rank hypothesis test of identical survival functions.
    # Compares the pooled hazard rate with each group-specific hazard rate. The alternative hypothesis is that the hazard rate of at least one group differs from the others at some time.

    # Run the survival comparison
    chisq, pvalue, stats, covariance = compare_survival(
        y=y_test_surv_transformed,
        group_indicator=y_test,
        return_stats=True
    )

    # Prepare the data to create a reportable DataFrame
    comparison_data = {
        "Chi-Square": chisq,
        "p-value": pvalue,
        "Statistics": stats,
        "Covariance": covariance
    }

    # Convert the dictionary to a DataFrame for better readability
    comparison_df = pd.DataFrame([comparison_data])
    comparison_df.to_excel("comparison_df_surv_testset.xlsx")


# %%
if survival_analysis and external_val:
    # Predict cumulative hazard function for all external validation samples
    surv_extval = rsf.predict_cumulative_hazard_function(X_extval_surv, return_array=True)

    class_0_indices = np.where(y_extval_data.values == False)[0] 
    class_1_indices = np.where(y_extval_data.values == True)[0]  
    
  
    # Separate predictions into classes
    surv_class_0 = surv_extval[class_0_indices]
    surv_class_1 = surv_extval[class_1_indices]

    # Calculate median and interquartile range for both classes
    median_surv_class_0_extval = np.median(surv_class_0, axis=0)
    q1_surv_class_0_extval = np.percentile(surv_class_0, 25, axis=0)
    q3_surv_class_0_extval = np.percentile(surv_class_0, 75, axis=0)
    iqr_surv_class_0_extval = q3_surv_class_0_extval - q1_surv_class_0_extval

    median_surv_class_1_extval = np.median(surv_class_1, axis=0)
    q1_surv_class_1_extval = np.percentile(surv_class_1, 25, axis=0)
    q3_surv_class_1_extval = np.percentile(surv_class_1, 75, axis=0)
    iqr_surv_class_1_extval = q3_surv_class_1_extval - q1_surv_class_1_extval
    
   
    rsf_riskscores_extval = rsf.predict(X_extval_surv)
    
    # Calculate average risk scores for each class in the test set
    predicted_risk_socres_class_0 = rsf_riskscores_extval[y_extval_data]
    predicted_risk_socres_class_1 = rsf_riskscores_extval[~y_extval_data]
    
    # Perform Mann-Whitney U test to compare the medians of the two classes
    statistic, p_value = mannwhitneyu(predicted_risk_socres_class_0, predicted_risk_socres_class_1)
    # Add annotation for statistical test
    if p_value < 0.001:
        p_value_text = "<0.001"
    else:
        p_value_text = f"= {p_value:.4f}"
        
    # Create subplots: one for the survival plot and one for the table
    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))

    # Plot median and interquartile range for class 0 on the top plot
    ax1.step(rsf.unique_times_, median_surv_class_0_extval, where="post", label=f"median of predicted cumulative hazard in samples from {class_labels_display[0]} class", color='b')
    ax1.fill_between(rsf.unique_times_, q1_surv_class_0_test, q3_surv_class_0_extval, step="post", alpha=0.3, color='b', label="IQR")

    # Plot median and interquartile range for class 1
    ax1.step(rsf.unique_times_, median_surv_class_1_extval, where="post", label=f"median of predicted cumulative hazard in samples from {class_labels_display[1]} class", color='r')
    ax1.fill_between(rsf.unique_times_, q1_surv_class_1_extval, q3_surv_class_1_extval, step="post", alpha=0.3, color='r', label="IQR")
    ax1.set_ylabel("Predicted cumulative hazard (external validation set)") 
    ax1.set_xlabel("Time in days") # Note: you should manually change it if for example the time is in months or years rather than in days
    ax1.legend()
    ax1.grid(True)
    
    # Divide time into 5 intervals
    num_intervals = 5
    time_intervals = np.linspace(min(rsf.unique_times_), max(rsf.unique_times_), num_intervals + 1)

    # Initialize arrays to store at risk, events, and censored counts for each interval
    at_risk_intervals = np.zeros(num_intervals, dtype=int)
    events_intervals = np.zeros(num_intervals, dtype=int)
    censored_intervals = np.zeros(num_intervals, dtype=int)
    
    # Loop through each interval and calculate metrics
    for i in range(num_intervals):
        start_time = time_intervals[i]
        end_time = time_intervals[i+1]
        
        # Patients at risk at the start of the interval
        at_risk_intervals[i] = np.sum(y_extval_surv >= start_time)
        
        # Events within the interval
        events_intervals[i] = np.sum((y_extval_surv > start_time) & (y_extval_surv <= end_time) & (y_extval_data == True))
        
        # Censored within the interval
        censored_intervals[i] = np.sum((y_extval_surv > start_time) & (y_extval_surv < end_time) & (y_extval_data == False))
        
    # Create the table data with individual and cumulative counts
    table_data = np.array([
        [f'{int(time_intervals[i])}-{int(time_intervals[i+1])}' for i in range(num_intervals)],
        [f'{at_risk_intervals[i]}' for i in range(num_intervals)],
        [f'{events_intervals[i]}' for i in range(num_intervals)],
        [f'{censored_intervals[i]}' for i in range(num_intervals)]
    ])

    # Create the table in the second subplot
    row_labels = ["time interval",'at risk', 'events', 'censored']

    # Hide the axis for the table
    ax2.axis('tight')
    ax2.axis('off')

    # Add the table to the second subplot
    table_display = table(ax2, cellText=table_data, rowLabels=row_labels,cellLoc='center', loc='center')
    
    # Add annotation for statistical test
    ax1.annotate(f'Mann-Whitney U test \n comparing the predicted risk scores \np {p_value_text}', xy=(0.15, 0.3), xycoords='axes fraction', ha='center', va='center', fontsize = 10)
    
    # Adjust the layout to ensure the plots don't overlap
    plt.gca().set_facecolor('white')
    plt.grid(which='both', color="grey")
    plt.grid(which='minor', alpha=0.1)
    plt.grid(which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig("predicted_hazard_extval.tif", bbox_inches='tight')
    # Show the final plot
    plt.show()


# %%
if survival_analysis and external_val:
    # Run the survival comparison
    chisq, pvalue, stats, covariance = compare_survival(
        y=y_extval_surv_transformed,
        group_indicator=y_extval_data,
        return_stats=True
    )

    # Prepare the data to create a reportable DataFrame
    comparison_data = {
        "Chi-Square": chisq,
        "p-value": pvalue,
        "Statistics": stats,
        "Covariance": covariance
    }

    # Convert the dictionary to a DataFrame for better readability
    comparison_df = pd.DataFrame([comparison_data])
    comparison_df.to_excel("comparison_df_surv_extval.xlsx")

# %%
if survival_analysis:
    
    try:

        # Ensure test times are correctly extracted from the structured survival arrays
        test_times_rsf = y_test_surv_transformed['time']  # Extract the time from the structured array
        valid_times_rsf = np.unique(test_times_rsf)
        valid_times_rsf = np.clip(valid_times_rsf, test_times_rsf.min(), test_times_rsf.max() - 1e-5)

        # Predict the risk scores using the Random Survival Forest model
        rsf_risk_scores = rsf.predict(X_test_surv)

        # Calculate cumulative dynamic AUC
        rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
            survival_train=y_train_surv_transformed,
            survival_test=y_test_surv_transformed,
            estimate=rsf_risk_scores,
            times=valid_times_rsf
        )

        # Plot the AUC over time
        plt.figure(figsize=(8, 6))
        plt.plot(valid_times_rsf, rsf_auc, label=f'RSF AUC (mean = {rsf_mean_auc:.2f})', color='b')
        plt.xlabel('Time')
        plt.ylabel('AUC (test set)')
        plt.title('AUC over time for random survival forest')
        plt.legend()
        plt.grid(True)
        plt.show()


    except Exception as e:
        print(f"An error occurred: {e}. Skipping to the next block.")


# %%
if survival_analysis and external_val:
    
    try:

        # Ensure test times are correctly extracted from the structured survival arrays
        test_times_rsf = y_extval_surv_transformed['time']  # Extract the time from the structured array
        valid_times_rsf = np.unique(test_times_rsf)
        valid_times_rsf = np.clip(valid_times_rsf, test_times_rsf.min(), test_times_rsf.max() - 1e-5)

        # Predict the risk scores using the Random Survival Forest model
        rsf_risk_scores = rsf.predict(X_extval_surv)

        # Calculate cumulative dynamic AUC
        rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
            survival_train=y_train_surv_transformed,
            survival_test=y_extval_surv_transformed,
            estimate=rsf_risk_scores,
            times=valid_times_rsf
        )

        # Plot the AUC over time
        plt.figure(figsize=(8, 6))
        plt.plot(valid_times_rsf, rsf_auc, label=f'RSF AUC (mean = {rsf_mean_auc:.2f})', color='b')
        plt.xlabel('Time')
        plt.ylabel('AUC (external validation set)')
        plt.title('AUC over time for random survival forest')
        plt.legend()
        plt.grid(True)
        plt.show()


    except Exception as e:
        print(f"An error occurred: {e}. Skipping to the next block.")


# %% [markdown]
# ## Regression models
# 
# Like survival analysis, if the data contains a column for continuous outcome variable then this analysis is relevant and can be conducted using the following code chunks. The continuous outcome is provided from a copy of the data that is saved in the beginning of the pipeline and it gets merged back to the train and test sets.

# %% [markdown]
# ### Interpreting Regression Model Performance Metrics
# 
# 1) Mean Squared Error (MSE)
# - **Formula:** MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
# - **Where:**
#   - \( n \) is the number of observations
#   - \( yᵢ \) is the actual value
#   - \( ŷᵢ \) is the predicted value
# - **Interpretation:**
#   - Measures the average squared difference between actual and predicted values.
#   - Lower MSE indicates better fit.
#   - Sensitive to outliers.
# 
# 2) Mean Absolute Error (MAE)
# - **Formula:** MAE = (1/n) * Σ|yᵢ - ŷᵢ|
# - **Where:**
#   - \( n \) is the number of observations
#   - \( yᵢ \) is the actual value
#   - \( ŷᵢ \) is the predicted value
# - **Interpretation:**
#   - Measures the average absolute difference between actual and predicted values.
#   - Lower MAE indicates better fit.
#   - Less sensitive to outliers than MSE.
#   - Same units as the original data.
# 
# 3) R-squared (R²)
# - **Formula:** R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
# - **Where:**
#   - \( yᵢ \) is the actual value
#   - \( ŷᵢ \) is the predicted value
#   - \( ȳ \) is the mean of actual values
# - **Interpretation:**
#   - Measures the proportion of variance in the dependent variable explained by the model.
#   - Values range from -∞ to 1.
#   - Higher values indicate better fit.
#   - Negative values indicate the model performs worse than a horizontal line (mean of the target variable).

# %%
if regression_analysis:
    y_train_reg = pd.merge(y_train, mydata_copy_regression[regression_outcome], left_index=True, right_index=True, how='left')
    y_train_reg.drop(columns=outcome_var,inplace = True)
    y_test_reg = pd.merge(y_test, mydata_copy_regression[regression_outcome], left_index=True, right_index=True, how='left')
    y_test_reg.drop(columns=outcome_var,inplace = True)
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train_OHE_nocv, y_train_reg)

    # Make predictions on the test set
    y_pred = model.predict(X_test_OHE)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_reg, y_pred)
    mae = mean_absolute_error(y_test_reg, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print("Mean Squared Error (MSE): {:.2f}".format(mse))
    print("Mean Absolute Error (MAE): {:.2f}".format(mae))
    print("R-squared (R2): {:.2f}".format(r2))


# %%
if regression_analysis:
    # Define the parameter grid for the random search
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 500, 1000],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at each leaf node
    }

    # Create Random Forest model
    rf = RandomForestRegressor()

    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_grid, n_iter=n_iter_hptuning,
                                       scoring='neg_mean_squared_error', cv=cv_folds_hptuning)

    # Perform random search to find the best hyperparameters
    random_search.fit(X_train_OHE_nocv, y_train_reg)

    # Extract the best parameters
    best_params = random_search.best_params_
    print("Best parameters found: ", best_params)

    # Train the final model on the entire training set with the best hyperparameters
    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X_train_OHE_nocv, y_train_reg)

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test_OHE)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_reg, y_pred)
    mae = mean_absolute_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)

    # Print the evaluation metrics
    print("Mean Squared Error (MSE): {:.2f}".format(mse))
    print("Mean Absolute Error (MAE): {:.2f}".format(mae))
    print("R-squared (R2): {:.2f}".format(r2))


# %% [markdown]
# ### Model interpretation

# %%
if regression_analysis:
    
    # Initialize explainer with the best model
    explainer = shap.TreeExplainer(best_model)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test_OHE)

    # Summary plot
    shap.summary_plot(shap_values, X_test_OHE, feature_names=X_train_OHE_nocv.columns)
    
    # save the model to disk
    joblib.dump(best_model, 'regression_model.pkl')


# %% [markdown]
# ## Report the environment
# 
# Report Conda packages required to run the pipeline

# %%
def get_conda_environment():
    """
    This function retrieves the list of packages currently installed in a conda environment. It runs the `conda list` command and returns the output as a string.

    ### Returns

    A string containing the list of packages installed in the conda environment.

    ### Notes

    *   This function assumes that the `conda` command is available on the system and that the user has permission to run it.
    *   The returned string may be truncated if it exceeds the maximum allowed size by the system.
    """
    conda_list = subprocess.check_output(['conda', 'list']).decode('utf-8')
    return conda_list

def get_python_info():
    """
    This function retrieves the version of Python currently installed on the system. It uses the `platform.python_version()` function to determine the version.

    ### Returns

    A string representing the version of Python.

    ### Notes

    *   This function may return a version in the format 'X.X.Y' or 'X.X', depending on the version of the platform library being used.
    *   The returned version is specific to the Python interpreter being run, not necessarily the default Python interpreter for the system.
    """
    
    python_version = platform.python_version()
    return python_version

def get_system_info():
    """
    This function retrieves information about the operating system and hardware configuration of the system. It uses various functions from the `platform` and `psutil` libraries to gather information about the OS, number of CPUs, and memory usage.

    ### Returns

    A dictionary containing three key-value pairs:

    *   `'OS'`: The name of the operating system (e.g., 'Windows', 'Linux', 'Darwin').
    *   `'Number of CPUs'`: The total number of CPU cores available on the system.
    *   `'Memory'`: A string representation of the current memory usage, including both physical and virtual memory.

    ### Notes

    *   This function assumes that the necessary permissions are available to access information about the system hardware.
    *   The returned dictionary is specific to the current Python interpreter being run, not necessarily the default Python interpreter for the system.

    """
    system_info = {
        'OS': platform.system(),
        'Number of CPUs': n_cpu_model_training,
        'Memory': f'{psutil.virtual_memory().total / (1024 ** 3):.2f} GB'
    }
    return system_info

def get_gpu_info():
    """
    This function retrieves information about NVIDIA GPUs present on the system. If an NVIDIA GPU is detected, it runs the `nvidia-smi` command and returns its output as a string. Otherwise, it returns a message indicating that no NVIDIA GPU is available.

    ### Returns

    A string containing the output of the `nvidia-smi` command if an NVIDIA GPU is detected; otherwise, a message indicating that no NVIDIA GPU is available.

    ### Notes

    *   This function assumes that the `nvidia-smi` command is available on the system and that the user has permission to run it.
    *   The returned string may be truncated if it exceeds the maximum allowed size by the system.
    """
    if GPU_avail:
        try:
            gpu_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            return gpu_info
        except subprocess.CalledProcessError:
            return "GPU information not available (nvidia-smi not installed or no NVIDIA GPU detected)"
    else:
        return "GPU not used"
        
# Record end time
end_time = time.time()

# Calculate duration
duration = end_time - start_time

# Define the filename for the report
report_filename = 'pipeline_report.txt'

# Open the file for writing
with open(report_filename, 'w') as f:
    # Write Conda environment to the file
    f.write("Conda environment:\n")
    f.write(get_conda_environment())
    f.write("\n\n")
    
    # Write Python version to the file
    f.write("Python version:\n")
    f.write(get_python_info())
    f.write("\n\n")
    
    # Write system information to the file
    f.write("System information:\n")
    system_info = get_system_info()
    for key, value in system_info.items():
        f.write(f"{key}: {value}\n")
    f.write("\n")
    
    # Write GPU information to the file
    f.write("GPU information:\n")
    f.write(get_gpu_info())
    f.write("\n")
    
    # Write duration to the file
    f.write("Pipeline execution duration (seconds):\n")
    f.write(str(duration))

print(f"Report saved as {report_filename}")


# %% [markdown]
# ## Save the pipeline logs in HTML format
# 
# 

# %%
def save_notebook():
    """
    This function saves the current state of a Jupyter Notebook to disk, effectively pausing the notebook's execution and preserving its environment, code, and data. The `IPython.notebook.save_checkpoint()` function is used internally by Jupyter to perform this task.

    ### Notes

    *   When this function is called, it will save the notebook's current state to a file in the format `.ipynb`, which can be loaded later into the same or another instance of Jupyter Notebook.
    *   This function does not execute any code; it simply saves the current state of the notebook and returns control to the caller.

    By calling this function, you can:

    *   Pause an ongoing computation and resume it later without losing its progress
    *   Save a snapshot of your work for reference or sharing with others
    *   Ensure that your changes are persisted even if the notebook is terminated unexpectedly

    However, note that saving a notebook will also save any unsaved changes to the cell contents, which may not be what you want in all cases. To avoid this, consider using the `save` method explicitly.
    """
    display(Javascript('IPython.notebook.save_checkpoint();'))

save_notebook()





# %% [markdown]
# ## References
# 
# Information about all the packages (libraries) utilized in this pipeline is available in the exported report "pipeline_report.txt". 
# 
# More information about the methods used in this pipeline:
# 
# - **QLattice model**
#   - Broløs, K. R. et al. An Approach to Symbolic Regression Using Feyn. (2021)
# 
# - **Sci-kit learn**
#   - Pedregosa, F. et al. Scikit-learn: Machine learning in Python. J. Mach. Learn. Res. 12, 2825–2830 (2011)
# 
# - **CatBoost**
#   - Dorogush, A. V., Ershov, V. & Gulin, A. CatBoost: gradient boosting with categorical features support. (2018)
# 
# - **LightGBM**
#   - Ke, G. et al. LightGBM: A highly efficient gradient boosting decision tree. in Advances in Neural Information Processing Systems (2017)
# 
# - **SHAP**
#   - Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. in Advances in neural information processing systems 4765–4774 (2017)
#   - Lundberg, S. M. et al. From local explanations to global understanding with explainable AI for trees. Nat. Mach. Intell. 2, 56–67 (2020)
# 
# - **SHAP clustering**
#   - Zargari Marandi, R. et al. Development of a machine learning model for early prediction of plasma leakage in suspected dengue patients. PLoS Negl. Trop. Dis. 17, e0010758 (2023)
#   - Ramtin Zargari Marandi, ExplaineR: an R package to explain machine learning models, Bioinformatics Advances, Volume 4, Issue 1, 2024, vbae049, [link](https://doi.org/10.1093/bioadv/vbae049)
#   
# - **Survival SHAP**
#   - Krzyziński, Mateusz, et al. "SurvSHAP (t): time-dependent explanations of machine learning survival models." Knowledge-Based Systems 262 (2023): 110234.
#   - https://github.com/MI2DataLab/survshap
# 
# More information about for example data imputation and clustering methods, and other models can be found in https://scikit-learn.org/stable/.
# 


