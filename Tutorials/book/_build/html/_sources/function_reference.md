# Function Reference

## Glossary

- [__del__](#__del__)
- [__init__](#__init__)
- [adjust_hyperparameters](#adjust_hyperparameters)
- [adjust_hyperparameters_surv_models](#adjust_hyperparameters_surv_models)
- [aggregate_shap_values_with_base](#aggregate_shap_values_with_base)
- [bootstrap_sample](#bootstrap_sample)
- [calculate_biserial_corr](#calculate_biserial_corr)
- [calculate_confidence_interval](#calculate_confidence_interval)
- [calculate_metrics](#calculate_metrics)
- [calculate_missingness](#calculate_missingness)
- [calculate_mutual_info](#calculate_mutual_info)
- [calculate_surv_metrics](#calculate_surv_metrics)
- [categorical_shap_plot](#categorical_shap_plot)
- [check_categorical_difference](#check_categorical_difference)
- [check_numerical_difference](#check_numerical_difference)
- [combined_metric](#combined_metric)
- [conformal_p_value](#conformal_p_value)
- [cost_sensitive_net_benefit](#cost_sensitive_net_benefit)
- [create_summary_table](#create_summary_table)
- [cross_validate_model](#cross_validate_model)
- [cross_validate_surv_model](#cross_validate_surv_model)
- [decision_curve_analysis](#decision_curve_analysis)
- [display_confusion_matrix](#display_confusion_matrix)
- [euclidean_distance](#euclidean_distance)
- [evaluate_and_plot_model](#evaluate_and_plot_model)
- [evaluate_params](#evaluate_params)
- [evaluate_params_kfold](#evaluate_params_kfold)
- [extract_mean](#extract_mean)
- [f_imp_shapboxplot](#f_imp_shapboxplot)
- [f_imp_shapboxplot_surv](#f_imp_shapboxplot_surv)
- [filter_columns](#filter_columns)
- [filter_rows](#filter_rows)
- [find_feature_clusters](#find_feature_clusters)
- [generate_subsample](#generate_subsample)
- [get_conda_environment](#get_conda_environment)
- [get_gpu_info](#get_gpu_info)
- [get_python_info](#get_python_info)
- [get_samples_per_class](#get_samples_per_class)
- [get_system_info](#get_system_info)
- [integrated_brier_score](#integrated_brier_score)
- [ipy_exit](#ipy_exit)
- [is_hyperparameter_tuning_suitable](#is_hyperparameter_tuning_suitable)
- [parallel_compute_shap_surv](#parallel_compute_shap_surv)
- [permute_feature_pairs](#permute_feature_pairs)
- [PFI_median_wrap](#pfi_median_wrap)
- [plot_confusion_matrix_for_clusters](#plot_confusion_matrix_for_clusters)
- [plot_PFI](#plot_pfi)
- [plot_samples_per_class_per_fold](#plot_samples_per_class_per_fold)
- [plot_survshap_detailed](#plot_survshap_detailed)
- [plot_TFI](#plot_tfi)
- [save_notebook](#save_notebook)
- [set_parameters](#set_parameters)
- [set_params_surv_models](#set_params_surv_models)
- [shap_summary_plot](#shap_summary_plot)
- [shorten_column_names](#shorten_column_names)
- [shorten_data_dictionary](#shorten_data_dictionary)
- [subsample_iqr_test](#subsample_iqr_test)

---

## `__del__(self)`

**Description:** The `__del__` method closes the captured output buffer and restores it to its original value, ensuring proper cleanup when exiting.

---

## `__init__(self)`

**Description:** This defines the constructor (`__init__`) and destructor (`__del__`) methods for the `IpyExit` custom exception class.

---

## `adjust_hyperparameters(n_rows, n_cols)`

**Description:** Returns a dictionary of hyperparameter distributions for various machine learning models.

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

---

## `adjust_hyperparameters_surv_models(n_rows)`

**Description:** Returns a dictionary of hyperparameter distributions for various survival models.

## Parameters
    n_rows (int): The number of rows in the dataset.

## Returns
    dict: A dictionary containing the following keys:
        - 'RSF_param_dist': Hyperparameters for Random Forest Classifier.
        - 'CoxPH_param_dist': Hyperparameters for Cox Proportional Hazards Model.
        - 'Coxnet_param_dist': Hyperparameters for CoxnetSurvivalAnalysis.

---

## `aggregate_shap_values_with_base(survshaps, X_test_surv, aggregation='mean_abs')`

**Description:** Aggregate SHAP values from a survival model.

## Parameters
    survshaps (list): A list of Shapely objects containing survival model results.
    X_test_surv (pd.DataFrame): The test set used for feature selection and model evaluation.
    aggregation (str, optional): The method to use for aggregating SHAP values. Can be 'mean_abs', 'sum', or 'mean'. Defaults to 'mean_abs'.

## Returns
    tuple: A tuple containing two arrays:
        - shap_values (np.ndarray): An array of shape (n_samples, n_features) where each row represents the aggregated SHAP value for a sample.
        - base_values (np.ndarray): An array of shape (n_samples,) where each element is the base value for a sample.

---

## `bootstrap_sample(data, n_samples)`

**Description:** Perform bootstrap sampling on the input data.

## Parameters:
- data (array-like): Input data to be sampled.
- n_samples (int): Number of samples to generate.

## Returns
- indices (numpy array): Indices of the original data used for bootstrapping.
- resampled_data (array-like): Resampled data with shape (n_samples, len(data)).

---

## `calculate_biserial_corr(subsample, outcome_var)`

**Description:** Calculates point-biserial correlation for each variable in the subsample against the target.

## Parameters
    subsample (DataFrame): Subsample of the dataset.
    outcome_var (str): Name of the target variable.

## Returns
    corr_values (dict): Dictionary containing correlation values for each variable.

---

## `calculate_confidence_interval(metric_values, alpha=0.95)`

**Description:** Calculate the confidence interval for the given metric values.

## Parameters:
- metric_values (array-like): Input metric values.
- alpha (float, optional): Confidence level. Defaults to 0.95.

## Returns
- lower_bound (float or numpy array): Lower bound of the confidence interval.
- upper_bound (float or numpy array): Upper bound of the confidence interval.

---

## `calculate_metrics(y_true, y_pred, y_pred_proba)`

**Description:** Calculates various evaluation metrics for binary classification models.

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

---

## `calculate_missingness(data, output_file='missingness_report.xlsx')`

**Description:** Reports missingness in both categorical and continuous variables and saves the results to an Excel file.

This function calculates the percentage of missing values for each column in the input data,
corrects these percentages for categorical columns where 'missing' is a valid category,
and computes the mean and standard deviation of the missingness across all columns.

## Parameters
    data (pandas.DataFrame): The input data containing both categorical and continuous variables.
    output_file (str, optional): The file path to save the results. Defaults to 'missingness_report.xlsx'.

## Returns
    None

---

## `calculate_mutual_info(subsample, outcome_var)`

**Description:** Calculates mutual information between each feature in the subsample and the target variable.

## Parameters
    subsample (DataFrame): Subsample of the dataset.
    outcome_var (str): Name of the target variable.

## Returns
    mi_values (dict): Dictionary containing mutual information values for each feature.

---

## `calculate_surv_metrics(y_true, model, X_test, survival_train, survival_test)`

**Description:** Calculate and compute key metrics for survival prediction models.

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

---

## `categorical_shap_plot(shap_values, data, top_n=10, jitter=0.1)`

**Description:** This function creates a plot of SHAP values for the top N features in a dataset, where categorical features are displayed as scatter plots with different colors and numerical features are displayed as individual points. The plot includes a colorbar for numerical features and labels for categorical features.

## Parameters:
    shap_values (numpy array): Matrix of SHAP values to be plotted.
    data (pandas DataFrame): Dataset containing feature names and values.
    top_n (int, optional): Number of top features to include in the plot. Defaults to 10.
    jitter (float, optional): Jitter value for scatter plots. Defaults to 0.1.

## Returns
    fig: Matplotlib figure object containing the SHAP plot.

---

## `check_categorical_difference(train_data, test_data)`

**Description:** Compares categorical differences between training and testing datasets, 
using the Chi-square test to determine if there are significant differences in distribution.

## Parameters
train_data: The original training data.
test_data: The test set obtained from train_test_split.

## Returns
A dictionary containing the statistical difference results for each categorical variable.

---

## `check_numerical_difference(train_data, test_data)`

**Description:** check_numerical_difference compares statistical differences between numerical variables in training and testing datasets.

This function takes two dataframes (train_data and test_data) as input, identifies the numerical columns, and computes the Mann-Whitney U-statistic to determine if there are significant differences between the distributions of these numerical variables in the training and testing datasets. It returns a dictionary containing the results for each identified variable, including the statistic and p-value.

## Parameters
train_data: The original training data.
test_data: The test set obtained from train_test_split.

## Returns
A dictionary containing the statistical difference results for each numerical variable.

---

## `combined_metric(y_true, y_pred_proba)`

**Description:** Calculates the combined custom metric for hyperparameter tuning, which is the mean of 
ROCAUC (Receiver Operating Characteristic Area Under Curve) and PRAUC (Precision-Recall AUC).

This metric combines both the classification accuracy (ROCAUC) and precision (PRAUC) to 
provide a balanced evaluation of model performance, particularly useful for binary classification tasks.

## Parameters
    y_true (array-like): True labels for which each sample was predicted.
    y_pred_proba (array-like): Predicted probabilities for each sample.

## Returns
    float: The combined custom metric value, representing the mean of ROCAUC and PRAUC.

---

## `conformal_p_value(prob, calibration_scores, true_label)`

**Description:** Calculates the p-value for conformal prediction using the calibration scores.

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

---

## `cost_sensitive_net_benefit(tp, fp, threshold, N, w_tp=1, w_fp=1)`

**Description:** Calculates the net benefit of a classification model under a given probability threshold.

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

---

## `create_summary_table(dataframe, dataset_name)`

**Description:** Creates a summary table for a single dataset.

This function generates a table that summarizes key statistics about each variable in the dataset,
including numerical variables (median and quartiles) and categorical variables (categories, counts, and percentages).
Additionally, it includes information on missing values and adds a column for the dataset name.

## Parameters
    dataframe (pd.DataFrame): The input DataFrame to generate summary statistics from.
    dataset_name (str): The name of the dataset being summarized.

## Returns
    pd.DataFrame: A new DataFrame containing the summary statistics.

---

## `cross_validate_model(model_class, X, y, sample_weights=None, n_splits=cv_folds, random_state=SEED, measures=None, use_default_threshold=False)`

**Description:** Perform k-fold cross-validation and evaluate the model.

## Parameters:
    X (array-like): Feature data.
    y (array-like): Target labels.
    model: Trained model instance.
    use_default_threshold (bool, optional): Use default threshold (0.5) for classification. Defaults to True.

## Returns
    tuple: Fold results, aggregated results table, optimal threshold, feature importance lists, and SHAP values list.

---

## `cross_validate_surv_model(model_class, X, y, n_splits=cv_folds, random_state=SEED, measures=None, hp_tuning=False)`

**Description:** This function performs cross-validation on a specified survival model class, evaluating its performance on a dataset. It allows users to tune hyperparameters using random search or grid search.

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

---

## `decision_curve_analysis(pred_probs_selected_model, pred_probs_alternative_model, rand_pred_probs, w_tp=1, w_fp=1)`

**Description:** Performs cost-sensitive decision curve analysis on two models.

The analysis plots the net benefit of each model under different probability thresholds 
and allows for comparison between the two models. The plot also includes a reference line 
representing the default threshold used in many machine learning applications.

## Parameters
    pred_probs_selected_model (array-like): Predicted probabilities of positive classes for the first model.
    pred_probs_alternative_model (array-like): Predicted probabilities of positive classes for the second model.
    rand_pred_probs (array-like): Randomly generated predicted probabilities for a third set of models.

## Returns
    None

---

## `display_confusion_matrix(y_true, y_pred, labels)`

**Description:** Displays a confusion matrix for model performance evaluation.

## Parameters
    y_true (np.ndarray): Ground truth labels.
    y_pred (np.ndarray): Predicted labels.
    labels (list or tuple): List of unique class labels in the data.

## Returns
    None: The function modifies the specified axes object and displays the plot.

---

## `euclidean_distance(x, y)`

**Description:** Calculates the Euclidean distance between two vectors.

The Euclidean distance is a measure of the straight-line distance between two points in n-dimensional space.
It is defined as the square root of the sum of the squared differences between corresponding elements in the input vectors.

## Parameters
    x (numpy array): The first vector.
    y (numpy array): The second vector.

## Returns
    float: The Euclidean distance between the two input vectors.

Note:
    This function assumes that the input vectors are of equal length. If they are not, an error will be raised.

---

## `evaluate_and_plot_model(model, testset, y_test, filename, class_labels=class_labels_display, threshold=0.5, bootstrap_samples=1000, min_positive_instances=1)`

**Description:** Calculates and visualizes model performance using ROC curve, PR curve, and confusion matrix.

## Parameters
    y_test (np.ndarray): Ground truth labels for the test dataset.
    predictions_class (np.ndarray): Predicted labels for the test dataset.
    class_labels_display (list or tuple): List of unique class labels in the data for display purposes.
    threshold (float): Threshold value for model evaluation.
    filename (str): Output file name for visualization.

## Returns
    results_df (DataFrame): DataFrame containing model performance metrics.
    missclassified_samples (list): List of indices of samples that were misclassified by the model.

---

## `evaluate_params(n_epochs, max_complexity)`

**Description:** Evaluate a composite model by tuning hyperparameters using the Feyn framework.

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

---

## `evaluate_params_kfold(n_epochs, max_complexity)`

**Description:** Evaluate a QLattice model for given hyperparameters.

## Parameters:
    n_epochs (int): The number of epochs used for training.
    max_complexity (int): The maximum complexity of the model.

## Returns
    QL_composite_score (float): The composite score of the model.
    params (dict): The hyperparameters used to achieve this score.

---

## `extract_mean(value)`

**Description:** Extract the mean value from a string.

This function uses regular expressions to search for a decimal number in the input string.
If a match is found, it returns the extracted value as a float. Otherwise, it returns None.

Parameters:
    value (str): The input string to extract the mean value from.

## Returns
    float or None: The extracted mean value as a float, or None if no match is found.

---

## `f_imp_shapboxplot(shap_values, X_test_OHE, X_test, selected_model, data_dictionary, num_features=20, num_subsamples=1000, random_seed=None, apply_threshold=False)`

**Description:** Plot the SHAP values for the top N most important features.

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

---

## `f_imp_shapboxplot_surv(shap_values, X_test, num_features=20, num_subsamples=1000, random_seed=None, apply_threshold=False)`

**Description:** The `f_imp_shapboxplot_surv` function generates an importance plot for SHAP (SHapley Additive exPlanations) values to identify the most influential features in a model. The function takes in SHAP value arrays, test data, and various parameters to customize the plot.

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

---

## `filter_columns(df, threshold=0.9)`

**Description:** Filter out columns with missingness greater than the threshold.

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

---

## `filter_rows(df, threshold=0.9)`

**Description:** Filter out rows with missingness greater than the threshold.

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

---

## `find_feature_clusters(X, shap_values, selected_model, data_dictionary, top_n_f)`

**Description:** This function analyzes SHAP values to identify feature clusters and instance clusters using hierarchical clustering. The goal is to visualize the relationships between features and instances in the dataset.

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

---

## `generate_subsample(df, seed)`

**Description:** Generates a subsample of the dataset using random sampling.

## Parameters
    df (DataFrame): Original dataset.
    seed (int): Random seed for sampling.

## Returns
    DataFrame: Subsample of the dataset.

---

## `get_conda_environment()`

**Description:** This function retrieves the list of packages currently installed in a conda environment. It runs the `conda list` command and returns the output as a string.

### Returns

A string containing the list of packages installed in the conda environment.

### Notes

*   This function assumes that the `conda` command is available on the system and that the user has permission to run it.
*   The returned string may be truncated if it exceeds the maximum allowed size by the system.

---

## `get_gpu_info()`

**Description:** This function retrieves information about NVIDIA GPUs present on the system. If an NVIDIA GPU is detected, it runs the `nvidia-smi` command and returns its output as a string. Otherwise, it returns a message indicating that no NVIDIA GPU is available.

### Returns

A string containing the output of the `nvidia-smi` command if an NVIDIA GPU is detected; otherwise, a message indicating that no NVIDIA GPU is available.

### Notes

*   This function assumes that the `nvidia-smi` command is available on the system and that the user has permission to run it.
*   The returned string may be truncated if it exceeds the maximum allowed size by the system.

---

## `get_python_info()`

**Description:** This function retrieves the version of Python currently installed on the system. It uses the `platform.python_version()` function to determine the version.

### Returns

A string representing the version of Python.

### Notes

*   This function may return a version in the format 'X.X.Y' or 'X.X', depending on the version of the platform library being used.
*   The returned version is specific to the Python interpreter being run, not necessarily the default Python interpreter for the system.

---

## `get_samples_per_class(X, y)`

**Description:** No description available.

---

## `get_system_info()`

**Description:** This function retrieves information about the operating system and hardware configuration of the system. It uses various functions from the `platform` and `psutil` libraries to gather information about the OS, number of CPUs, and memory usage.

### Returns

A dictionary containing three key-value pairs:

*   `'OS'`: The name of the operating system (e.g., 'Windows', 'Linux', 'Darwin').
*   `'Number of CPUs'`: The total number of CPU cores available on the system.
*   `'Memory'`: A string representation of the current memory usage, including both physical and virtual memory.

### Notes

*   This function assumes that the necessary permissions are available to access information about the system hardware.
*   The returned dictionary is specific to the current Python interpreter being run, not necessarily the default Python interpreter for the system.

---

## `integrated_brier_score(survival_train, survival_test, estimate, times)`

**Description:** Compute the Integrated Brier Score (IBS) using scikit-learn's Brier score loss.

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

---

## `ipy_exit()`

**Description:** This function raises an exception to terminate the execution of a Jupyter Notebook or IPython environment. When raised, it creates an instance of `SystemExit` which is handled by Python's built-in exit mechanism.

---

## `is_hyperparameter_tuning_suitable(data, outcome_var, train_size=train_size_perc, n_splits_outer=cv_folds, n_splits_inner=cv_folds_hptuning)`

**Description:** Checks whether hyperparameter tuning is suitable for a given dataset.

## Parameters:
    data (pandas DataFrame): The input dataset.
    outcome_var (str): The name of the column containing the target variable.
    train_size (float or int, optional): The proportion of samples to use for training. Defaults to 0.7.
    n_splits_outer (int, optional): The number of folds for outer cross-validation. Defaults to 5.
    n_splits_inner (int, optional): The number of folds for inner cross-validation. Defaults to 5.

## Returns
    None

---

## `parallel_compute_shap_surv(data, i)`

**Description:** Computes the SHAP values for a single patient or sample using parallel computing.
This function is used to compute the SHAP (SHapley Additive exPlanations) values for a single patient or sample in parallel, using the `parallel_compute_shap_surv` approach. It takes a pandas DataFrame `data` and an integer index `i` as input, and returns an object of type `PredictSurvSHAP`, which contains the computed SHAP values.

The function uses the `Parallel` class from the `loky` library to run multiple iterations of the computation in parallel, taking advantage of multiple CPU cores for speedup. The number of CPU cores used is controlled by the `n_cpu_for_tuning` variable, which can be adjusted depending on the system's capabilities and the size of the dataset.

The function is then applied to a range of indices using a list comprehension, and the resulting objects are collected into a list called `survshaps`. This approach allows for efficient computation of SHAP values for all samples in the test set.

Parameters:
    data (pandas DataFrame): The input data.
    i (int): The index of the patient or sample to compute SHAP values for.

## Returns
    PredictSurvSHAP: An object containing the computed SHAP values.

---

## `permute_feature_pairs(model, X, y, pairs, n_repeats, random_state, scoring, n_jobs)`

**Description:** Computes the permutation importance of pairs of features in a given model.This function computes the permutation importance of pairs of features in a given machine learning model. It takes as input:

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

---

## `PFI_median_wrap(PFI_folds)`

**Description:** Computes the median importance across all folds and normalizes the importance values.

## Parameters:
PFI_folds (list of feature importance values in folds): List of DataFrames where each DataFrame contains 'Feature' and 'Importance' columns.

## Returns
pd.DataFrame: DataFrame with 'Feature' and normalized 'Importance' sorted by importance.

---

## `plot_confusion_matrix_for_clusters(X, y, cluster_info, model, shap_values, top_n)`

**Description:** Plots a confusion matrix for each cluster in the test data and visualizes the feature importance using SHAP values.

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

---

## `plot_PFI(PFI_folds, X, model_name)`

**Description:** Plot permutation-based feature importances (PFI) from multiple folds using a strip plot.

## Parameters:
- PFI_folds (list of DataFrames): List where each DataFrame contains 'Feature' and 'Importance' columns for each fold.
- X (DataFrame): DataFrame used to determine the number of features for plot sizing.
- model_name (str): A string representing the name of the model or experiment, used for naming the output files.

## Returns:
- Saves the plot with filenames including the model_name parameter and displays it.

---

## `plot_samples_per_class_per_fold(samples_per_class_per_fold)`

**Description:** Plots a heatmap of the samples per class per fold.

## Parameters:
    samples_per_class_per_fold (numpy array): A 3D array where each element represents the number of samples for a particular class and fold combination.

## Returns
    None

---

## `plot_survshap_detailed(shap_results, top_n=10, sample_percentage=100)`

**Description:** Optimized function to plot SHAP values over time for the top N features on separate subplots,
with an option to randomly sample a percentage of the data for each feature.

Parameters:
shap_results: List of SHAP results for each sample.
top_n: The number of top features to plot based on mean of max absolute SHAP values.
sample_percentage: Percentage of samples randomly selected to be displayed on the plots (0 < sample_percentage <= 100).

---

## `plot_TFI(X, tree_FI, model_name)`

**Description:** Plots the tree-based feature importances for a given model.

## Parameters:
X (pd.DataFrame): The training data used for feature names.
tree_FI (list of pd.Series): List of feature importance scores for tree-based feature importance from each fold.
model_name (str): Name of the model to use in the plot title and filenames.

## Returns
None: Displays and saves the plot.

---

## `save_notebook()`

**Description:** This function saves the current state of a Jupyter Notebook to disk, effectively pausing the notebook's execution and preserving its environment, code, and data. The `IPython.notebook.save_checkpoint()` function is used internally by Jupyter to perform this task.

### Notes

*   When this function is called, it will save the notebook's current state to a file in the format `.ipynb`, which can be loaded later into the same or another instance of Jupyter Notebook.
*   This function does not execute any code; it simply saves the current state of the notebook and returns control to the caller.

By calling this function, you can:

*   Pause an ongoing computation and resume it later without losing its progress
*   Save a snapshot of your work for reference or sharing with others
*   Ensure that your changes are persisted even if the notebook is terminated unexpectedly

However, note that saving a notebook will also save any unsaved changes to the cell contents, which may not be what you want in all cases. To avoid this, consider using the `save` method explicitly.

---

## `set_parameters(n_rows, n_cols, class_proportion)`

**Description:** Sets the parameters for different machine learning classifiers based on the given dataset characteristics.

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

---

## `set_params_surv_models(n_rows)`

**Description:** Sets the parameters for different survival models based on the given dataset characteristics.

Parameters:
    n_rows (int): The number of rows in the dataset.

## Returns
    A dictionary containing the parameters for each survival model, including:
        - Random Survival Forest (RSF)
        - Cox proportional hazards regression (CoxPH)
        - Cox proportional hazards regression with L1 regularization (Coxnet)

Note that these functions assume hyperparameter tuning is not done and set default values based on common practices in survival analysis.

---

## `shap_summary_plot(shap_values, data, model_name)`

**Description:** Generates and saves a SHAP summary plot based on provided SHAP values and data from cross validation

Parameters:
- shap_values: concatenated list of SHAP values arrays from different folds
- data: DataFrames (trainset or testset)
- model_name: Name of the model (e.g., "CB" for CatBoost)

## Returns
- None: Saves the plot as a .tif file and displays it.

---

## `shorten_column_names(df, max_length)`

**Description:** Shortens column names in a pandas DataFrame to fit within a maximum length.

Parameters:
    df (pandas.DataFrame): The input DataFrame containing the columns to be shortened.
    max_length (int): The maximum allowed length for each column name.

## Returns
    list: A list of shortened column names.

---

## `shorten_data_dictionary(data_dict, max_length)`

**Description:** Shortens values in a dictionary to fit within a specified maximum length.

This function replaces long values with shorter versions by truncating them and adding an ellipsis ('...') if necessary.
If two values have the same length after truncation, it appends a numeric suffix (e.g., 'value_1', 'value_2') to make them unique.

## Parameters:
    data_dict (dict): The input dictionary containing the keys and values to be shortened.
    max_length (int): The maximum allowed length for each value in the dictionary.

## Returns
    dict: A new dictionary with shortened values.

---

## `subsample_iqr_test(shap_values, num_subsamples=1000, threshold=0, confidence_level=0.95, random_seed=None)`

**Description:** Perform subsampling and check if the IQR crosses zero in the SHAP values.

Parameters:
- shap_values: Array of SHAP values for a given feature
- num_subsamples: Number of subsamples to generate
- threshold: Threshold to determine significance (default None means use zero)
- confidence_level: Threshold for determining significance (default 95%)
- random_seed: Seed for reproducibility of random sampling

## Returns
- proportion_crossing_zero: The proportion of subsamples where the IQR crosses zero

---

