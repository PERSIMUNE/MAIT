# MAIT 1.0.0 Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Workflow overview](#workflow-overview)
3. [Load data and libraries](#load-data-and-libraries)
4. [Data preparation](#data-preparation)
5. [Data split](#data-split)
6. [Feature selection and association analysis](#feature-selection-and-association-analysis)
7. [Sample size and data split assessment](#sample-size-and-data-split-assessment)
8. [Data overview](#data-overview)
9. [Data imputation](#data-imputation)
10. [Other selective operations on the Data](#other-selective-operations-on-the-data)
11. [Visual inspection](#visual-inspection)
12. [Model initialization](#model-initialization)
13. [Binary model evaluation](#binary-model-evaluation)
14. [Cross validation](#cross-validation)
16. [Stopping condition](#stopping-condition)
17. [Prediction block](#prediction-block)
18. [Model interpretation](#model-interpretation)
19. [Decision curve analysis](#decision-curve-analysis)
20. [Model calibration and conformal predictions](#model-calibration-and-conformal-predictions)
21. [Survival models](#survival-models)
22. [Regression models](#regression-models)
23. [Report the Python environment](#report-the-python-environment)
24. [Save the executed pipeline](#save-the-executed-pipeline)

## Introduction
This manual provides a detailed description of the methods and workflow of MAIT. This pipeline is designed to facilitate the end-to-end process of developing, evaluating, and deploying machine learning models for tabular data, focusing on binary classification but also supporting survival and regression models. The pipeline is implemented in Python using Jupyter Notebooks. This manual offers detailed instructions on using MAIT. For a more intuitive understanding of the pipeline, we recommend exploring the tutorials available on our GitHub page. Additionally, MAIT is discussed in a research paper for further insight.

To navigate the pipeline more easily, you can use the "Outline" feature in VS Code. Alternatively, you can search for specific code segments related to the conditions described in "Load Data, Libraries, and Set Parameters."

## Workflow Overview
The pipeline consists of scripts and functions to first prepare the datasets for binary classification followed by training, evaluation, and interpretation of those binary models. There are some optional operations that can be selected to be performed by the user. In addition, it is possible to develop survival and regression models when we approach the ending parts of the pipeline. All this is explained as follows.

## Load Data and Libraries
Load the necessary libraries and set parameters for the pipeline. Define the data file location, the variables to be used, and the computational resources (e.g., GPU and CPU).
### User-Defined Parameters:
1. **Data Loading and Processing:**
   - Specify the dataset to be loaded: `mydata = pd.read_csv("combined_data_Azithromycin.csv")`.
   - Set the name of the outcome variable: `outcome_var = "azm_sr"`.

2. **Model Selection and Configuration:**
   - Choose the models to include in the analysis: `models_to_include = [...]`.
   - Specify the number of features to select using feature selection: `num_features_sel = 30`.
   - Define categorical features, if any: `cat_features = [...]`.
   - Set options for model design: `extra_option1 = [...]`.

3. **Survival Analysis (Optional):**
   - Specify parameters if conducting survival analysis: `survival_analysis = True`.
   - Define the time-to-event column and backup data for survival analysis.

4. **Regression Analysis (Optional):**
   - Specify parameters if conducting regression analysis: `regression_analysis = True`.
   - Set the regression outcome variable and configure demo options.

5. **Reporting and Visualization:**
   - Specify class labels for display: `class_labels_display = [...]`.
   - Set the main folder name to save results: `main_folder_name = 'results_Azithromycin'`.

### Default Parameters:
1. **Resource Allocation:**
   - Default settings for CPU allocation: `n_cpu_for_tuning = 20`, `n_cpu_model_training = 20`.
   - Default settings for GPU availability: `GPU_avail = False`. If GPU is available, you can change it to True so that LightGBM model runs on GPU.
   - Default settings for hyperparameter tuning and cross-validation: `hp_tuning = True`, `n_iter_hptuning = 10`, `cv_folds = 5`, etc.

2. **Data Manipulation:**
   - Default settings for data manipulation: `oversampling = False`, `scale_data = False`.
   - Default settings for feature filtering and handling: `filter_highly_mis_feats = True`, `shorten_feature_names = True`.

3. **Model Training and Validation:**
   - Default settings for model training and validation: `tun_score = "roc_auc"`, `test_only_best_cvmodel = True`.

4. **Feature Selection:**
   - Default settings for feature selection: `feat_sel = True`, `train_size_perc = 0.8`.

5. **Data Splitting:**
   - Default settings for data splitting: `data_split = True`, `already_split = False`.

6. **Visualization Options:**
   - Default settings for visualization formats: `fig_file_format = "tif"`.
     
## Data Preparation
Prepare the data by handling missing values, encoding categorical features, and defining feature types.

- **Using Data Dictionary:** Feature names to be displayed on figures.
- **Data Types:** Identify categorical and numerical features.
- **Shorten Feature Names:** Shorten column names in the dataFrame (e.g., train set) for feature names for easier handling.

This section manipulates the dataset and prepares it for analysis:

```python
# Example of data dictionary
data_dictionary = {
    "p_age": "patient age",
    "gender": "Patient gender",
    "cd4_counts": "CD4 counts",
    ...
}
```

**Outlier removal:**
Outliers and anomalies can negatively affect models. Another optional but useful functionality of the pipeline (set by remove_outliers = True) is to detect and remove anomalies (outliers) from the data. It is done using isolation forest algorithm. It includes these steps:
(1) Data Preparation:
Separates the input features (X) and the target variable (y) from the original dataset (mydata).
Encodes categorical features using one-hot encoding to convert them into numerical format, avoiding multicollinearity by dropping the first category.
(2) Handling Missing Values:
Imputes missing values in the combined dataset (X_combined), which includes both numerical and encoded categorical features, using the K-Nearest Neighbors (KNN) imputation method. The number of neighbors used for imputation is calculated based on the size of the dataset.
(3) Outlier Detection:
Initializes an [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) model to detect outliers.
Fits the model to the data and predicts outliers, labeling them as -1.
(4) Filtering Outliers:
Filters out rows marked as outliers from both the features (X) and the target variable (y).
Combines the cleaned features and target variable back into a single DataFrame (mydata).
(5) Final Cleanup:
Removes the 'outlier' column from the final DataFrame.

**Handling Missing Values:**
   Missing data can significantly impact model performance and introduce bias, making consistent preprocessing crucial. 
   - Drops rows where the outcome variable column contains NaN values: `mydata = mydata.dropna(subset=[outcome_var])`.
   - If `demo_configs` is enabled, randomly sets some entries to NaN and adds a categorical column for race.

In addition, it is possible to enable filtering highly missing data. It can be done as follows:
```python
exclude_highly_missing_columns = True # True to exclude features with high missingness
exclude_highly_missing_rows = True # True to exclude rows (samples) with high missingness
```
Filtering highly missing data is done first column-wise followed by row-wise. To address this, the following steps are undertaken:

**Filter Columns in `mydata`:** Identify and retain columns in `mydata` where the proportion of missing values is below a specified threshold. This step removes columns with excessive missing data that could skew analysis or model training.

**Apply Identified Columns to Other Datasets:** Ensure that `testset` and `extval_data` are aligned with `mydata` by selecting only the columns present in the filtered `mydata`. This maintains consistency across datasets, which is essential for reliable model evaluation and comparison.

**Filter Rows in All Datasets:** After aligning columns, filter out rows from all datasets where the proportion of missing values exceeds the threshold. This step ensures that all datasets have comparable completeness, supporting fair and accurate modeling.

By following this approach, all datasets are harmonized with respect to both columns and rows, ensuring consistency and reducing potential bias from missing data.

2. **Column Dropping:**
   - Drops specified columns from the dataset: `mydata.drop(columns=columns_to_drop, inplace=True)`.

3. **Data Type Conversion:**
   - Converts specified categorical features to the category data type: `mydata[cat_features] = mydata[cat_features].astype('category')`.
   - Converts categories to strings for each categorical column: `for col in cat_features: mydata[col] = mydata[col].astype(str).astype('category')`.

4. **Handling Empty Entries:**
   - Replaces empty entries with NaN: `mydata.replace(" ", np.nan, inplace=True)`.

5. **External Validation:**
   - If `ext_val_demo` is enabled, selects a subset of samples from the dataframe for external validation.
   - If `external_val` is enabled, removes specified columns from the external validation data, converts categorical features, and handles empty entries.

6. **Continuous Variables (Optional):**
   - If `specify_continuous_variables` is enabled, replaces non-numeric values with NaN and converts to float64.

7. **Merging Rare Categories (Optional):**
   - If `merged_rare_categories` is enabled, identifies and groups rare categories into a single category.
   - Adds a "Missing" category for missing values and converts categories to strings for mixed category features.

These steps ensure data consistency, handle missing values, and prepare the dataset for further analysis. See also run_pipeline.ipynb if you need to test many different parametrizations and data configurations efficiently.

## Data Split
Split the data into training and test sets with stratification based on the target variable to ensure balanced distribution of target classes in both training and test sets.

This part of the pipeline handles data splitting and statistical checks for training and test datasets. It configures whether to apply a stratified data split by the outcome variable (e.g., 80% training and 20% test data), with options to split by patient ID or use multiple stratification variables if specified. If the data is already split, it reads the pre-split datasets from CSV files; otherwise, the entire dataset is used for cross-validation. The code includes functions to check for statistical differences between the training and test sets: it uses the Mann-Whitney U test for numerical variables and the Chi-square test for categorical variables, generating results with test statistics and p-values. The statistical checks ensure the training and test sets are similar, aiding in the development of robust machine learning models.

## Feature Selection and Association Analysis
Select features using Minimum Redundancy Maximum Relevance (mRMR) and conduct association analyses. mRMR is one of the most popular algorithms for feature selection. For more information on its implementation see https://github.com/smazzanti/mrmr.

- **Feature Selection:** Select a predefined number of features based on folds of the training set.

MAIT includes a block of code to perform feature selection using a combination of techniques to identify the most relevant features for a predictive model. This is a step-by-step breakdown of the process:

1. **Data Preparation**:
   - **Feature and Outcome Separation**: The dataset `mydata` is divided into features (`X`) and the outcome variable (`y`).
   - **Column Identification**: Numerical columns and categorical columns are identified for separate processing.

2. **Cross-Validation Setup**:
   - **Stratified K-Fold Cross-Validation**: The data is split into `cv_folds` folds using `StratifiedKFold`, ensuring that each fold has a representative distribution of the outcome variable. This helps in avoiding data leakage and ensures that feature selection is robust across different subsets of data. By default k=5.

3. **Data Scaling** (Optional):
   - **Robust Scaling**: If `scale_data` is set to `True`, numerical features are scaled using `RobustScaler`, which is less sensitive to outliers compared to other scalers.

4. **Missing Value Imputation**:
   - **KNN Imputation**: Missing values in numerical features are imputed using `KNNImputer`. The number of neighbors used for imputation is determined by the square root of the number of training samples. This approach fills in missing values based on the values of the nearest neighbors, weighted by distance.

5. **Categorical Feature Encoding**:
   - **Label Encoding**: Categorical features are encoded into integer values using `LabelEncoder` to convert them into a format suitable for the mRMR feature selection method.

6. **Feature Selection**:
   - **Minimum Redundancy Maximum Relevance (mRMR)**: The `mrmr_classif` function is used to select the top `num_features_sel` features based on their relevance to the outcome variable and redundancy among features. This step identifies the most informative features while minimizing redundancy.

7. **Consolidation of Selected Features**:
   - **Intersection Across Folds**: The selected features from each fold are compared, and only the features that are consistently selected across all folds are retained. This ensures that the final set of features is robust and consistently important across different subsets of the data.

8. **Finalization**:
   - **Print Results**: The final list of selected features is printed for review.
   - **Dataset Adjustment**: The original dataset (`mydata`) and optionally the external validation dataset (`extval_data`) are updated to include only the selected features along with the outcome variable.

By using this approach we can get the selected features that are both relevant to the outcome variable and stable across different data splits, making the model more likely to be generalizable and reliable.

- **Association Analyses:** Use Spearman, point-biserial correlation, and mutual information for data exploration. This is only used for data exploration and not for the subsequent machine learning analyses.

- Below is a summary for the heatmap plot generation based on Spearman correlation of features:

This series of steps encompasses the preprocessing and visualization of data. Initially, the process involves imputation, where missing values are filled with the median for each class, followed by the imputation of any remaining NaN values with the median of the entire column. Subsequently, categorical features are transformed into binary format through one-hot encoding. Following this, the Spearman rank-order correlation matrix is computed to assess relationships between variables. In handling missing values within this correlation matrix, pairs of features with NaN correlation values are identified and replaced with 0. Visualization is then conducted through the creation of a clustermap using the seaborn library, allowing for an intuitive representation of the correlation matrix. Adjustments are made for figure size, color mapping, and hiding the upper triangle to enhance clarity. Further adjustments are made to the grid lines and axes, including hiding the x-axis dendrogram for a cleaner presentation. Finally, the clustermap plot is saved in both SVG and TIFF formats for future reference, and then displayed for immediate interpretation.

- Below is a summary of the point-biserial correlation:

This code conducts feature selection based on point biserial correlation against a target variable. It involves data preparation, subsampling to generate 1000 bootstrap samples, calculation of correlation coefficients for each feature across subsamples, quantile calculation to identify significant features, and filtering to create a DataFrame with only significant features.

Subsequently, a plot is generated to visualize the median and quantile correlation coefficients for features. It involves color coding for significance, sorting the DataFrame by median correlation coefficient, defining the plot size, plotting median correlation coefficients with significant features marked in different colors, adding error bars representing the interquartile range, customizing plot elements, and displaying the plot.

- And finally a brief explanation about the mutual information method for association analyses:

Initially, the dataset is copied, and the target variable is converted to numerical format. Then, 1000 subsamples of the dataset are generated in parallel. Mutual information is calculated for each variable against the target within each subsample. Afterward, a DataFrame containing mutual information values is created. Significant features are identified based on quantiles of mutual information, and the original DataFrame is filtered to include only these features. Finally, a plot visualizes the median and quantile mutual information for features based on random subsamples of the development set across 1000 iterations. The plot distinguishes significant and non-significant features with different colors and includes error bars representing the interquartile range of mutual information values for all features.


## Sample Size and Data Split Assessment
Estimate and visualize the number of samples per class for the cross-validation scheme. This visualization is done to present how many samples will be available for training and hyperparameter tuning. 

- **Statistical Tests:** Compare the difference in variables between the training and test set.

## Data Overview
Visualize the results of the association analyses and the distribution of values in all features.

- **Missingness:** Report statistical information about missing data.
- **Training and Test Sets:** Overview of the types of variables.

## Data Imputation
Impute missing values using KNN for continuous features and one-hot encoding for categorical features.

- **Imputation and Encoding:** Handle imputation and encoding fold-wise during cross-validation.
- **Model-Specific Handling:** Note that some models (e.g., CatBoost, LightGBM) handle missingness algorithmically.

Here we apply k-nearest neighbors (KNN) algorithm to impute missing values in continuous variables. This is done in fold-wise as in cross validation so that the informaiton from one fold does not leak to other folds. This means that the training data is split to a number of folds as the same as in cross validation and then the imputation is performed on the fold under test, for all folds. then they are merged back to recreate the training set with imputation. The test set and external datasets are also imputed based on the KNN algorithm.

## Other Selective Operations on the Data
Scale the data using robust scaling and handle class imbalance using oversampling techniques.

- **Data Scaling:** Use robust scaling method. Read more here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
- **Class Imbalance:** Apply oversampling to handle minority class imbalance. Read more here: https://imbalanced-learn.org/stable/over_sampling.html

## Visual Inspection
Evaluate binary classification models and generate receiver operating characteristics (ROC) curves, precision-recall (PR) curves, and confusion matrices.

- **ROC and PR Curves:** Visualize model performance.
- **Confusion Matrix:** Assess classification accuracy.

Binary classification models can be evaluated by visual inspection of ROC and PR curves as well as confusion matrices. MAIT visualize those for binary classification models at each time a model is evaluated.
You can find more information about these methods and their implementations from here:
- ROC curve:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
- Precision-recall curve:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
- Confusion matrix:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

## Model Initialization
Initialize various binary classification models, from logistic regression to tree-based ensemble models.

- **Model Selection:** Choose from a selection of 7 different models. If you want to save time, you can remove some of the models from the list of included models. QLattice is the slowest one to train.
- **Sampling Weights:** Set weights based on class balance in the training set. This is done in the pipeline to take care of class imbalance issues.
- **Parameter Grid:** Define parameters for random search in hyperparameter tuning. The search space is pre-defined but you can change them for each model.

An overview of the various binary classification models included in MAIT, including their hyperparameters, interpretation methods, and a brief description of each model.

### Models Overview

| Model                   | Hyperparameters                                                                                                                                           | Interpretation Method                                      | Description                                                                                   |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **QLattice**            | - `n_epochs`: [50, 100, 150]<br>- `max_complexity`: [5, 10, 15]                                                                                                     | Model block diagram and closed-form mathematical expression| The QLattice, integrated into the Feyn Python library, represents a cutting-edge approach to supervised machine learning known as symbolic regression. It specializes in identifying the most suitable mathematical models to describe complex datasets. Through an iterative process of training, the QLattice prioritizes simplicity while maintaining high performance. [More information](https://docs.abzu.ai/docs/guides/getting_started/qlattice) |
| **Gaussian Naive Bayes**| Not applicable (no hyperparameters to tune)                                                                                                                | Feature permutation, SHAP                                  | Gaussian Naive Bayes (GaussianNB) is a classification algorithm implemented in Python's scikit-learn library. It assumes that the likelihood of features follows a Gaussian distribution. The algorithm estimates parameters using maximum likelihood. In practice, GaussianNB is commonly used for classification tasks when dealing with continuous data. [Read more here](https://scikit-learn.org/stable/modules/naive_bayes.html) |
| **RandomForestClassifier** | - `n_estimators`: randint(100, min(1000, 2*n_rows))<br>- `max_depth`: [3, 4, 5, 6, 7, 8, 9, 10]<br>- `min_samples_split`: [2, 5, 10, int(15 + n_rows/1000)]<br>- `min_samples_leaf`: [1, 2, 4, int(5 + n_rows/1000)]<br>- `max_features`: ['sqrt', 'log2', None] | Feature permutation, SHAP, Tree-based feature importance    | The RandomForestClassifier, part of the `sklearn.ensemble` module in scikit-learn, is a versatile and powerful tool for classification tasks. It operates as a meta estimator that fits multiple decision tree classifiers on various sub-samples of the dataset, using averaging to enhance predictive accuracy and mitigate overfitting. By default, the classifier uses bootstrap sampling (`bootstrap=True`), and each tree is built using a random subset of features (`max_features='sqrt'`).<br><br>Key parameters include:<br>- `n_estimators`: Number of trees in the forest.<br>- `criterion`: Function to measure the quality of a split (`'gini'` or `'entropy'`).<br>- `max_depth`: Maximum depth of the trees.<br>- `min_samples_split`: Minimum number of samples required to split an internal node.<br>- `min_samples_leaf`: Minimum number of samples required to be at a leaf node.<br>- `class_weight`: Adjusts weights inversely proportional to class frequencies to handle imbalanced datasets.<br><br>The RandomForestClassifier is highly customizable, allowing for fine-tuning to suit specific datasets and classification challenges. It provides robust performance, especially in scenarios where feature interactions are complex or when the dataset contains a mix of categorical and numerical features. [Read more here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| **LightGBM**            | - `num_leaves`: randint(6, min(50, 2*n_rows))<br>- `min_child_samples`: randint(4, min(100, n_rows))<br>- `min_child_weight`: [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]<br>- `subsample`: uniform(loc=max(0.2, 0.5 - class_proportion/2), scale=min(0.8, 0.5 + class_proportion/2))<br>- `colsample_bytree`: uniform(loc=0.4, scale=0.6)<br>- `reg_alpha`: [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]<br>- `reg_lambda`: [0, 1e-1, 1, 5, 10, 20, 50, 100]<br>- `n_estimators`: randint(50, min(1000, 2*n_rows))<br>- `max_depth`: [3, 4, 5, 6, 7, 8, 9, 10]        | Feature permutation, SHAP, Tree-based feature importance    | LightGBM represents an open-source, distributed, and high-performance gradient boosting framework, engineered by Microsoft, to tackle machine learning challenges with precision and efficiency. It operates on decision trees, finely tuned to optimize model efficiency while minimizing memory consumption. A key innovation is the Gradient-based One-Side Sampling (GOSS) method, which intelligently retains instances with significant gradients during training, thereby optimizing memory usage and training duration. Additionally, LightGBM employs histogram-based algorithms for rapid and resource-efficient tree construction. These advanced techniques, alongside optimizations such as leaf-wise tree growth and streamlined data storage formats, collectively contribute to LightGBM's remarkable efficiency and competitive edge in the realm of gradient boosting frameworks. [Read more here](https://lightgbm.readthedocs.io/en/stable/) |
| **CatBoost**            | - `learning_rate`: np.logspace(-3, 0, num=100)<br>- `depth`: [3, 4, 5, 6, 7, 8, 9, 10]<br>- `l2_leaf_reg`: np.logspace(-1, 3, num=100)<br>- `iterations`: randint(100, min(1000, 2*n_rows))<br>- `subsample`: np.linspace(0.1, 1, 10)<br>- `random_strength`: np.linspace(0, 10, 100)             | Feature permutation, SHAP, Tree-based feature importance    | CatBoost is a supervised machine learning method utilized for classification and regression tasks, particularly useful for handling categorical data without the need for extensive preprocessing. Employing gradient boosting, CatBoost iteratively constructs decision trees to refine predictions, achieving enhanced accuracy over time. Notably, CatBoost employs ordered encoding to effectively handle categorical features, utilizing target statistics from all rows to inform encoding decisions. Additionally, it introduces symmetric trees, ensuring uniformity in split conditions at each depth level. Compared to similar methods like XGBoost, CatBoost have often demonstrates superior performance across datasets of varying sizes, retaining key features such as cross-validation, regularization, and support for missing values. [Read more here](https://catboost.ai/docs/features/categorical-features) |
| **LogisticRegression**  | - `C`: [0.01, 0.1, 1, 10, 100]<br>- `max_iter`: [500, 1000, 2000]<br>- `tol`: [1e-3, 1e-4, 1e-5]                                                           | Feature permutation, SHAP                                  | Logistic Regression is a linear model for binary classification that uses the logistic function. This model is widely used for its simplicity and effectiveness in binary classification tasks. |
| **HistGBC**             | - `max_iter`: randint(100, min(1000, 2*n_rows))<br>- `validation_fraction`: uniform(0.1, 0.3)<br>- `learning_rate`: uniform(0.01, 0.2)<br>- `max_depth`: [3, 4, 5, 6, 7, 8, 9, 10]<br>- `min_samples_leaf`: randint(1, min(5, n_rows))<br>- `max_leaf_nodes`: randint(10, 100)<br>- `l2_regularization`: uniform(0.01, 0.2)       | Feature permutation, SHAP, Tree-based feature importance    | The HistGradientBoostingClassifier, part of the scikit-learn library, offers a histogram-based approach to gradient boosting for classification tasks. Notably, it exhibits significantly faster performance on large datasets (with n_samples >= 10,000) compared to the traditional GradientBoostingClassifier. The implementation of HistGradientBoostingClassifier is inspired by LightGBM and offers various parameters for customization, such as learning rate, maximum depth of trees, and early stopping criteria. This classifier is an excellent choice for classification tasks with large datasets, providing both speed and accuracy. [Read more here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) |

- `randint` and `uniform` represent ranges from which hyperparameter values are drawn.
- `np.logspace` and `np.linspace` are used to generate evenly spaced values on a log scale and linear scale, respectively.

## Binary Model Evaluation
### Function to Calculate Evaluation Metrics

This function, `calculate_metrics`, computes various evaluation metrics based on the predictions and probabilities generated by a machine learning model.

### Function Definition

```python
def calculate_metrics(y_true, y_pred, y_pred_proba):
```

### Parameters
- **y_true**: True labels.
- **y_pred**: Predicted labels.
- **y_pred_proba**: Predicted probabilities for the positive class.

### Measures computed

1. **Confusion Matrix (CM)**: Computes the confusion matrix to extract true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP). For further information, refer to [Wikipedia - Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix).

2. **Positive Predictive Value (PPV)** (Precision): Measures the proportion of true positive predictions among all positive predictions.
   
   <img src="https://latex.codecogs.com/svg.image?PPV=TP/(TP&plus;FP)" title="PPV=TP/(TP+FP)" />

3. **Negative Predictive Value (NPV)**: Measures the proportion of true negative predictions among all negative predictions.
   
   <img src="https://latex.codecogs.com/svg.image?NPV=TN/(TN&plus;FN)" title="NPV=TN/(TN+FN)" />

4. **True Positive Rate (Sensitivity)**: Measures the proportion of true positive predictions among all actual positive instances.
   
   <img src="https://latex.codecogs.com/svg.image?TPR=TP/(TP&plus;FN)" title="TPR=TP/(TP+FN)" />

5. **True Negative Rate (Specificity)**: Measures the proportion of true negative predictions among all actual negative instances.
   
   <img src="https://latex.codecogs.com/svg.image?TNR=TN/(TN&plus;FP)" title="TNR=TN/(TN+FP)" />

6. **Balanced Accuracy**: Calculates the average of Sensitivity and Specificity to provide a balanced measure of model performance.
   
   <img src="https://latex.codecogs.com/svg.image?BlanacedAccuracy=(Sensitivity&plus;Specificity)/2" title="BlanacedAccuracy=(Sensitivity+Specificity)/2" />

7. **Matthews Correlation Coefficient (MCC)**: Computes the correlation coefficient between true and predicted binary classifications.
   
    <img src="https://latex.codecogs.com/svg.image?MCC=(TP\times&space;TN-FP\times&space;FN)/\sqrt{(TP&plus;FP)(TP&plus;FN)(TN&plus;FP)(TN&plus;FN)}" title="MCC=(TP\times TN-FP\times FN)/\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}" />

8. **AUC Score**: Computes the Area Under the Receiver Operating Characteristic curve, indicating model discriminative ability across different thresholds.

9. **Precision-Recall AUC Score**: Computes the Area Under the Precision-Recall curve, providing insight into model performance at different recall levels.

10. **Brier Score**: Measures the mean squared difference between predicted probabilities and actual outcomes. https://en.wikipedia.org/wiki/Brier_score

11. **F1 Score**: Computes the harmonic mean of Precision and Recall, providing a balance between them.

    <img src="https://latex.codecogs.com/svg.image?F1&space;score=2TP/(2TP&plus;FP&plus;FN)" title="F1 score=2TP/(2TP+FP+FN)" />

The function returns a dictionary containing the computed measures. To rigorously evaluate binary classification models, the default criterion involves assessing the mean values of ROC-AUC, PRAUC, and MCC across cross-validation folds. ROC-AUC and PRAUC capture the model's discrimination capability across various probability thresholds, crucial for imbalanced datasets, while MCC consolidates information from the confusion matrix to provide a balanced assessment of true positives, true negatives, false positives, and false negatives. By collectively considering these metrics, we ensure a comprehensive evaluation of the model's predictive performance and robustness, facilitating informed model selection decisions. 

In summary, this function is essential for evaluating the performance of binary classification models. The calculated measures provide insights into the model's predictive capabilities and generalization ability.

## Cross validation
Perform cross-validation, report model performance, and visualize results. Also, conduct hyperparameter tuning and feature importance analysis.

- **Model Performance:** Evaluate performance based on multiple measures.
- **Feature Importance:** Analyze using SHAP, feature permutation, and tree-based methods.
- **Optimal Threshold:** Determine optimal probability threshold for each model.

Note: There is also a function for cross validation for survival models (`cross_validate_surv_model`).

### Function Definition

```python
def cross_validate_model(model_class, X, y, sample_weights=None, n_splits=cv_folds, random_state=SEED, measures=None,
                         use_default_threshold=False, **model_params):
```

#### Parameters
- **model_class**: The class of the model to be cross-validated.
- **X**: Features dataset.
- **y**: Labels dataset.
- **sample_weights**: Weights for the samples (default: None).
- **n_splits**: Number of folds for cross-validation (default: `cv_folds`).
- **random_state**: Random seed for reproducibility (default: `SEED`).
- **measures**: Performance measures to evaluate (default: list of various metrics).
- **use_default_threshold**: Boolean to use the default threshold (default: False).
- **model_params**: Additional model parameters.

#### Initial Setup

```python
n_repeats = n_rep_feature_permutation
if measures is None:
    measures = ['PPV', 'NPV', 'Sensitivity', 'Specificity', 'Balanced Accuracy', 'MCC', 'ROCAUC', 'PRAUC', 'Brier Score', 'F1 Score']

fold_results = pd.DataFrame()
fold_results_plt = pd.DataFrame()
aggregated_thr = np.array([])
aggregated_predictions = np.array([])
aggregated_labels = np.array([])
skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
feature_importance_list = []
treebased_feature_importance_list = []
shap_values_list = []
```
- **n_repeats**: Number of repetitions for feature permutation importance calculation.
- **measures**: List of performance metrics to evaluate.
- **fold_results**: DataFrame to store results for each fold.
- **aggregated_thr**: Aggregated list of estimated optimal thresholds.
- **skf**: Stratified K-Folds cross-validator to ensure balanced folds.
- **feature_importance_list**: List to store feature importances from permutation.
- **shap_values_list**: List to store SHAP values for interpretability.

#### Cross-Validation Loop

```python
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    sample_weights_fold = sample_weights[train_index] if sample_weights is not None else None
```
- Splitting data into training and testing sets for each fold in the cross-validation process.

#### Model Training and Evaluation

Depending on the model class, different models are trained and evaluated:

##### RandomForestClassifier

```python
# Check if the model class is RandomForestClassifier
if model_class == RandomForestClassifier:
    
    # Initialize the RandomForestClassifier with specified parameters
    rf_model = RandomForestClassifier(random_state=SEED, n_jobs=n_cpu_model_training, **rf_params)
    
    if hp_tuning:
        # Perform hyperparameter tuning using RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=rf_model, 
            param_distributions= rf_param_dist,  # Hyperparameter distribution
            n_iter= n_iter_hptuning,             # Number of iterations for the search
            scoring= custom_scorer,               # Scoring metric for evaluation (by default it is the mean of PRAUC and ROCAUC)
            cv= cv_folds_hptuning,              # Number of cross-validation folds
            refit= tun_scoring_single,          # Metric to use for selecting the best model
            random_state= SEED,                 # Random state for reproducibility
            verbose=0,                         # Verbosity level
            n_jobs=n_cpu_for_tuning)           # Number of jobs for parallel processing
        
        # Fit the RandomizedSearchCV object to the training data
        random_search.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
        
        # Retrieve the best hyperparameters from the search
        best_params = random_search.best_params_
        
        # Reinitialize the RandomForestClassifier with the best parameters
        rf_model = RandomForestClassifier(random_state=SEED, n_jobs=n_cpu_model_training, **best_params)
    
    # Train the RandomForestClassifier on the training data
    rf_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
    
    # Make predictions and get probability estimates for the positive class
    predictions_proba = rf_model.predict_proba(X_test_fold)[:, 1]
    
    # Extract feature importances from the trained model
    treebased_feature_importance = rf_model.feature_importances_
    treebased_feature_importance_list.append(treebased_feature_importance)
    
    # Compute permutation importance to assess feature importance
    perm_result = permutation_importance(
        rf_model, X_test_fold, y_test_fold, n_repeats=n_repeats, random_state=random_state, n_jobs=n_cpu_model_training, scoring="roc_auc")
    feature_importance = perm_result.importances_mean
    feature_importance_df = pd.DataFrame({"Feature": X_train_fold.columns, "Importance": feature_importance}).sort_values(by="Importance", ascending=False)
    feature_importance_list.append(feature_importance_df)
    
    # Create SHAP explainer and compute SHAP values for model interpretability
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_fold)
    shap_values_list.append(shap_values)
```

- **RandomForestClassifier**: This function initializes and trains a RandomForestClassifier model, optionally performs hyperparameter tuning using RandomizedSearchCV, and evaluates its performance using feature importance metrics and SHAP values.
- **Feature Importance**: Uses both tree-based and permutation importance to identify key features.
- **SHAP Values**: Computes SHAP values for interpretability.

##### QLattice

QLattice is a powerful model but is different from other models that are available in the pipeline and one of its significant difference is that it is interpreted using the model block diagram visualization rather than by SHAP. Thus the pipeline does not apply SHAP method when QLattice is the selected model.

```python
elif model_class == 'QLattice':
    X_train_fold_ql = X_train_fold.copy()
    X_train_fold_ql[outcome_var] = y_train_fold.values
    if hp_tuning:
        from joblib import Parallel, delayed
        best_composite_score = 0
        best_parameters = {'n_epochs': 50, 'max_complexity': 10}
```

- **QLattice**: For QLattice models, the training data is prepared, and hyperparameter tuning is performed if specified.

MAIT allows using a custom metric with hyperparameter tuning:

1. **Define Custom Metric:**
   - Create a function `combined_metric` to calculate the average of AUC and PR AUC:
     ```python
     def combined_metric(y_true, y_pred_proba):
         roc_auc = roc_auc_score(y_true, y_pred_proba)
         pr_auc = average_precision_score(y_true, y_pred_proba)
         return (roc_auc + pr_auc) / 2  # Mean of ROCAUC and PRAUC
     ```
   - Define the custom scorer using this metric:
     ```python
     custom_scorer = make_scorer(combined_metric, needs_proba=True)
     ```

2. **Select Single Metric (Optional):**
   - If you prefer to use only one metric for evaluation, set `use_single_metric` to `True` and choose either "ROCAUC" or "PRAUC":
     ```python
     use_single_metric = False  # Change to True if using a single metric
     if use_single_metric:
         single_score = "ROCAUC"  # Options: "ROCAUC" or "PRAUC"
         if single_score == "ROCAUC":
             custom_scorer = make_scorer(roc_auc_score, needs_proba=True)
         elif single_score == "PRAUC":
             custom_scorer = make_scorer(average_precision_score, needs_proba=True)
     ```
By following these steps, you can make sure that the model is evaluated and optimized according to your custom metric, and that the best hyperparameters are used to enhance the classifier's performance.

### Threshold Optimization Method
By default, probability threshold optimization is not done. When applying the threshold optimization, consider running the pipeline without doing so and comparing the results to avoid biased interpretation.
- **Initial Threshold**: Use a default threshold of 0.5 for the first fold.
- **Threshold Adjustment**:
  - For each fold, predict probabilities for the validation set.
  - Compute the median predicted probability for each class.
  - Calculate the midpoint of these medians and use it as the threshold for the next fold.
- **Sequential Application**: Continue this process, adjusting the threshold sequentially across folds.

This novel method is viable under certain conditions. On one hand, it adapts the threshold based on the model’s performance, potentially improving the balance between precision and recall dynamically. It captures how the model's predictive probabilities distribute over different folds, potentially leading to a more informed threshold. On the other hand, the threshold for each fold depends on the performance of previous folds, introducing a form of sequential dependency that could sligthly bias performance metrics.

### Model Uncertainty Reduction (MUR)
Model Uncertainty Reduction (MUR) is our novel technique applied post-cross-validation to enhance model reliability by filtering out predictions with high uncertainty. 

### Key Points:
1. **Objective**: To reduce model uncertainty while preserving a sufficient number of samples for robust evaluation.
2. **Method**:
   - **Thresholds and Percentiles**: MUR employs a grid search over different margins around prediction probabilities and SHAP percentiles.
   - **Filtering**: After cross-validation, samples are filtered out if their predicted probabilities or SHAP values fall within specified uncertainty margins.
   - **Sample Retention**: Ensures that the proportion of discarded samples does not exceed a predefined maximum percentage, balancing uncertainty reduction with sample retention.
3. **Implementation**:
   - **Model-Specific**: Applied to various models (e.g., HistGBC, RandomForest) to calculate SHAP values and filter predictions accordingly.
   - **Selection**: Post-filtering, the best model is selected based on evaluation metrics such as AUC, PR AUC, and MCC, while ensuring minimal sample loss.

MUR is used after cross-validation to refine the model by discarding less certain predictions, ultimately selecting the best-performing model with the highest confidence. The thresholds (margins) for SHAP percentile and the prediciton probabilities are then used to discard uncertain predictions on new samples (e.g., test set).

#### Additional sections

Other sections of the code perform similar operations for different model classes. Each model class has specific configurations and evaluation methods.

In sum, this function provides a comprehensive approach to cross-validation for various machine learning models. It supports model training, hyperparameter tuning, performance evaluation, feature importance calculation, and model interpretability using SHAP values. The function is designed to be flexible and applicable to multiple types of machine learning models.

## Stopping Condition
Define the stopping condition for the pipeline if there is no data split and only cross-validation is performed.

## Prediction Block
Train the selected model on the entire training set and evaluate it on the test set.

- **Model Training:** Train the final model using the full training set.
- **Test Evaluation:** Assess model performance on the test set (and external validation set if available).

## Model Interpretation
Analyze the selected model using SHAP for model interpretation, including SHAP values and plots.

- **SHAP Analysis:** Interpret model predictions and feature importance.
- **SHAP Plots:** Generate summary and decision plots. It includes SHAP values association with predicted probabilities, SHAP summary plot (works only for continuous variables), enhanced SHAP summary plot (custom-made function that can also handle categorical variables), SHAP plots only for correctly predicted samples, SHAP decision plot, SHAP clustering.

The best performing model (the selected model) is chosen based on the performance of the models on cross validation as the model with the highest mean of MCC, ROCAUC, and PRAUC (note that ROCAUC is also often written as AUC). This model may not necessarily have the best performance on the test set, especially if the models perform closely similar on the cross validation. Since most of the data is used in cross validation, the model that is chosen based on that is prefered to the best performing model based only on the test set.

For tree-based ensemble models, TreeExplainer is used to calculate the SHAP values, for Logistic Regression model LinearExplainer is used, and for Gaussian Naive Bayes model KernelExplainer is used. SHAP is not implemented for QLattice model as it has its specific approach for model interpretation.

### Statistical significance of features

Sometimes it is favorable to point out significant features, like statistical analysis, and here we so far had a list of most important (impactful in terms of SHAP values). SHAP summary gives an idea on both population-based importance and individual-based importance of features. To have more emphasize on population-based importance (global importance in explainable AI) we apply the following approach based on bootstrap testing.

The significance test is based on the subsampling method (with replication), where if the IQR crosses zero less than 5% of the time (95% confidence) via subsample_iqr_test function, the feature is marked as significant. The results will be depicted as boxplots with indication of significant features with light green color (as oppposed to light red color for non-significant features) and an "*" in front of the feature name via f_imp_shapboxplot function. This is similarly done for survival models.

Derivation and interpretation:

Data-driven threshold: By using the sum of absolute SHAP values and defining the threshold based on the 1st percentile, you're taking into account the overall contribution of each feature across all instances. Features with lower total contributions are compared against this data-derived threshold, rather than simply comparing them against zero.

Significance Test: For each feature, you conduct a subsampling test to see how often the IQR of the SHAP values crosses this threshold. If it crosses less than 5% of the time, the feature is considered significant and marked with an asterisk.

Note that the significance refers to population-wide (global) importance of the features, and for subsets of the studied population there might be variations in the importance of the features.

### SHAP decision plot description

The SHAP decision plot centers around the `explainer.expected_value` on the x-axis, with colored lines representing predictions for each observation. Moving upwards, these lines intersect the x-axis at the prediction specific to each observation, depicted in varying colors on a gradient scale. The plot integrates SHAP values for each feature, illustrating their contributions to the overall prediction relative to the model's baseline value. At the plot's bottom, observations converge at `explainer.expected_value`.

1. **Demonstrating feature effects:**
   - Visualizes the impact of multiple features on predictions and their individual contributions.

2. **Revealing interaction effects:**
   - shows how interactions between features influence predictions by incorporating SHAP values.

3. **Exploring feature effects across values:**
   - Enables exploration of feature effects by showcasing prediction variations across different feature values.

4. **Identifying outliers:**
   - Enables outlier detection by pinpointing observations deviating significantly from expected values or prediction trends.

5. **Understanding prediction paths:**
   - Facilitates the identification of common prediction patterns, offering insight into model behavior.

6. **Model comparison:**
   - Allows comparing predictions across multiple models.

### Feature interactions based on SHAP method

There is also a code chunk within the pipeline that generates a heatmap visualization representing the interaction between features using SHAP (SHapley Additive exPlanations) values. It is done once for all samples from the test set and once for each subset of the test set by their class from the outcome variable.

**Process Overview:**

1. **Interaction Matrix Calculation**:
   - **Pairwise SHAP Values**: For each pair of features, the interaction is assessed by summing their SHAP values across samples.
   - **Metrics Computed**: The script calculates median, minimum, and maximum values for each feature pair to quantify interactions.

2. **Data Visualization**:
   - **Heatmaps**: Three types of heatmaps are produced for each class:
     - **Median SHAP Values**: Displays the median interaction strength (i.e., median of pairwise SHAP values).
     - **Minimum SHAP Values**: Shows the minimum interaction observed (i.e.,  minimum of pairwise SHAP values).
     - **Maximum SHAP Values**: Highlights the maximum interaction observed (i.e., maximum of pairwise SHAP values).
   - **Box Plots**:
     - **All Feature Pairs**: Illustrates the distribution of interaction values for each feature pair, ordered by median interaction strength.
     - **Top and Bottom 10% Feature Pairs**: Focuses on the feature pairs within the top and bottom 10% of median SHAP values, revealing the most and least significant interactions.

Seaborn and Matplotlib are used to create heatmaps and box plots, with results saved as TIFF files for further analysis. These visualizations demonstrate the combined effects of feature pairs on model predictions that could be useful to detect interacting features.

### Feature interactions based on feature permutation method for feature pairs

This code chunk provides insight into the interaction effects between pairs of features in the machine learning model, helping identify which combinations of features contribute significantly to the model's performance.

- The `permute_feature_pairs` function calculates the permutation importances for pairs of features.
- It converts the binary target variable to numeric format and calculates the baseline score using AUC.
- For each pair of features, it shuffles their values multiple times and computes the change in AUC compared to the baseline. The average change in AUC is stored as the importance score for that feature pair.

- It generates all possible pairs of features from the input feature set.
- It computes the permutation importances for pairs of features using the defined function.
- The results are stored in a DataFrame, where each row represents a feature pair along with its importance score.
- The DataFrame is sorted based on importance in descending order and printed to display the importance of feature pairs.

### SHAP dependence plots

We also have a custom code within the pipeline that generates SHAP dependence plots for a selected machine learning model. The code calculates the median absolute SHAP values for each feature, sorts them, and determines the number of features to plot based on a predefined threshold (`top_n_f`). It initializes subplots to display the dependence plots and iterates over the top features. For each feature, the code handles both categorical and numerical data, ensuring any missing values are addressed. It then creates scatter plots of the SHAP values against feature values, adding a regression line to indicate the trend. Misclassified samples are marked distinctly with an 'X'. Correlation between the feature values and SHAP values is assessed using the Spearman correlation coefficient, and a corresponding p-value is calculated. These statistics are displayed in the plot titles, indicating whether the correlation is statistically significant. The plots are customized with color bars representing predicted probabilities. Finally, the layout is adjusted for clarity, and the plots are displayed using `plt.show()`.

### SHAP clustering

MAIT also includes a code chunk aimed to identify and analyze clusters of features and instances using SHAP (SHapley Additive exPlanations) values in the context of precision medicine. It does so by employing hierarchical clustering techniques. It includes the following steps:

1. **SHAP Values Preparation**:
   - The SHAP values are converted into a DataFrame with features as columns.

2. **Clustermap Visualization**:
   - A clustermap is generated to visualize clusters in both features and instances, providing an initial view of potential patterns.

3. **Feature Clustering**:
   - **Hierarchical Clustering**: Features (columns) are clustered into 3 groups using Agglomerative Clustering.
   - **Feature Grouping**: Features are grouped based on their cluster assignments.
   - **Top N Features**: The top clusters with the most features are identified.

4. **Instance Clustering**:
   - **Silhouette Score Calculation**: For each number of clusters (from 3 to 5), silhouette scores are computed to determine the optimal number of clusters for instances (rows).
   - **Optimal Clustering**: The best number of clusters is selected based on the highest silhouette score.

5. **Final Clustering and Output**:
   - **Hierarchical Clustering**: Instances are clustered into the optimal number of clusters.
   - **Instance Grouping**: Instances are grouped based on their cluster assignments.
   - **Top N Clusters**: The top N clusters with the most instances are identified. (N is determined by the Silhouette score)

6. **Model-specific Execution**:
   - The function `find_feature_clusters` is called based on the type of model (`selected_model`), which could be either from `sklearn` or other libraries like `catboost` or `lightgbm`.

7. **Results Display**:
   - The top clusters for features and instances are printed out.

This approach helps in understanding how different subgroups of features and patient instances behave differently with the model, potentially revealing high or low-risk clusters and offering valuable insights for personalized patient treatment strategies.

#### Plotting Confusion Matrix for Clusters

It is done by `plot_confusion_matrix_for_clusters` function that introduces the following enhancements:

1. **SHAP Values Handling**:
   - It includes SHAP values as an input to visualize feature importances specific to each cluster.

2. **Conditional SHAP Plotting**:
   - The function decides on the appropriate SHAP plot type based on the model and data characteristics:
     - For models like HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression, and GaussianNB, it uses the standard SHAP summary plot.
     - For models with categorical features (e.g., LGBM, CatBoost when X_test has categorical variables), it also uses a custom summary plot using categorical_shap_plot function as the original function from shap package for shap summary plot cannot display categories.

3. **Plot Adjustments**:
   - Confusion matrix plots are adjusted for figure size, font size, and annotations to ensure clarity.

#### Execution Based on Model Type

The function `plot_confusion_matrix_for_clusters` is called conditionally based on the type of selected model:
- **For sklearn Models**: It processes `X_test_OHE``.
- **For CatBoost and LightGBM Models**: It processes `X_test`.

In sum, this enhanced code provides a comprehensive approach to analyze and visualize the performance of a machine learning model within different clusters of data, leveraging confusion matrices and SHAP values to gain deeper insights into model behavior across diverse patient subgroups.

## Decision Curve Analysis
Compare the selected model against alternative approaches using decision curve analysis.

Net benefit of the model compared to random guessing, extreme cases, and an alternative method or model. Read more here: https://en.wikipedia.org/wiki/Decision_curve_analysis#:~:text=Decision%20curve%20analysis%20evaluates%20a,are%20positive%20are%20also%20plotted.
as an alternative model we here use logistic regression model but you can modify this or import prediction probabilities for the test samples from elsewhere.

### Cost-sensitive Model Evaluation

In cost-sensitive model evaluation, we incorporate weights into performance metrics to account for varying costs associated with true positive (TP) and false positive (FP) cases. Two key metrics are introduced:

- **Cost-sensitive Net Benefit:** This metric adjusts the traditional net benefit by applying weights to TP and FP cases. It is defined as:
  
<img src="https://latex.codecogs.com/svg.image?%5Ctext%7BNet%20Benefit%7D%20%3D%20%5Cfrac%7B(TP%20%5Ctimes%20w_%7BTP%7D%20-%20FP%20%5Ctimes%20w_%7BFP%7D%20%5Ctimes%20%5Cfrac%7BThreshold%7D%7B1%20-%20Threshold%7D)%7D%7BN%7D" />

### Cost-sensitive Decision Curve Analysis

Cost-sensitive Decision Curve Analysis (CDCA) is used to evaluate the cost-sensitive net benefit of different models across various probability thresholds. The function `calculate_cost_sensitive_net_benefit` computes the net benefit for given weights and thresholds. The `decision_curve_analysis` function generates a plot comparing the net benefits of the selected model, an alternative model, random predictions, and extreme cases (all positive or all negative). It should be noted that the net benefit by itself does not provide an overall assessment as it only relies on true positives and false positives.

## Model calibration and conformal predictions
Here we applied isotonic regression as the model calibration method. Isotonic regression is a non-parametric approach used to calibrate the predicted probabilities of a classifier. Note that the calibration should be preferrebly done based on an unseen dataset (not the dataset the model is already trained).

The following steps are followed:

1) Test Set Split:
We split the test set into a calibration set (X_calibration, y_calibration) and a new test set (X_new_test, y_new_test). The calibration set is used to compute the nonconformity scores for Conformal Prediction.

2) Isotonic Regression:
We calibrate the predicted probabilities using Isotonic Regression to make the predicted probabilities more reliable.

3) Conformal Prediction:
To understand conformal prediction you can refer to Shafer and Vovk, 2008. Below is the steps performed in the following code:

conformal prediction for binary classification is based on a split-conformal approach. The goal is to provide prediction sets for each test instance, ensuring 95% coverage (i.e., that the true label is included in the prediction set for approximately 95% of instances).

Non-conformity Scores: These scores are calculated for the calibration set based on the predicted probabilities for the true class: ( s_i = 1 - p_i ), where ( p_i ) is the predicted probability for the true class.

Threshold Calculation: The 95th percentile of the non-conformity scores from the calibration set is used to determine the threshold for prediction sets.

Prediction Sets: For each test instance, the non-conformity scores for both classes (class 0 and class 1) are compared to the threshold. The class(es) whose non-conformity scores fall below the threshold are included in the prediction set.

Coverage and Metrics: The coverage, or proportion of test instances where the true label is in the prediction set, is reported. Additional metrics like Brier Score, MCC, and AUC are also evaluated for confident predictions.

Coverage is the proportion of test instances for which the true label is included in the prediction set. In this analysis, coverage was calculated as the fraction of confident predictions made by the model:

The percentage of confident predictions was calculated as the fraction of predictions where the model was able to predict a single class with confidence.

4) Filtering Confident Predictions:
We filter out the predictions where the p-value is less than alpha (indicating less confidence). Only single-class prediction sets are retained, which means the model is confident enough to assign a label with a clear margin.

5) Evaluation:
Various metrics like Brier Score, Matthews Correlation Coefficient (MCC), AUC, and PR AUC are computed for the confident predictions only. We also report the percentage of confident predictions, giving insight into how often the model is making confident predictions.

## Survival Models
This part of the pipeline is intended to be used in case the data contains a column for time-to-event information as a survival outcome variable. If so, it is possible to develop a random survival forest (RSF) model and a Cox’s proportional hazard’s model (CPH) with elastic net penalty and compare their prediction performance. For survival models we use scikit-survival package and you can read about here: https://scikit-survival.readthedocs.io/en/stable/#
By default, RSF is chosen to be interpreted for its powerful algorithm that can detect nonlinearities allowing it to potentially represent the data better and outperform its linear alternative (Cox model). It is of course possible to include more models from scikit-survival package, however it is expected that RSF to have similar performance to its alternative ensemble models. 

Note that the survival models can work with one-hot encoded data with no missingness. So X_train_OHE and X_test_OHE are suitable for the analyses. Another thing to note is that the time-to-event column is not in X_train_OHE and X_test_OHE and so we get that column from the copy of the dataset that was initially made in the beginning of the pipeline as a back up to extract that information.

This is how the outcome column has to be formatted for survival models. In each array the first entry determines if there is any event or not and the second entry determines the last follow up time within a specific observation period. For example, when there is an event (e.g. daignosed disease) the first entry becomes True and the second entry show when it was recorded with respect to the baseline time (e.g. time of transplantation). If there was no event, then the last recorded sample of a patient is considered for the time and the event entry is False that clarifies that there was no event up to that time. Read more about the censoring here: https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html. 

The performance of the survival models and binary classification models cannot be directly compared. One solution proposed in this pipeline is to convert the predicted cumulative hazard from each patient to binary labels (e.g., death or survival). This way the performance of the survival model can be compared with the binary classificaiton models. Note that the definition used for censoring should be the same for the two types of models for their comparison. It means that, if there are censored data (e.g., lost to follow up cases), the binary labels can either be assumed to be assigned to a class according to expert knowledge or to be assigned using other methods like semi-supervised learning that is also available in the pipeline using [label propagation method](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html#sklearn.semi_supervised.LabelPropagation) (see `semi_supervised` condition in the pipeline).

hyperparameter tuning of random survival forest (RSF) model using random search method in repeated 5-fold cross validation (parameters can be modified). The best parameters are used when training the RSF model on the train set. It is important that the data is structured like the examples provided by tutorials to follow for example the naming protocols etc. Customization can be made to the source code when the plots require modifications (default time is in days but it could be in years for example).

- **Survival Analysis:** Train on the training set and test on the test set.
- **Model Comparison:** Evaluate and compare model performance.

In summary, `RandomSurvivalForest` and `CoxnetSurvivalAnalysis` are used here for survival analysis, along with hyperparameter tuning, model evaluation, and interpretation. `CoxPHSurvivalAnalysis` is also available but it has technical limitations and has less regularization parameters than `CoxnetSurvivalAnalysis` and so `CoxnetSurvivalAnalysis` is more preferred here.

#### Random Survival Forest (RSF)
- **Hyperparameter Tuning:** Utilizes `RandomizedSearchCV` with parameters for the number of estimators, minimum samples split, and minimum samples leaf.
- **Model Training:** Trains the best model found through the randomized search on the training data.
- **Model Saving:** Saves the trained model using `joblib`.

#### Cox Proportional Hazards Model (CPH) 
- **Hyperparameter Tuning:** Uses `RandomizedSearchCV` with parameters including regularization, method for handling tied event times, number of iterations, tolerance, and verbosity.
- **Model Training:** Trains the best model from the randomized search on the training data.
- **Model Saving:** Saves the trained model using `joblib`.

#### Training and evaluation of the survival models

First we do corss validation using the traing set (development set) to assess the prediction performance of RSF and CPH models. The cross validation follows the same folding setting (i.e., number of folds) of the binary classification models (except for the survival models it is not stratified by the biary outcome variable). After we do the assessment of the models based on cross validation, we train the models on the whole trainig set and evaluate them on the test set. Two metrics are used to evaluate the models: (1) concordance index (CI), and (2) Integrated Brier Score (IBS). These scores are explained here: https://scikit-survival.readthedocs.io/en/v0.23.0/api/metrics.html.

#### Concordance Index (CI) and Integrated Brier Score (IBS)

##### Concordance Index (CI)
The **Concordance Index (CI)** is a performance measure for survival models. It evaluates how well the model can correctly rank survival times. The CI measures the proportion of all usable pairs of individuals where the model correctly predicts the order of survival times. A CI of `1.0` indicates perfect predictions, while `0.5` represents random guessing.

- **Interpretation**: 
  - **CI = 1**: Perfect prediction, the model correctly ranks all pairs of individuals.
  - **CI = 0.5**: Random prediction, no better than chance.
  - **CI < 0.5**: Worse than random, model is predicting the reverse order of survival times.

For more details: [Concordance Index in scikit-survival](https://scikit-survival.readthedocs.io/en/v0.23.0/api/generated/sksurv.metrics.concordance_index_censored.html#sksurv.metrics.concordance_index_censored).

##### Integrated Brier Score (IBS)
The **Integrated Brier Score (IBS)** is a measure of the accuracy of predicted survival probabilities over time. It is the average Brier score, which measures the difference between the predicted survival probability and the actual outcome (whether the event occurred or not), across a range of time points. A lower IBS indicates better performance.

- **Interpretation**:
  - **IBS = 0**: Perfect prediction, the model’s predicted probabilities match the true outcomes.
  - **Higher IBS values**: Less accurate predictions.

For more details: [Integrated Brier Score in scikit-survival](https://scikit-survival.readthedocs.io/en/v0.23.0/api/generated/sksurv.metrics.integrated_brier_score.html#sksurv.metrics.integrated_brier_score).

The above measures are already sufficient to assess the quality of the models. As a supplementary option, time-updated AUC is also depicted for RSF model on the test set.

#### Model Interpretation Using SHAP Values

SHAP method has recently been developed for survival models.

An elaborative method is to calculate SHAP values for variables over time. This has been implemented using SurvSHAP(t) package developed by [Krzyziński et al](https://www.sciencedirect.com/science/article/pii/S0950705122013302?via%3Dihub).
In our customized implementation we follow below steps:
1) we have `rsf_exp = SurvivalModelExplainer(model = rsf, data = X_train_surv, y = y_train_surv_transformed)` to create an explainer object.
2) we set a seed for reproducibility and the outcome type that is  cumulative hazard function for calculation of SHAP values using `survshap = PredictSurvSHAP(random_state = SEED, function_type = "chf")`
3) we use `compute_shap` function and parallel processing to compute the SHAP values efficiently.
4) a plot is generated for one sample (instance/patient). It displays the impact of each (baseline) variable over time. The impact of a variable may vary substantially over time and that is an important information revealed by this method.
5) Using `plot_shap_heatmap` function we aggregate the survival SHAP values and get the overall importance (impact) of the variables on the survival model. It ranks variables from top to bottom by their mean absolute SHAP values.
6) For more detailed visualization, we also use `plot_shap_time_series_all_features` that plots SHAP values for each variable and sample over time. In addition, it displays the median absolute SHAP values. The ranking of variables here are based on the medican absolute SHAP values.
7) At the end, we also get the aggregated SHAP plot (similar to binary classification models) for the survival model (RSF). 
   
#### Feature Importance Using Permutation Importance

- **Permutation Importance:** Computes feature importance for both the RSF and CPH models using permutation importance and sorts features by their mean importance scores.

#### Predicting cumulative hazard function

- **Prediction:** The cumulative hazard function is predicted for all training samples using the Random Survival Forest (RSF).
- **Separation into Classes:** Predictions are separated into two classes based on the binary target variable.
- **Survival Probabilities:** Cumulative hazards are converted to survival probabilities.
- **Statistics Calculation:** Median and interquartile range (IQR) are calculated for both classes.
In addition, a table is displayed that summarizes the counts for patients at risk, censored, and events for different time intervals. The table contains both counts for each time interval and accumulative counts for each measure from the baseline.

#### Visualization of predicted survival probabilities

Cumulative hazard is the output of the RSF model that can be converted to survival probabilities.
- **Plotting:** Median survival probabilities and IQR for both classes are plotted against time.
- **Annotations:** The plots include annotations for the Mann-Whitney U test results, comparing the risk scores between classes.

#### Evaluation on test set

- **Median Hazard Calculation:** Similar calculations (median and IQR) are performed for the test set.
- **Prediction Comparison:** Euclidean distances from the median curves are calculated to determine predicted classes based on proximity.

#### Translation of the Predicted Hazard Curves to Binary Predictions

- **Confusion Matrix:** Confusion matrix is computed, along with metrics such as sensitivity, specificity, PPV, NPV, balanced accuracy, and Matthews correlation coefficient (MCC).
- **Heatmap:** A heatmap of the confusion matrix is plotted.

This is how it's done:

1. **Calculate Distances:**
   - Calculate the Euclidean distances from each predicted hazard curve to the median hazard curves of both classes.

2. **Predict Classes:**
   - Determine the predicted class for each sample based on proximity to the median curves.

3. **Compute Metrics:**
   - Construct a confusion matrix and compute metrics such as sensitivity, specificity, PPV, NPV, balanced accuracy, and MCC.

#### Risk Scores Analysis
- **Average Risk Scores:** Average risk scores for each class are computed for both the training and test sets.
- **Comparison:** Differences and proportions of average risk scores between the two classes are calculated and compared.

#### Time-Dependent AUC
- **Cumulative Dynamic AUC:** Time-dependent AUC is calculated using cumulative dynamic AUC, with results plotted against time.
- **Integrated Brier Score:** The integrated Brier score is calculated to assess the accuracy of survival predictions over time.

#### Visualizations
- **Survival Probability Plots:** Plots showing median survival probabilities and IQR for both classes.
- **Cumulative Hazard Plots:** Plots showing median cumulative hazards and IQR for both classes, with additional samples plotted.
- **Confusion Matrix Heatmap:** Heatmap visualization of the confusion matrix.
- **Time-Dependent AUC Plot:** Plot showing time-dependent AUC over days from baseline.
- **Integrated Brier Score:** Calculation and visualization of integrated Brier scores.

## Regression Models
Train and evaluate regression models like Linear Regression and Random Forest Regression.

- **Regression Analysis:** Train on the training set and test on the test set.
- **Model Interpretation:** Interpret using SHAP method.

### Model evaluation and interpretation

1) Mean Squared Error (MSE)
- **Formula:** MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
- **Where:**
  - \( n \) is the number of observations
  - \( yᵢ \) is the actual value
  - \( ŷᵢ \) is the predicted value
- **Interpretation:**
  - Measures the average squared difference between actual and predicted values.
  - Lower MSE indicates better fit.
  - Sensitive to outliers.

2) Mean Absolute Error (MAE)
- **Formula:** MAE = (1/n) * Σ|yᵢ - ŷᵢ|
- **Where:**
  - \( n \) is the number of observations
  - \( yᵢ \) is the actual value
  - \( ŷᵢ \) is the predicted value
- **Interpretation:**
  - Measures the average absolute difference between actual and predicted values.
  - Lower MAE indicates better fit.
  - Less sensitive to outliers than MSE.
  - Same units as the original data.

3) R-squared (R²)
- **Formula:** R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
- **Where:**
  - \( yᵢ \) is the actual value
  - \( ŷᵢ \) is the predicted value
  - \( ȳ \) is the mean of actual values
- **Interpretation:**
  - Measures the proportion of variance in the dependent variable explained by the model.
  - Values range from -∞ to 1.
  - Higher values indicate better fit.
  - Negative values indicate the model performs worse than a horizontal line (mean of the target variable).

## Report the Python Environment
Report the Python environment and dependencies used in the pipeline.

- **Environment Report:** List the Python version, platform, and installed packages.

## Save the Executed Pipeline
Save the entire executed pipeline in HTML format for reproducibility.

In case there was any issue when saving output files like SHAP figures on disk, check your permission. For example, see below:
https://stackoverflow.com/questions/66496890/vs-code-nopermissions-filesystemerror-error-eacces-permission-denied
that explains how to fix probable permission issues, especially when using VS code:
`sudo chown -R username path`
like `sudo chown -R emanuel /home/emanuel/test/`

- **Export Pipeline:** Save the notebook as an HTML file for documentation and sharing.
