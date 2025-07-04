![Alt Text](https://github.com/PERSIMUNE/MAIT/blob/main/MAITlogo.gif)
[![GitHub release](https://img.shields.io/github/v/release/PERSIMUNE/MAIT)](https://github.com/PERSIMUNE/MAIT/releases)  
[![Docker Pulls](https://img.shields.io/docker/pulls/danishdyna/mait_30092024gpu)](https://hub.docker.com/r/danishdyna/mait_30092024gpu)  
![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue) 
[![License](https://img.shields.io/github/license/PERSIMUNE/MAIT)](https://github.com/PERSIMUNE/MAIT/blob/main/LICENSE)  
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPERSIMUNE%2FMAIT&count_bg=%233DC853&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=MAIT+codes+visitors&edge_flat=false)](https://hits.seeyoufarm.com)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fpersimune.github.io%2FMAIT%2F&count_bg=%233D7CC8&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=MAIT+book+portal+visitors&edge_flat=false)](https://hits.seeyoufarm.com)
[![arXiv](https://img.shields.io/badge/arXiv-2501.04547-B31B1B)](https://arxiv.org/abs/2501.04547)  
# MAIT - medical artificial intelligence toolbox

## Introduction
Welcome to the MAIT repository! This pipeline, implemented in Python, is designed to streamline your machine learning workflows using Jupyter Notebooks. It is compatible with both Windows and Linux operating systems. This repository also includes several tutorial notebooks to help you get started quickly. You can also refer to the `MANUAL` of MAIT for documentation. To overview MAIT's unique features and capabilities, we highly recommend reading the [MAIT preprint](https://arxiv.org/abs/2501.04547) on arXiv. If you use MAIT in your research, please remember to cite it in your publications.


## MAIT Workflow Overview

The MAIT framework provides a comprehensive machine learning pipeline for medical data analysis. The workflow consists of several interconnected stages, from data preprocessing to model interpretation:

```mermaid
flowchart TD
    %% Data Input Layer
    subgraph DataInput ["� Data Input Layer"]
        RawData[("Raw Medical Data<br/>CSV format<br/>Tabular structure")] 
        ExtData[("External Validation Data<br/>Independent test set<br/>Same feature structure")]
        ConfigFile[("Configuration Settings<br/>JSON parameters<br/>Streamlit interface")]
    end
    
    %% Data Preprocessing Layer
    subgraph Preprocessing ["🧹 Data Preprocessing Layer"]
        DataClean["Data Cleaning & Validation<br/>• Drop NaN outcome rows<br/>• Handle empty entries<br/>• Data type conversion"]
        MissingData["Handle Missing Values<br/>• KNN imputation<br/>• Filter high missingness<br/>• Column/row filtering"]
        OutlierDetect["Outlier Detection<br/>• Isolation Forest algorithm<br/>• Anomaly removal<br/>• Optional preprocessing"]
        RareCategories["Rare Categories Merging<br/>• Threshold: 5% default<br/>• 'Missing' category<br/>• String conversion"]
    end
    
    %% Data Split Layer
    subgraph DataSplit ["� Data Split Layer"]
        TrainSet[("Training Set (80%)<br/>Model development<br/>Cross-validation")] 
        TestSet[("Test Set (20%)<br/>Final evaluation<br/>Hold-out validation")] 
        ExtVal[("External Validation<br/>Independent dataset<br/>Real-world testing")]
    end
    
    %% Feature Engineering Layer
    subgraph FeatureEng ["⚙️ Feature Engineering Layer"]
        FeatureSelect["Feature Selection (mRMR)<br/>• Min-redundancy Max-relevance<br/>• Configurable num features<br/>• Mutual information"]
        DataScale["Data Scaling<br/>• Robust scaling method<br/>• Optional preprocessing<br/>• Numerical features"]
        Encoding["Categorical Encoding<br/>• One-hot encoding<br/>• String conversion<br/>• Handle categories"]
    end
    
    %% Model Training Layer
    subgraph ModelTraining ["Model Training Layer"]
        CVFolds["Cross-Validation (5-fold)<br/>• Stratified K-fold<br/>• Random state control<br/>• Performance estimation"]
        HPTuning["Hyperparameter Tuning<br/>• RandomizedSearchCV<br/>• 10 iterations default<br/>• ROC-AUC optimization"]
        ModelFit["Model Training<br/>• Class weight balancing<br/>• Sample weights<br/>• Fit-predict cycle"]
        ThresholdOpt["Threshold Optimization<br/>• Youden's J statistic<br/>• ROC curve analysis<br/>• Optimal cutoff"]
    end
    
    %% Available Models
    subgraph Models ["Available Models"]
        RF["Random Forest<br/>• n_estimators: 100-1000<br/>• max_depth: 3-10<br/>• Bootstrap sampling"]
        LGB["LightGBM<br/>• Gradient boosting<br/>• GPU support<br/>• GOSS algorithm"]
        CB["CatBoost<br/>• Categorical features<br/>• Ordered encoding<br/>• Symmetric trees"]
        QLattice["QLattice (Symbolic)<br/>• Feyn library<br/>• Mathematical expressions<br/>• Symbolic regression"]
        LR["Logistic Regression<br/>• Linear classification<br/>• L1/L2 regularization<br/>• Probability estimates"]
        NB["Naive Bayes<br/>• Gaussian distribution<br/>• Probabilistic classifier<br/>• No hyperparameters"]
        HGBC["Hist Gradient Boosting<br/>• Histogram-based<br/>• Fast performance<br/>• Large datasets"]
    end
    
    %% Model Evaluation Layer
    subgraph Evaluation ["� Model Evaluation Layer"]
        Performance["Performance Metrics<br/>• ROC-AUC & PR-AUC<br/>• MCC, F1-score<br/>• Sensitivity, Specificity"]
        BestModel["Best Model Selection<br/>• Mean CV performance<br/>• ROC-AUC ranking<br/>• Statistical comparison"]
        TestEval["Test Set Evaluation<br/>• Hold-out validation<br/>• Final performance<br/>• Generalization test"]
    end
    
    %% Model Interpretation Layer
    subgraph Interpretation ["� Model Interpretation Layer"]
        SHAP["SHAP Analysis<br/>• TreeExplainer<br/>• Feature contributions<br/>• Local explanations"]
        FeatureImp["Feature Importance<br/>• Permutation importance<br/>• Tree-based importance<br/>• Ranking analysis"]
        ModelExplain["Model Explainability<br/>• Global interpretations<br/>• Feature interactions<br/>• Decision paths"]
    end
    
    %% Results and Visualization Layer
    subgraph Results ["� Results & Visualization Layer"]
        ROCCurve["ROC Curves<br/>• Multi-model comparison<br/>• AUC visualization<br/>• Threshold analysis"]
        SHAPPlots["SHAP Plots<br/>• Summary plots<br/>• Decision plots<br/>• Feature contributions"]
        ConfMatrix["Confusion Matrix<br/>• Classification results<br/>• Error analysis<br/>• Threshold evaluation"]
        Reports["Performance Reports<br/>• Comprehensive metrics<br/>• Model comparisons<br/>• Export formats"]
    end
    
    %% Optional Advanced Analysis
    subgraph Advanced ["� Advanced Analysis (Optional)"]
        Survival["Survival Analysis<br/>• Random Survival Forest<br/>• Cox regression<br/>• Time-to-event modeling"]
        Regression["Regression Analysis<br/>• Random Forest Regressor<br/>• Linear regression<br/>• Continuous outcomes"]
        Calibration["Model Calibration<br/>• Probability calibration<br/>• Conformal predictions<br/>• Uncertainty quantification"]
    end
    
    %% Connections between layers
    DataInput --> Preprocessing
    Preprocessing --> DataSplit
    DataSplit --> FeatureEng
    FeatureEng --> ModelTraining
    ModelTraining --> Models
    Models --> Evaluation
    Evaluation --> Interpretation
    Interpretation --> Results
    Results --> Advanced
    
    %% Styling for better readability
    classDef dataInput fill:#bbdefb,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    classDef preprocessing fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef models fill:#a5d6a7,stroke:#388e3c,stroke-width:2px,color:#1b5e20
    classDef interpretation fill:#ffe0b2,stroke:#ef6c00,stroke-width:2px,color:#bf360c
    classDef advanced fill:#f8bbd9,stroke:#c2185b,stroke-width:2px,color:#880e4f
    
    %% Apply classes to subgraphs and nodes
    class DataInput,RawData,ExtData,ConfigFile dataInput
    class Preprocessing,DataSplit,FeatureEng,ModelTraining,DataClean,MissingData,OutlierDetect,RareCategories,TrainSet,TestSet,ExtVal,FeatureSelect,DataScale,Encoding,CVFolds,HPTuning,ModelFit,ThresholdOpt preprocessing
    class Models,Evaluation,RF,LGB,CB,QLattice,LR,NB,HGBC,Performance,BestModel,TestEval models
    class Interpretation,Results,SHAP,FeatureImp,ModelExplain,ROCCurve,SHAPPlots,ConfMatrix,Reports interpretation
    class Advanced,Survival,Regression,Calibration advanced
```

### Key Features:
- **Multi-Modal Analysis**: Binary classification, survival analysis, and regression
- **Automated Pipeline**: Streamlined workflow from raw data to interpretable results
- **Explainable AI**: SHAP-based model interpretation and feature importance analysis
- **Multiple Algorithms**: Seven different machine learning models including tree-based, linear, and symbolic regression
- **Robust Validation**: Cross-validation, external validation, and comprehensive performance metrics
- **Advanced Analysis**: Optional survival analysis, regression modeling, and model calibration

### Technical Implementation Details

#### Data Processing Pipeline
- **Missing Data Handling**: KNN imputation with adaptive neighbor selection based on dataset size
- **Outlier Detection**: Isolation Forest algorithm with contamination auto-estimation
- **Categorical Processing**: One-hot encoding with rare category merging (5% threshold default)
- **Feature Selection**: mRMR (minimum Redundancy Maximum Relevance) algorithm with mutual information
- **Data Scaling**: Robust scaling method resistant to outliers using median and IQR

#### Model Training & Evaluation
- **Cross-Validation**: 5-fold stratified cross-validation with class balance preservation
- **Hyperparameter Tuning**: RandomizedSearchCV with 10 iterations for efficient parameter optimization
- **Performance Metrics**: Comprehensive evaluation including ROC-AUC, PR-AUC, MCC, F1-score, sensitivity, specificity, and Brier score
- **Threshold Optimization**: Youden's J statistic for optimal classification threshold selection
- **Model Selection**: Best model selection based on mean cross-validation performance

#### Machine Learning Models
| Model | Key Features | Hyperparameters |
|-------|-------------|----------------|
| **Random Forest** | Bootstrap sampling, feature randomness | n_estimators (100-1000), max_depth (3-10), min_samples_split |
| **LightGBM** | Gradient boosting, GPU support, GOSS algorithm | num_leaves, min_child_samples, learning_rate, regularization |
| **CatBoost** | Categorical features, ordered encoding, symmetric trees | learning_rate, depth, l2_leaf_reg, iterations |
| **QLattice** | Symbolic regression, mathematical expressions | n_epochs, max_complexity |
| **Logistic Regression** | Linear classification, L1/L2 regularization | C, max_iter, tolerance |
| **Naive Bayes** | Probabilistic classifier, no hyperparameters | N/A |
| **Hist Gradient Boosting** | Histogram-based, fast on large datasets | max_iter, learning_rate, max_depth, regularization |

#### Interpretability & Explainability
- **SHAP Analysis**: TreeExplainer for tree-based models with local and global explanations
- **Feature Importance**: Multiple methods including permutation importance and tree-based importance
- **Visualization**: Comprehensive plots including ROC curves, SHAP summary plots, decision plots, and confusion matrices
- **Model Calibration**: Probability calibration and conformal predictions for uncertainty quantification

#### Configuration Options
The pipeline supports extensive configuration through JSON files or the Streamlit interface:
- **Data Configuration**: File paths, outcome variables, categorical features, data splitting options
- **Model Selection**: Choose from 7 algorithms, configure hyperparameter tuning
- **Feature Engineering**: Feature selection, scaling, outlier removal, rare category handling
- **Training Parameters**: Cross-validation folds, CPU/GPU allocation, performance metrics
- **Output Settings**: Result folders, visualization formats, class labels

### MAIT Workflow - Process Flow

For a more detailed view of the decision points and process flow, here's the complete MAIT workflow as a flowchart:

```mermaid
flowchart TD
    %% Start
    Start([Start MAIT Pipeline]) --> ConfigInput["Load Configuration Settings<br/>JSON config file<br/>Streamlit interface<br/>Parameter validation"]
    
    %% Data Input
    ConfigInput --> DataInput{Data Input Configuration}
    DataInput -->|Raw Data| RawData[("Load Raw Medical Data<br/>CSV format<br/>Feature validation<br/>Data profiling")]
    DataInput -->|With External| ExtData[("Load External Validation Data<br/>Independent test set<br/>Same feature structure<br/>Real-world validation")]
    RawData --> DataPrep["Data Preparation<br/>Initial cleaning<br/>Column verification<br/>Data type inference"]
    ExtData --> DataPrep
    
    %% Data Preprocessing
    DataPrep --> CleanData["Clean Data & Handle Missing Values<br/>• Drop NaN outcome rows<br/>• Replace empty strings with NaN<br/>• Data type conversion<br/>• Categorical string conversion"]
    CleanData --> CheckOutliers{Remove Outliers?<br/>User configurable<br/>remove_outliers parameter}
    CheckOutliers -->|Yes| OutlierRemoval["Isolation Forest Outlier Detection<br/>Isolation Forest algorithm<br/>KNN imputation preprocessing<br/>One-hot encoding<br/>Contamination estimation"]
    CheckOutliers -->|No| CatProcess["Process Categorical Variables<br/>Category data type<br/>String conversion<br/>Handle mixed types"]
    OutlierRemoval --> CatProcess
    CatProcess --> RareCategories["Merge Rare Categories<br/>5% threshold default<br/>'Missing' category creation<br/>Frequency analysis"]
    RareCategories --> ValidationCheck["Data Validation & Type Conversion<br/>Feature consistency<br/>Missing value patterns<br/>Data quality metrics"]
    
    %% Data Splitting Decision
    ValidationCheck --> SplitDecision{Data Split Strategy<br/>Discovery vs Prediction<br/>data_split parameter}
    SplitDecision -->|Discovery Mode| DiscoveryMode["Use All Data for Cross-Validation<br/>5-fold stratified CV<br/>Full dataset utilization<br/>Exploratory analysis"]
    SplitDecision -->|Prediction Mode| PredictionMode["Split Data 80/20<br/>Stratified split<br/>Train/test separation<br/>Hold-out validation"]
    
    %% Prediction Mode Path
    PredictionMode --> TrainTest["Training Set 80% / Test Set 20%<br/>Stratified sampling<br/>Class balance preservation<br/>Random state control"]
    TrainTest --> FeatureEng["Feature Engineering Pipeline<br/>Preprocessing pipeline<br/>Feature transformations<br/>Data preparation"]
    DiscoveryMode --> FeatureEng
    
    %% Feature Engineering
    FeatureEng --> FeatureSelect{Feature Selection?<br/>feat_sel parameter<br/>30 features default}
    FeatureSelect -->|Yes| mRMR["Apply mRMR Feature Selection<br/>Min-redundancy Max-relevance<br/>Mutual information<br/>Feature ranking<br/>Configurable feature count"]
    FeatureSelect -->|No| DataScale
    mRMR --> DataScale{Scale Data?<br/>scale_data parameter<br/>Numerical features}
    DataScale -->|Yes| Scaling["Apply Robust Scaling<br/>Robust scaler method<br/>Outlier resistant<br/>Median & IQR based"]
    DataScale -->|No| Encoding["Categorical Encoding<br/>One-hot encoding<br/>Handle categories<br/>Binary features"]
    Scaling --> Encoding
    Encoding --> ModelSelection["Select Models to Train<br/>7 algorithms available<br/>User configurable<br/>Performance comparison"]
    
    %% Model Selection
    ModelSelection --> Models{Choose Models<br/>Multi-algorithm support<br/>Configurable selection}
    Models --> RF["Random Forest<br/>n_estimators: 100-1000<br/>max_depth: 3-10<br/>Bootstrap sampling<br/>Class weight balancing"]
    Models --> LGB["LightGBM<br/>Gradient boosting<br/>GPU support available<br/>GOSS algorithm<br/>Histogram-based"]
    Models --> CB["CatBoost<br/>Categorical features<br/>Ordered encoding<br/>Symmetric trees<br/>Fast training"]
    Models --> QLattice["QLattice (Symbolic)<br/>Feyn library<br/>Mathematical expressions<br/>Symbolic regression<br/>Interpretable models"]
    Models --> LR["Logistic Regression<br/>Linear classification<br/>L1/L2 regularization<br/>Probability estimates<br/>Baseline model"]
    Models --> NB["Naive Bayes<br/>Gaussian distribution<br/>Probabilistic classifier<br/>No hyperparameters<br/>Fast training"]
    Models --> HGBC["Hist Gradient Boosting<br/>Histogram-based<br/>Fast performance<br/>Large datasets<br/>Early stopping"]
    
    %% Model Training Process
    RF --> CVProcess["Cross-Validation Process<br/>5-fold stratified<br/>Performance estimation<br/>Model comparison<br/>Sample weights"]
    LGB --> CVProcess
    CB --> CVProcess
    QLattice --> CVProcess
    LR --> CVProcess
    NB --> CVProcess
    HGBC --> CVProcess
    
    %% Cross-Validation
    CVProcess --> HPTuning{Hyperparameter Tuning?<br/>hp_tuning parameter<br/>10 iterations default}
    HPTuning -->|Yes| RandomSearch["RandomizedSearchCV<br/>Random parameter sampling<br/>ROC-AUC optimization<br/>Grid search alternative<br/>Efficient search"]
    HPTuning -->|No| ModelTrain["Train Models<br/>Fit algorithms<br/>Class weight balancing<br/>Sample weights<br/>CV training"]
    RandomSearch --> ModelTrain
    ModelTrain --> ThresholdOpt["Optimize Classification Threshold<br/>Youden's J statistic<br/>ROC curve analysis<br/>Optimal cutoff<br/>Sensitivity-specificity balance"]
    ThresholdOpt --> PerformanceEval["Evaluate Performance Metrics<br/>Multiple metrics<br/>Comprehensive evaluation<br/>Statistical measures"]
    
    %% Model Evaluation
    PerformanceEval --> CalcMetrics["Calculate Metrics<br/>ROC-AUC & PR-AUC<br/>MCC & F1-score<br/>Sensitivity & Specificity<br/>Balanced accuracy<br/>Brier score"]
    CalcMetrics --> BestModel["Select Best Model<br/>Mean CV performance<br/>ROC-AUC ranking<br/>Statistical comparison<br/>Multi-metric evaluation"]
    BestModel --> TestEval{Test Set Available?<br/>Prediction mode<br/>Hold-out validation}
    TestEval -->|Yes| TestPerformance["Evaluate on Test Set<br/>Hold-out validation<br/>Final performance<br/>Generalization test<br/>Unbiased evaluation"]
    TestEval -->|No| ModelInterpret["Model Interpretation<br/>Explainability analysis<br/>Feature importance<br/>Decision insights"]
    TestPerformance --> ModelInterpret
    
    %% Model Interpretation
    ModelInterpret --> SHAPAnalysis["SHAP Analysis<br/>TreeExplainer<br/>Feature contributions<br/>Local explanations<br/>Additive explanations"]
    SHAPAnalysis --> FeatureImp["Feature Importance Analysis<br/>Permutation importance<br/>Tree-based importance<br/>Feature ranking<br/>Global importance"]
    FeatureImp --> ExplainPlots["Generate Explanation Plots<br/>SHAP visualizations<br/>Feature plots<br/>Decision paths<br/>Model insights"]
    
    %% Results Generation
    ExplainPlots --> ResultsGen["Generate Results<br/>Comprehensive reports<br/>Performance summaries<br/>Model comparisons<br/>Export formats"]
    ResultsGen --> ROCCurves["ROC Curves<br/>Multi-model comparison<br/>AUC visualization<br/>Threshold analysis<br/>Performance curves"]
    ResultsGen --> SHAPPlots["SHAP Plots<br/>Summary plots<br/>Decision plots<br/>Feature contributions<br/>Waterfall plots"]
    ResultsGen --> ConfMatrix["Confusion Matrix<br/>Classification results<br/>Error analysis<br/>Threshold evaluation<br/>TP/FP/TN/FN"]
    ResultsGen --> Reports["Performance Reports<br/>Comprehensive metrics<br/>Model comparisons<br/>Export formats<br/>Summary statistics"]
    
    %% Advanced Analysis Decision
    ROCCurves --> AdvancedAnalysis{Advanced Analysis?<br/>Optional modules<br/>User configurable}
    SHAPPlots --> AdvancedAnalysis
    ConfMatrix --> AdvancedAnalysis
    Reports --> AdvancedAnalysis
    
    %% Advanced Analysis Options
    AdvancedAnalysis -->|Survival Analysis| SurvivalAnalysis["Survival Analysis<br/>Random Survival Forest<br/>Cox regression<br/>Time-to-event modeling<br/>Kaplan-Meier curves"]
    AdvancedAnalysis -->|Regression Analysis| RegressionAnalysis["Regression Analysis<br/>Random Forest Regressor<br/>Linear regression<br/>Continuous outcomes<br/>R² metrics"]
    AdvancedAnalysis -->|Model Calibration| Calibration["Model Calibration<br/>Probability calibration<br/>Conformal predictions<br/>Uncertainty quantification<br/>Reliability curves"]
    AdvancedAnalysis -->|No| EndPipeline
    
    %% Advanced Analysis Outputs
    SurvivalAnalysis --> AdvancedResults["Advanced Analysis Results<br/>Specialized metrics<br/>Advanced visualizations<br/>Domain-specific insights<br/>Extended reports"]
    RegressionAnalysis --> AdvancedResults
    Calibration --> AdvancedResults
    AdvancedResults --> EndPipeline["End Pipeline<br/>Analysis complete<br/>Results exported<br/>Ready for deployment<br/>All outputs saved"]
    
    %% External Validation Path
    TestPerformance --> ExtValidation{External Validation?<br/>Independent dataset<br/>Real-world testing}
    ExtValidation -->|Yes| ExtVal["Apply Model to External Data<br/>Independent validation<br/>Model generalization<br/>Real-world performance<br/>Deployment readiness"]
    ExtValidation -->|No| ModelInterpret
    ExtVal --> ExtResults["External Validation Results<br/>Performance metrics<br/>Generalization assessment<br/>Deployment confidence<br/>Model robustness"]
    ExtResults --> ModelInterpret
    
    %% Styling
    classDef startEnd fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px,color:#1b5e20
    classDef process fill:#bbdefb,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    classDef decision fill:#ffe0b2,stroke:#ef6c00,stroke-width:2px,color:#bf360c
    classDef data fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef model fill:#a5d6a7,stroke:#388e3c,stroke-width:2px,color:#1b5e20
    classDef results fill:#f8bbd9,stroke:#c2185b,stroke-width:2px,color:#880e4f
    classDef advanced fill:#dcedc8,stroke:#689f38,stroke-width:2px,color:#33691e
    
    class Start,EndPipeline startEnd
    class ConfigInput,DataPrep,CleanData,CatProcess,RareCategories,ValidationCheck,FeatureEng,Encoding,CVProcess,ModelTrain,ThresholdOpt,PerformanceEval,CalcMetrics,BestModel,TestPerformance,ModelInterpret,SHAPAnalysis,FeatureImp,ExplainPlots,ResultsGen process
    class DataInput,CheckOutliers,SplitDecision,FeatureSelect,DataScale,Models,HPTuning,TestEval,AdvancedAnalysis,ExtValidation decision
    class RawData,ExtData,TrainTest,DiscoveryMode,PredictionMode data
    class RF,LGB,CB,QLattice,LR,NB,HGBC,OutlierRemoval,mRMR,Scaling,RandomSearch model
    class ROCCurves,SHAPPlots,ConfMatrix,Reports,ExtVal,ExtResults results
    class SurvivalAnalysis,RegressionAnalysis,Calibration,AdvancedResults advanced
```

This flowchart illustrates the complete decision-making process and shows how MAIT adapts to different analysis needs:

#### Key Decision Points:
- **Discovery vs. Prediction Mode**: Choose between exploration (cross-validation only) or prediction (train/test split)
- **Data Preprocessing Options**: Outlier removal, feature selection, data scaling
- **Model Selection**: Choose from 7 different machine learning algorithms
- **Advanced Analysis**: Optional survival analysis, regression modeling, or calibration
- **External Validation**: Apply trained models to independent datasets

## How to Use

### Step 1: Clone the Repository
First, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/PERSIMUNE/MAIT.git
cd MAIT
```

### Step 2: Install Conda and Setup the Environment

There are different ways to create an environment to use MAIT.
Ensure that you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed. 

Create a new conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate mait_py3_10_9
```
Or if you prefer to use `pip` to install the required packages, you can create a conda environment and install the dependencies using `pip`:

```bash
conda create --name mait_py3_10_9 python=3.10.9
conda activate mait_py3_10_9
pip install -r requirements.txt
```
Also you can try an updated Python version, and also use mlflow (experimental).

```bash
conda create -n mait_env python=3.12 \
  numpy pandas scikit-learn scipy matplotlib seaborn joblib lightgbm catboost ipykernel \
  imbalanced-learn mlflow shap scikit-survival -c conda-forge
conda activate mait_env
pip install feyn mrmr_selection survshap openpyxl
conda install protobuf=3.20.* -c conda-forge
pip install --upgrade mlflow
```
### Step 3: Using Docker
A Docker image is available for this pipeline. You can build and run the Docker container using the `Dockerfile` provided in the repository. Here are the steps:

1. Build the Docker image (or use the one that is already available: https://hub.docker.com/r/danishdyna/mait_30092024gpu):

    ```bash
    docker build -t mait_py3_10_9:latest .
    ```

2. Run the Docker container where your MAIT pipeline files are located:

    ```bash
    docker run -it --rm -p 8888:8888 mait_py3_10_9:latest
    ```
    or do this
    ```bash
    docker run  --gpus all -p 8888:8888 -v "$(pwd):/app" -it danishdyna/mait_30092024gpu /bin/bash
    ```

Inside the container, you should activate the conda environment using `activate the conda mait_py3_10_9_30092024gpu` then run this to initiate Jupyter Notebook: `jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root`.

You can also refer to the `environment.yml` file to understand the dependencies and create your Docker environment. If you run the Docker container on an HPC server or a remote computer you can use http://your_HPC_server_address:8888/ to get access to the Jupyter Notebook to run MAIT.


## Quick Start with Streamlit Configuration Interface

For new users who want to get started quickly without diving into Jupyter notebooks, MAIT includes a user-friendly **Streamlit web interface** for configuration:

### Try Online (No Installation Required)
**Access the live app at: https://maitconfig.streamlit.app/**

### Launch Locally
```bash
cd MAIT
./run_streamlit.sh
```

Or manually:
```bash
streamlit run streamlit_app.py
```

### Features
- **Interactive Configuration**: Set up your machine learning pipeline through an intuitive web interface
- **Demo Data Included**: Test MAIT immediately with built-in medical dataset
- **Ready-to-Run Scripts**: Generate complete Python scripts with your configurations pre-filled
- **No Manual Editing**: Generated scripts are ready to execute without any code modifications

### Quick Demo
1. Launch the Streamlit app
2. Click "Use Demo Data" in the Data Configuration section
3. Configure your preferred models and parameters
4. Generate and download your customized MAIT pipeline script

This interface is perfect for users who want to explore MAIT's capabilities before diving into the detailed Jupyter tutorial notebooks.

## Tutorials
The repository includes several Jupyter Notebooks that serve as tutorials. These notebooks cover various aspects of the pipeline and demonstrate how to use different components effectively. Below you can find a list of available tutorials:

1. [Tutorial 1: Prediction of antimicrobial resistance for Azithromycin](https://github.com/PERSIMUNE/MAIT/blob/main/Tutorials/MAIT_Tutorial_Azithromycin_pub.html)
2. [Tutorial 2: Prediction of antimicrobial resistance for Ciprofloxacin](https://github.com/PERSIMUNE/MAIT/blob/main/Tutorials/MAIT_Tutorial_Ciprofloxacin_pub.html)
3. [Tutorial 3: Prediction of Dementia](https://github.com/PERSIMUNE/MAIT/blob/main/Tutorials/MAIT_Tutorial_Dementia_pub.html)
4. [Tutorial 4: Prediction of Breast Cancer](https://github.com/PERSIMUNE/MAIT/blob/main/Tutorials/MAIT_Tutorial_BreastCancer_pub.html)

## How to Cite
Here's how to cite MAIT:

### **APA Style**
Zargari Marandi, R., Frahm, A. S., Lundgren, J., Murray, D. D., & Milojevic, M. (2025) "Medical artificial intelligence toolbox (MAIT): An explainable machine learning framework for binary classification, survival modelling, and regression analyses." arXiv preprint arXiv:2501.04547.

### **BibTeX**
```bibtex
@article{zargari2025mait,
  title={Medical artificial intelligence toolbox (MAIT): An explainable machine learning framework for binary classification, survival modelling, and regression analyses},
  author={Zargari Marandi, Ramtin and Frahm, Anne Svane and Lundgren, Jens and Murray, Daniel Dawson and Milojevic, Maja},
  journal={arXiv preprint arXiv:2501.04547},
  year={2025},
  url={https://arxiv.org/abs/2501.04547}
}
```

## License
This pipeline is free to use for research purposes. Please ensure you follow the licenses of the individual packages used within this pipeline. For more details, refer to the `LICENSE` file in the repository.

---

We hope you find this pipeline useful for your machine learning projects. If you encounter any issues or have any questions, feel free to open an issue on GitHub.
![Alt Text](https://github.com/PERSIMUNE/MAIT/blob/main/MAIT_results_examples.gif)
