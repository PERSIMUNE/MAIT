import streamlit as st
import pandas as pd
import os
import json
import subprocess
import sys
import time
import re
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="MAIT Pipeline Configuration",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e7d32;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e88e5;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header"> MAIT Pipeline Configuration Interface</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to MAIT (Medical Artificial Intelligence Toolbox)!</strong><br>
        This interface helps you configure and run MAIT pipelines for binary classification, survival modeling, and regression analysis.
        Configure your parameters below and generate a ready-to-run notebook or execute the pipeline directly.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section:", [
        "Data Configuration",
        "Model Selection", 
        "Feature Engineering",
        "Training Parameters",
        "Output Settings",
        "Advanced Options",
        "Generate & Run"
    ])

    # Initialize session state for configuration
    if 'config' not in st.session_state:
        st.session_state.config = init_default_config()

    # Page routing
    if page == "Data Configuration":
        data_configuration_page()
    elif page == "Model Selection":
        model_selection_page()
    elif page == "Feature Engineering":
        feature_engineering_page()
    elif page == "Training Parameters":
        training_parameters_page()
    elif page == "Output Settings":
        output_settings_page()
    elif page == "Advanced Options":
        advanced_options_page()
    elif page == "Generate & Run":
        generate_run_page()

def init_default_config():
    """Initialize default configuration"""
    return {
        # Data Configuration
        'data_file': '',
        'outcome_var': '',
        'columns_to_drop': [],
        'cat_features': [],
        'external_val': False,
        'ext_val_demo': False,
        
        # Data Split
        'data_split': True,
        'train_size_perc': 0.8,
        'data_split_by_patients': False,
        'patient_id_col': '',
        'data_split_multi_strats': False,
        'already_split': False,
        
        # Model Selection
        'models_to_include': ["RandomForest_mdl", "LightGBM_mdl", "LogisticRegression_mdl"],
        
        # Feature Engineering
        'feat_sel': True,
        'num_features_sel': 30,
        'merged_rare_categories': True,
        'rarity_threshold': 0.05,
        'remove_outliers': False,
        'filter_highly_mis_feats': True,
        'exclude_highly_missing_columns': True,
        'exclude_highly_missing_rows': True,
        
        # Training Parameters
        'hp_tuning': True,
        'n_iter_hptuning': 10,
        'cv_folds': 5,
        'tun_score': 'roc_auc',
        'oversampling': False,
        'scale_data': False,
        
        # Resource Configuration
        'n_cpu_for_tuning': 4,
        'n_cpu_model_training': 4,
        'GPU_avail': False,
        
        # Output Settings
        'main_folder_name': 'mait_results',
        'fig_file_format': 'png',
        'class_labels_display': ['No', 'Yes'],
        
        # Analysis Types
        'survival_analysis': False,
        'regression_analysis': False,
        
        # Advanced Options
        'demo_configs': False,
        'shorten_feature_names': True,
        'test_only_best_cvmodel': True
    }

def data_configuration_page():
    st.markdown('<h2 class="section-header"> Data Configuration</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Demo Data Available:</strong><br>
        Use <code>demo_medical_data.csv</code> for testing. It contains 50 synthetic patients with cardiovascular risk factors 
        for binary classification (heart_disease). Suggested categorical features: gender, smoking_status.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        
        # File upload or path
        data_option = st.radio("Data Source:", ["Upload File", "File Path"])
        
        if data_option == "Upload File":
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            if uploaded_file:
                st.session_state.config['data_file'] = uploaded_file.name
                # Preview data
                df = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                # Auto-populate column options
                st.session_state.available_columns = df.columns.tolist()
        else:
            data_path = st.text_input("Enter file path:", value=st.session_state.config.get('data_file', ''))
            
            # Add demo file suggestion
            if st.button("Use Demo Data"):
                data_path = "demo_medical_data.csv"
                st.session_state.config['data_file'] = data_path
                st.rerun()
            
            st.session_state.config['data_file'] = data_path
            
            if data_path and os.path.exists(data_path):
                try:
                    df = pd.read_csv(data_path)
                    st.success(f" File found! Shape: {df.shape}")
                    st.dataframe(df.head())
                    st.session_state.available_columns = df.columns.tolist()
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    
    with col2:
        st.subheader("Variable Configuration")
        
        # Outcome variable
        if 'available_columns' in st.session_state:
            outcome_var = st.selectbox(
                "Outcome Variable (Target):", 
                [''] + st.session_state.available_columns,
                index=0 if not st.session_state.config['outcome_var'] else st.session_state.available_columns.index(st.session_state.config['outcome_var']) + 1
            )
            st.session_state.config['outcome_var'] = outcome_var
        else:
            outcome_var = st.text_input("Outcome Variable:", value=st.session_state.config['outcome_var'])
            st.session_state.config['outcome_var'] = outcome_var
        
        # Categorical features
        if 'available_columns' in st.session_state:
            cat_features = st.multiselect(
                "Categorical Features:", 
                st.session_state.available_columns,
                default=st.session_state.config['cat_features']
            )
            st.session_state.config['cat_features'] = cat_features
        
        # Columns to drop
        if 'available_columns' in st.session_state:
            columns_to_drop = st.multiselect(
                "Columns to Drop:", 
                st.session_state.available_columns,
                default=st.session_state.config['columns_to_drop']
            )
            st.session_state.config['columns_to_drop'] = columns_to_drop
    
    # Data Split Configuration
    st.subheader("Data Split Configuration")
    
    col3, col4 = st.columns(2)
    
    with col3:
        data_split = st.checkbox("Enable Data Split", value=st.session_state.config['data_split'])
        st.session_state.config['data_split'] = data_split
        
        if data_split:
            train_size = st.slider("Training Set Size (%)", 50, 95, int(st.session_state.config['train_size_perc'] * 100))
            st.session_state.config['train_size_perc'] = train_size / 100
    
    with col4:
        already_split = st.checkbox("Data Already Split", value=st.session_state.config['already_split'])
        st.session_state.config['already_split'] = already_split
        
        data_split_by_patients = st.checkbox("Split by Patient ID", value=st.session_state.config['data_split_by_patients'])
        st.session_state.config['data_split_by_patients'] = data_split_by_patients
        
        if data_split_by_patients and 'available_columns' in st.session_state:
            patient_id_col = st.selectbox("Patient ID Column:", [''] + st.session_state.available_columns)
            st.session_state.config['patient_id_col'] = patient_id_col

def model_selection_page():
    st.markdown('<h2 class="section-header"> Model Selection</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Select the machine learning models you want to include in your analysis. 
        MAIT supports multiple algorithms for comprehensive comparison.
    </div>
    """, unsafe_allow_html=True)
    
    # Available models
    available_models = {
        "RandomForest_mdl": "Random Forest - Ensemble method with decision trees",
        "LightGBM_mdl": "LightGBM - Gradient boosting framework",
        "CatBoost_mdl": "CatBoost - Gradient boosting for categorical features",
        "LogisticRegression_mdl": "Logistic Regression - Linear model for classification",
        "QLattice_mdl": "QLattice - Symbolic regression (requires Feyn)",
        "NaiveBayes_mdl": "Naive Bayes - Probabilistic classifier",
        "HistGBC_mdl": "Histogram-based Gradient Boosting - Fast gradient boosting"
    }
    
    st.subheader("Select Models to Include")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    selected_models = []
    
    with col1:
        for i, (model_key, description) in enumerate(list(available_models.items())[:4]):
            is_selected = model_key in st.session_state.config['models_to_include']
            if st.checkbox(f"**{model_key.replace('_mdl', '')}**", value=is_selected, key=f"model_{i}"):
                selected_models.append(model_key)
            st.caption(description)
            st.write("")
    
    with col2:
        for i, (model_key, description) in enumerate(list(available_models.items())[4:], 4):
            is_selected = model_key in st.session_state.config['models_to_include']
            if st.checkbox(f"**{model_key.replace('_mdl', '')}**", value=is_selected, key=f"model_{i}"):
                selected_models.append(model_key)
            st.caption(description)
            st.write("")
    
    # Update session state with selected models
    final_selected = []
    for model in available_models.keys():
        if model in [st.session_state.get(f"model_{i}") for i in range(len(available_models))] or model in selected_models:
            # Check if checkbox is selected
            for i, (key, _) in enumerate(available_models.items()):
                if key == model and st.session_state.get(f"model_{i}", False):
                    final_selected.append(model)
                    break
    
    # Manual tracking of selected models
    currently_selected = []
    for i, model_key in enumerate(available_models.keys()):
        if st.session_state.get(f"model_{i}", False):
            currently_selected.append(model_key)
    
    st.session_state.config['models_to_include'] = currently_selected
    
    # Analysis type selection
    st.subheader("Analysis Type")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        survival_analysis = st.checkbox("Survival Analysis", value=st.session_state.config['survival_analysis'])
        st.session_state.config['survival_analysis'] = survival_analysis
    
    with col4:
        regression_analysis = st.checkbox("Regression Analysis", value=st.session_state.config['regression_analysis'])
        st.session_state.config['regression_analysis'] = regression_analysis
    
    with col5:
        demo_configs = st.checkbox("Demo Configuration", value=st.session_state.config['demo_configs'])
        st.session_state.config['demo_configs'] = demo_configs
    
    # Show selected models summary
    if currently_selected:
        st.success(f"Selected Models: {', '.join([m.replace('_mdl', '') for m in currently_selected])}")
    else:
        st.warning(" Please select at least one model to proceed.")

def feature_engineering_page():
    st.markdown('<h2 class="section-header"> Feature Engineering</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Selection")
        
        feat_sel = st.checkbox("Enable Feature Selection", value=st.session_state.config['feat_sel'])
        st.session_state.config['feat_sel'] = feat_sel
        
        if feat_sel:
            num_features = st.number_input(
                "Number of Features to Select:", 
                min_value=5, max_value=1000, 
                value=st.session_state.config['num_features_sel']
            )
            st.session_state.config['num_features_sel'] = num_features
        
        st.subheader("Categorical Feature Handling")
        
        merged_rare = st.checkbox("Merge Rare Categories", value=st.session_state.config['merged_rare_categories'])
        st.session_state.config['merged_rare_categories'] = merged_rare
        
        if merged_rare:
            rarity_threshold = st.slider(
                "Rarity Threshold (%):", 
                1, 20, 
                int(st.session_state.config['rarity_threshold'] * 100)
            )
            st.session_state.config['rarity_threshold'] = rarity_threshold / 100
    
    with col2:
        st.subheader("Data Cleaning")
        
        remove_outliers = st.checkbox("Remove Outliers", value=st.session_state.config['remove_outliers'])
        st.session_state.config['remove_outliers'] = remove_outliers
        
        filter_missing = st.checkbox("Filter Highly Missing Features", value=st.session_state.config['filter_highly_mis_feats'])
        st.session_state.config['filter_highly_mis_feats'] = filter_missing
        
        exclude_missing_cols = st.checkbox("Exclude Highly Missing Columns", value=st.session_state.config['exclude_highly_missing_columns'])
        st.session_state.config['exclude_highly_missing_columns'] = exclude_missing_cols
        
        exclude_missing_rows = st.checkbox("Exclude Highly Missing Rows", value=st.session_state.config['exclude_highly_missing_rows'])
        st.session_state.config['exclude_highly_missing_rows'] = exclude_missing_rows
        
        st.subheader("Data Preprocessing")
        
        scale_data = st.checkbox("Scale Data", value=st.session_state.config['scale_data'])
        st.session_state.config['scale_data'] = scale_data
        
        oversampling = st.checkbox("Enable Oversampling", value=st.session_state.config['oversampling'])
        st.session_state.config['oversampling'] = oversampling
        
        shorten_names = st.checkbox("Shorten Feature Names", value=st.session_state.config['shorten_feature_names'])
        st.session_state.config['shorten_feature_names'] = shorten_names

def training_parameters_page():
    st.markdown('<h2 class="section-header"> Training Parameters</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hyperparameter Tuning")
        
        hp_tuning = st.checkbox("Enable Hyperparameter Tuning", value=st.session_state.config['hp_tuning'])
        st.session_state.config['hp_tuning'] = hp_tuning
        
        if hp_tuning:
            n_iter = st.number_input(
                "Number of Tuning Iterations:", 
                min_value=5, max_value=100, 
                value=st.session_state.config['n_iter_hptuning']
            )
            st.session_state.config['n_iter_hptuning'] = n_iter
            
            tuning_score = st.selectbox(
                "Tuning Score Metric:",
                ['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'average_precision'],
                index=['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'average_precision'].index(st.session_state.config['tun_score'])
            )
            st.session_state.config['tun_score'] = tuning_score
        
        st.subheader("Cross Validation")
        
        cv_folds = st.number_input(
            "Cross Validation Folds:", 
            min_value=3, max_value=10, 
            value=st.session_state.config['cv_folds']
        )
        st.session_state.config['cv_folds'] = cv_folds
        
        test_best_only = st.checkbox("Test Only Best CV Model", value=st.session_state.config['test_only_best_cvmodel'])
        st.session_state.config['test_only_best_cvmodel'] = test_best_only
    
    with col2:
        st.subheader("Resource Configuration")
        
        n_cpu_tuning = st.number_input(
            "CPUs for Tuning:", 
            min_value=1, max_value=32, 
            value=st.session_state.config['n_cpu_for_tuning']
        )
        st.session_state.config['n_cpu_for_tuning'] = n_cpu_tuning
        
        n_cpu_training = st.number_input(
            "CPUs for Training:", 
            min_value=1, max_value=32, 
            value=st.session_state.config['n_cpu_model_training']
        )
        st.session_state.config['n_cpu_model_training'] = n_cpu_training
        
        gpu_available = st.checkbox("GPU Available", value=st.session_state.config['GPU_avail'])
        st.session_state.config['GPU_avail'] = gpu_available
        
        if gpu_available:
            st.info(" GPU acceleration will be used for compatible models (LightGBM)")

def output_settings_page():
    st.markdown('<h2 class="section-header"> Output Settings</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Output Configuration")
        
        folder_name = st.text_input(
            "Results Folder Name:", 
            value=st.session_state.config['main_folder_name']
        )
        st.session_state.config['main_folder_name'] = folder_name
        
        fig_format = st.selectbox(
            "Figure File Format:",
            ['png', 'pdf', 'svg', 'tif', 'jpg'],
            index=['png', 'pdf', 'svg', 'tif', 'jpg'].index(st.session_state.config['fig_file_format'])
        )
        st.session_state.config['fig_file_format'] = fig_format
    
    with col2:
        st.subheader("Display Settings")
        
        # Class labels
        st.write("Class Labels for Display:")
        class_label_0 = st.text_input("Class 0 Label:", value=st.session_state.config['class_labels_display'][0])
        class_label_1 = st.text_input("Class 1 Label:", value=st.session_state.config['class_labels_display'][1])
        st.session_state.config['class_labels_display'] = [class_label_0, class_label_1]
    
    # Preview output structure
    st.subheader("Output Structure Preview")
    st.markdown(f"""
    ```
    {folder_name}/
    ├── models/
    │   ├── trained_models.pkl
    │   └── model_performance.json
    ├── figures/
    │   ├── feature_importance.{fig_format}
    │   ├── roc_curves.{fig_format}
    │   └── confusion_matrices.{fig_format}
    ├── reports/
    │   ├── model_comparison.html
    │   └── feature_analysis.csv
    └── logs/
        └── pipeline_execution.log
    ```
    """)

def advanced_options_page():
    st.markdown('<h2 class="section-header">🔬 Advanced Options</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <strong>Advanced Settings:</strong><br>
        These options are for advanced users. Modify only if you understand their implications.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("External Validation")
        
        external_val = st.checkbox("Enable External Validation", value=st.session_state.config['external_val'])
        st.session_state.config['external_val'] = external_val
        
        ext_val_demo = st.checkbox("External Validation Demo", value=st.session_state.config['ext_val_demo'])
        st.session_state.config['ext_val_demo'] = ext_val_demo
        
        st.subheader("Multi-stratification")
        
        multi_strats = st.checkbox("Multiple Stratification Variables", value=st.session_state.config['data_split_multi_strats'])
        st.session_state.config['data_split_multi_strats'] = multi_strats
        
        if multi_strats:
            st.info("Configure stratification variables in the generated notebook")
    
    with col2:
        st.subheader("Custom Configuration")
        
        # Allow users to edit raw config
        if st.button("Show Raw Configuration"):
            st.json(st.session_state.config)
        
        # Configuration file upload/download
        st.subheader("Configuration Management")
        
        # Download current config
        config_json = json.dumps(st.session_state.config, indent=2)
        st.download_button(
            label="Download Configuration",
            data=config_json,
            file_name="mait_config.json",
            mime="application/json"
        )
        
        # Upload config
        uploaded_config = st.file_uploader("Upload Configuration", type=['json'])
        if uploaded_config:
            try:
                config_data = json.load(uploaded_config)
                st.session_state.config.update(config_data)
                st.success(" Configuration loaded successfully!")
            except Exception as e:
                st.error(f"Error loading configuration: {e}")

def generate_run_page():
    st.markdown('<h2 class="section-header"> Generate Notebook & Setup Instructions</h2>', unsafe_allow_html=True)
    
    # Configuration summary
    st.subheader("Configuration Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data File", st.session_state.config['data_file'] or "Not set")
        st.metric("Outcome Variable", st.session_state.config['outcome_var'] or "Not set")
        st.metric("Selected Models", len(st.session_state.config['models_to_include']))
    
    with col2:
        st.metric("Train Size", f"{st.session_state.config['train_size_perc']:.0%}")
        st.metric("CV Folds", st.session_state.config['cv_folds'])
        st.metric("Feature Selection", "Enabled" if st.session_state.config['feat_sel'] else "Disabled")
    
    with col3:
        st.metric("Output Folder", st.session_state.config['main_folder_name'])
        st.metric("Figure Format", st.session_state.config['fig_file_format'].upper())
        st.metric("GPU Support", "Yes" if st.session_state.config['GPU_avail'] else "No")
    
    # Validation
    errors = validate_configuration()
    
    if errors:
        st.error(" Configuration Errors:")
        for error in errors:
            st.write(f"• {error}")
        return
    
    st.success(" Configuration is valid!")
    
    # Generation options
    st.subheader("Generation Options")
    
    col4, col5 = st.columns(2)
    
    with col4:
        if st.button(" Generate Notebook", type="primary"):
            generate_notebook()
    
    with col5:
        if st.button(" Get Setup Instructions", type="secondary"):
            run_pipeline()
    
    # Additional options
    st.subheader("Additional Options")
    
    col6, col7 = st.columns(2)
    
    with col6:
        if st.button(" Save Configuration"):
            save_configuration()
    
    with col7:
        if st.button(" Copy Parameters"):
            copy_parameters_to_clipboard()

def validate_configuration():
    """Validate the current configuration"""
    errors = []
    
    config = st.session_state.config
    
    if not config['data_file']:
        errors.append("Data file is required")
    
    if not config['outcome_var']:
        errors.append("Outcome variable is required")
    
    if not config['models_to_include']:
        errors.append("At least one model must be selected")
    
    if not config['main_folder_name']:
        errors.append("Output folder name is required")
    
    if config['feat_sel'] and config['num_features_sel'] <= 0:
        errors.append("Number of features must be greater than 0")
    
    if config['train_size_perc'] <= 0 or config['train_size_perc'] >= 1:
        errors.append("Train size percentage must be between 0 and 1")
    
    return errors

def generate_notebook():
    """Generate a Jupyter notebook with the current configuration"""
    try:
        notebook_content = create_notebook_content()
        
        # Save notebook
        output_path = f"generated_pipeline_{st.session_state.config['main_folder_name']}.ipynb"
        
        with open(output_path, 'w') as f:
            f.write(notebook_content)
        
        st.success(f" Notebook generated successfully!")
        st.info(f" Saved to: {output_path}")
        
        # Download button
        st.download_button(
            label=" Download Notebook",
            data=notebook_content,
            file_name=f"mait_pipeline_{st.session_state.config['main_folder_name']}.ipynb",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f" Error generating notebook: {e}")

def create_notebook_content():
    """Create notebook content based on configuration by copying and modifying tutorial notebook"""
    config = st.session_state.config
    
    # Read the tutorial notebook (JSON format)
    tutorial_path = "Tutorials/MAIT_Tutorial_Azithromycin_pub.ipynb"
    
    try:
        with open(tutorial_path, 'r', encoding='utf-8') as f:
            tutorial_content = f.read()
        
        # Apply configuration replacements to the entire notebook content
        modified_content = apply_config_replacements(tutorial_content, config)
        return modified_content
        
    except Exception as e:
        st.error(f"Error reading tutorial notebook: {e}")
        # Fallback to simple template
        return create_simple_notebook_template(config)

def apply_config_replacements(notebook_content, config):
    """Apply configuration replacements to the notebook content"""
    
    # Parse the JSON notebook
    import json
    notebook_data = json.loads(notebook_content)
    
    # Create a list of replacements to apply
    replacements = {
        # Data configuration
        '"combined_data_Azithromycin.csv"': f'"{config["data_file"]}"',
        '"azm_sr"': f'"{config["outcome_var"]}"',
        'cat_features = []': f'cat_features = {config["cat_features"]}',
        'columns_to_drop = [': f'columns_to_drop = {config["columns_to_drop"]} # Original: [',
        
        # Model selection
        '"QLattice_mdl", "NaiveBayes_mdl", "RandomForest_mdl", "LightGBM_mdl", "CatBoost_mdl", "LogisticRegression_mdl", "HistGBC_mdl"': ', '.join([f'"{m}"' for m in config["models_to_include"]]),
        
        # Results folder
        "'results_Azithromycin'": f"'{config['main_folder_name']}'",
        
        # Data split settings
        'data_split = True': f'data_split = {config["data_split"]}',
        'train_size_perc = 0.8': f'train_size_perc = {config["train_size_perc"]}',
        'data_split_by_patients = False': f'data_split_by_patients = {config["data_split_by_patients"]}',
        'data_split_multi_strats = False': f'data_split_multi_strats = {config["data_split_multi_strats"]}',
        'already_split = False': f'already_split = {config["already_split"]}',
        
        # Feature selection
        'feat_sel = True': f'feat_sel = {config["feat_sel"]}',
        'num_features_sel = 30': f'num_features_sel = {config["num_features_sel"]}',
        
        # External validation
        'external_val = False': f'external_val = {config["external_val"]}',
        'ext_val_demo = False': f'ext_val_demo = {config["ext_val_demo"]}',
        
        # Processing settings
        'merged_rare_categories = True': f'merged_rare_categories = {config["merged_rare_categories"]}',
        'rarity_threshold = 0.05': f'rarity_threshold = {config["rarity_threshold"]}',
        'remove_outliers = False': f'remove_outliers = {config["remove_outliers"]}',
        
        # Resource configuration
        'GPU_avail = False': f'GPU_avail = {config["GPU_avail"]}',
        'hp_tuning = True': f'hp_tuning = {config["hp_tuning"]}',
        'n_cpu_for_tuning = 4': f'n_cpu_for_tuning = {config["n_cpu_for_tuning"]}',
        'n_cpu_model_training = 4': f'n_cpu_model_training = {config["n_cpu_model_training"]}',
        'n_iter_hptuning = 10': f'n_iter_hptuning = {config["n_iter_hptuning"]}',
        
        # Cross validation
        'cv_folds = 5': f'cv_folds = {config["cv_folds"]}',
        'test_only_best_cvmodel = True': f'test_only_best_cvmodel = {config["test_only_best_cvmodel"]}',
        
        # Missing data handling
        'exclude_highly_missing_columns = True': f'exclude_highly_missing_columns = {config["exclude_highly_missing_columns"]}',
        'exclude_highly_missing_rows = True': f'exclude_highly_missing_rows = {config["exclude_highly_missing_rows"]}',
        
        # Analysis types
        'survival_analysis = False': f'survival_analysis = {config["survival_analysis"]}',
        'regression_analysis = False': f'regression_analysis = {config["regression_analysis"]}',
    }
    
    # Convert notebook back to string to apply replacements
    notebook_str = json.dumps(notebook_data, indent=1)
    
    # Apply all replacements
    for old_value, new_value in replacements.items():
        notebook_str = notebook_str.replace(old_value, new_value)
    
    # Add a comment at the top of the first code cell to indicate it's been customized
    customization_comment = (f'\\n\\n#  MAIT Pipeline - Customized via Streamlit Interface\\n'
                            f'# Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}\\n'
                            f'# Configuration applied for: {config["main_folder_name"]}\\n'
                            f'# Data: {config["data_file"]} | Outcome: {config["outcome_var"]}\\n'
                            f'# Models: {", ".join([m.replace("_mdl", "") for m in config["models_to_include"]])}\\n')
    
    # Find the first code cell and add the comment
    notebook_data = json.loads(notebook_str)
    for cell in notebook_data["cells"]:
        if cell["cell_type"] == "code":
            if isinstance(cell["source"], list):
                cell["source"].insert(0, customization_comment)
            else:
                cell["source"] = customization_comment + cell["source"]
            break
    
    # Update the notebook title
    if notebook_data["cells"] and notebook_data["cells"][0]["cell_type"] == "markdown":
        title_content = (f'# MAIT Pipeline - {config["main_folder_name"]}\\n\\n'
                        f'**Generated automatically using MAIT Streamlit Configuration Interface**\\n\\n'
                        f'This notebook has been customized with your specific configuration parameters.\\n\\n'
                        f'## Configuration Summary\\n'
                        f'- **Data File**: {config["data_file"]}\\n'
                        f'- **Outcome Variable**: {config["outcome_var"]}\\n'
                        f'- **Selected Models**: {", ".join([m.replace("_mdl", "") for m in config["models_to_include"]])}\\n'
                        f'- **Feature Selection**: {"Enabled" if config["feat_sel"] else "Disabled"} ({config["num_features_sel"]} features)\\n'
                        f'- **Data Split**: {config["train_size_perc"]:.0%} training, {1-config["train_size_perc"]:.0%} testing\\n'
                        f'- **Cross Validation**: {config["cv_folds"]} folds\\n'
                        f'- **Hyperparameter Tuning**: {"Enabled" if config["hp_tuning"] else "Disabled"}\\n'
                        f'- **GPU Support**: {"Yes" if config["GPU_avail"] else "No"}\\n'
                        f'- **Output Folder**: {config["main_folder_name"]}\\n\\n'
                        f'## Instructions\\n'
                        f'1. **Review the configuration** in the parameter cells below\\n'
                        f'2. **Run all cells sequentially** to execute the complete MAIT pipeline\\n'
                        f'3. **Check the results** in the generated output folder\\n'
                        f'4. **Modify parameters** if needed and re-run specific sections\\n\\n'
                        f'---\\n\\n'
                        f'*This pipeline is based on the MAIT framework. Please cite the MAIT paper if you use this in your research.*\\n')
        
        if isinstance(notebook_data["cells"][0]["source"], list):
            notebook_data["cells"][0]["source"] = [title_content]
        else:
            notebook_data["cells"][0]["source"] = title_content
    
    return json.dumps(notebook_data, indent=1)

def generate_config_cell_content(config):
    """Generate the configuration cell content based on user settings"""
    
    # Handle data split by patients
    patient_split_code = ""
    if config['data_split_by_patients']:
        patient_split_code = f"""
if data_split_by_patients:
    patient_id_col = "{config['patient_id_col']}"  # the column name that contains patient ID"""
    
    # Handle multi-stratification
    multi_strat_code = ""
    if config['data_split_multi_strats']:
        multi_strat_code = """
if data_split_multi_strats:  # the names of the columns used for multiple stratification should be specified by user
    strat_var1 = "stratification variable 1"  # UPDATE THIS with your stratification variable"""
    
    # Handle already split data
    already_split_code = ""
    if config['already_split']:
        already_split_code = """
if already_split:  # specify the names of the train (development) and test sets
    # Splitting based on values - UPDATE THE COLUMN NAME AND VALUES
    testset = mydata[mydata['subset'] == 'Test']
    mydata = mydata[mydata['subset'] == 'Train']"""
    
    config_content = f"""# MAIT Configuration Parameters - Customized via Streamlit Interface
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

# settings for categorical variables
cat_features = {config['cat_features']}  # categorical features specified via interface
merged_rare_categories = {config['merged_rare_categories']}  # merge rare categories and unify missing categories
rarity_threshold = {config['rarity_threshold']}  # threshold for rarity (e.g., 0.05 means 5%)

###################################################################################
# specify columns that must be removed
columns_to_drop = {config['columns_to_drop']}  # columns to drop specified via interface

###################################################################################
# import data
mydata = pd.read_csv("{config['data_file']}")  # data file specified via interface
external_val = {config['external_val']}  # external validation enabled/disabled via interface
ext_val_demo = {config['ext_val_demo']}  # external validation demo mode

###################################################################################
# random data split
data_split = {config['data_split']}  # data split enabled/disabled via interface
train_size_perc = {config['train_size_perc']}  # training set percentage specified via interface
data_split_by_patients = {config['data_split_by_patients']}  # patient-based splitting{patient_split_code}
data_split_multi_strats = {config['data_split_multi_strats']}  # multiple stratification variables{multi_strat_code}
already_split = {config['already_split']}  # data already split indicator{already_split_code}

###################################################################################
# available binary classification models in the pipeline
models_to_include = {config['models_to_include']}  # models selected via interface

# outcome variable
outcome_var = "{config['outcome_var']}"  # outcome variable specified via interface

###################################################################################
# set a directory to save the results
main_folder_name = '{config['main_folder_name']}'  # output folder specified via interface
# Define class labels for display
class_labels_display = {config['class_labels_display']}  # class labels specified via interface

# Specify the class labels
class_0 = class_labels_display[0]
class_1 = class_labels_display[1]

# Create a mapping dictionary for class labels
class_label_dict = {{0.0: class_0, 1.0: class_1}}  # mapping for class labels

###################################################################################
# feature selection
feat_sel = {config['feat_sel']}  # feature selection enabled/disabled via interface
num_features_sel = {config['num_features_sel']}  # number of features to select specified via interface
top_n_f = 20  # number of top features for SHAP plots (default)

###################################################################################
# survival analysis
survival_analysis = {config['survival_analysis']}  # survival analysis enabled/disabled via interface
if survival_analysis:
    survival_demo = False  # demo mode for survival analysis
    time_to_event_column = ""  # UPDATE THIS with your time-to-event column name
    if survival_demo: 
        mydata[time_to_event_column] = np.random.randint(90, 366, size=len(mydata))
    mydata_copy_survival = mydata.copy()

###################################################################################
# regression analysis
regression_analysis = {config['regression_analysis']}  # regression analysis enabled/disabled via interface
if regression_analysis:
    regression_outcome = "regression_outcome_var"  # UPDATE THIS with your regression outcome variable
    demo_regression_analysis = False  # demo mode for regression analysis
    if demo_regression_analysis:
        mydata_copy_regression = mydata.copy()
        X = np.random.randn(mydata_copy_regression.shape[0], mydata_copy_regression.shape[1])
        true_calculate = np.random.randn(mydata_copy_regression.shape[1])
        noise = np.random.randn(mydata_copy_regression.shape[0]) * 0.5
        mydata_copy_regression[regression_outcome] = np.dot(X, true_calculate) + noise

###################################################################################
# settings for processing resources
GPU_avail = {config['GPU_avail']}  # GPU availability specified via interface
hp_tuning = {config['hp_tuning']}  # hyperparameter tuning enabled/disabled via interface
n_cpu_for_tuning = {config['n_cpu_for_tuning']}  # CPUs for tuning specified via interface
n_cpu_model_training = {config['n_cpu_model_training']}  # CPUs for training specified via interface
n_rep_feature_permutation = 100  # number of repetitions for feature permutation (default)
n_iter_hptuning = {config['n_iter_hptuning']}  # iterations for hyperparameter tuning specified via interface
SEED = 123  # random seed for reproducibility

###################################################################################
cv_folds = {config['cv_folds']}  # cross validation folds specified via interface
cv_folds_hptuning = 5  # folds for hyperparameter tuning (default)
use_default_threshold = True  # use default threshold of 0.5 for binary classification
test_only_best_cvmodel = {config['test_only_best_cvmodel']}  # test only best CV model specified via interface

###################################################################################
# handle missingness
exclude_highly_missing_columns = {config['exclude_highly_missing_columns']}  # exclude highly missing columns via interface
exclude_highly_missing_rows = {config['exclude_highly_missing_rows']}  # exclude highly missing rows via interface
column_threshold = 0.99  # threshold for variables - columns (default)
row_threshold = 0.90     # threshold for samples - rows (default)

###################################################################################
remove_outliers = {config['remove_outliers']}  # outlier removal enabled/disabled via interface

###################################################################################
# Specify the filename of this Jupyter notebook
JupyterNotebook_filename = "MAIT_Pipeline_{config['main_folder_name']}.ipynb"

print(" Configuration loaded successfully!")
print(f" Data file: {{config['data_file']}}")
print(f" Outcome variable: {{config['outcome_var']}}")
print(f" Models: {{', '.join([m.replace('_mdl', '') for m in config['models_to_include']])}}")
print(f" Output folder: {{config['main_folder_name']}}")"""

    return config_content

def create_simple_notebook_template(config):
    """Create a simple notebook template as fallback"""
    template = f"""<VSCode.Cell language="markdown">
# MAIT Pipeline - {config['main_folder_name']}

Generated automatically using MAIT Streamlit Configuration Interface

## Configuration Summary
- **Data File**: {config['data_file']}
- **Outcome Variable**: {config['outcome_var']}
- **Models**: {', '.join(config['models_to_include'])}
- **Output Folder**: {config['main_folder_name']}
</VSCode.Cell>
<VSCode.Cell language="python">
# Import libraries and load data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Add other imports as needed

# Load data
mydata = pd.read_csv('{config['data_file']}')

# Configuration parameters
outcome_var = '{config['outcome_var']}'
columns_to_drop = {config['columns_to_drop']}
cat_features = {config['cat_features']}
models_to_include = {config['models_to_include']}
main_folder_name = '{config['main_folder_name']}'
train_size_perc = {config['train_size_perc']}
data_split = {config['data_split']}
feat_sel = {config['feat_sel']}
num_features_sel = {config['num_features_sel']}
hp_tuning = {config['hp_tuning']}
cv_folds = {config['cv_folds']}
GPU_avail = {config['GPU_avail']}
</VSCode.Cell>
<VSCode.Cell language="markdown">
## Next Steps

1. Review the configuration parameters above
2. Add the complete MAIT pipeline code (copy from tutorial notebooks)
3. Execute the cells to run your analysis

**Note**: This is a simplified template. For the complete pipeline, please copy the full tutorial notebook.
</VSCode.Cell>"""
    
    return template

def run_pipeline():
    """Provide clear instructions for running the pipeline manually"""
    st.info(" **Manual Setup Instructions** - Follow these steps to run your configured MAIT pipeline:")
    
    config = st.session_state.config
    tutorial_path = "Tutorials/MAIT_Tutorial_Azithromycin_pub.ipynb"
    
    # Step 1: Copy template
    st.markdown("### Step 1: Copy the Template Notebook")
    st.markdown(f"""
    1. **Copy** the template notebook: `{tutorial_path}`
    2. **Rename** it to: `MAIT_Pipeline_{config['main_folder_name']}.ipynb`
    3. **Open** the copied notebook in Jupyter Notebook or VS Code
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Open Tutorials Folder"):
            try:
                subprocess.Popen(["xdg-open", "Tutorials"])
                st.success(" Opening tutorials folder...")
            except Exception as e:
                st.info("Please navigate to: Tutorials/")
    
    with col2:
        if st.button(" Copy Template Path"):
            st.code(tutorial_path)
            st.info("Copy the path above to navigate to the template")
    
    # Step 2: Update configurations
    st.markdown("### Step 2: Update Configuration Values")
    st.markdown("In your copied notebook, **find and replace** the following values in the configuration cell:")
    
    # Create two columns for the find/replace table
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("** Find (Original Value):**")
        original_values = [
            "outcome_var = 'Azithromycin_Use'",
            "columns_to_drop = ['Patient_ID']",
            "main_folder_name = 'Azithromycin_Analysis'",
            "train_size_perc = 0.8",
            "feat_sel = True",
            "num_features_sel = 50",
            "cv_folds = 5",
            "hp_tuning = True",
            "n_iter_hptuning = 50",
            "GPU_avail = False",
            "oversampling = False",
            "scale_data = True"
        ]
        for val in original_values:
            st.code(val, language="python")
    
    with col_right:
        st.markdown("** Replace With (Your Configuration):**")
        new_values = [
            f"outcome_var = '{config['outcome_var']}'",
            f"columns_to_drop = {config['columns_to_drop']}",
            f"main_folder_name = '{config['main_folder_name']}'",
            f"train_size_perc = {config['train_size_perc']}",
            f"feat_sel = {config['feat_sel']}",
            f"num_features_sel = {config['num_features_sel']}",
            f"cv_folds = {config['cv_folds']}",
            f"hp_tuning = {config['hp_tuning']}",
            f"n_iter_hptuning = {config['n_iter_hptuning']}",
            f"GPU_avail = {config['GPU_avail']}",
            f"oversampling = {config['oversampling']}",
            f"scale_data = {config['scale_data']}"
        ]
        for val in new_values:
            st.code(val, language="python")
    
    # Step 3: Update additional settings
    if config['models_to_include']:
        st.markdown("### Step 3: Update Model Selection")
        st.markdown("Find the `models_to_include` list and replace it with:")
        st.code(f"models_to_include = {config['models_to_include']}", language="python")
    
    if config['cat_features']:
        st.markdown("### Step 4: Update Categorical Features")
        st.markdown("Find the `cat_features` list and replace it with:")
        st.code(f"cat_features = {config['cat_features']}", language="python")
    
    # Step 5: Update data file path
    st.markdown("### Step 5: Update Data File Path")
    st.markdown("Find the data loading cell and update the file path to:")
    st.code(f"# Update this line in the data loading cell:\ndf = pd.read_csv('{config['data_file']}')", language="python")
    
    # Step 6: Run the pipeline
    st.markdown("### Step 6: Run Your Pipeline")
    st.markdown("""
    1. **Save** your updated notebook
    2. **Run all cells** sequentially (or use "Run All" in Jupyter)
    3. **Monitor progress** - the pipeline will create output folder: `{}`
    4. **Check results** in the generated reports and visualizations
    """.format(config['main_folder_name']))
    
    # Quick copy section for all parameters
    st.markdown("###  Quick Copy: All Your Parameters")
    with st.expander("Click to see all parameters for easy copying"):
        parameters_text = f"""
# Your MAIT Configuration
outcome_var = '{config['outcome_var']}'
columns_to_drop = {config['columns_to_drop']}
cat_features = {config['cat_features']}
models_to_include = {config['models_to_include']}
main_folder_name = '{config['main_folder_name']}'
train_size_perc = {config['train_size_perc']}
feat_sel = {config['feat_sel']}
num_features_sel = {config['num_features_sel']}
hp_tuning = {config['hp_tuning']}
n_iter_hptuning = {config['n_iter_hptuning']}
cv_folds = {config['cv_folds']}
tun_score = '{config['tun_score']}'
GPU_avail = {config['GPU_avail']}
oversampling = {config['oversampling']}
scale_data = {config['scale_data']}
merged_rare_categories = {config['merged_rare_categories']}
rarity_threshold = {config['rarity_threshold']}
remove_outliers = {config['remove_outliers']}

# Data file path (update in data loading cell):
# df = pd.read_csv('{config['data_file']}')
"""
        st.code(parameters_text, language="python")
        st.info(" **Tip:** Copy this entire block and replace the configuration section in your notebook")
    
    # Summary
    st.success(" **Ready to go!** Follow the steps above to run your customized MAIT pipeline.")

def save_configuration():
    """Save configuration to file"""
    config_path = f"configs/config_{st.session_state.config['main_folder_name']}.json"
    
    # Create configs directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(st.session_state.config, f, indent=2)
    
    st.success(f" Configuration saved to: {config_path}")

def copy_parameters_to_clipboard():
    """Generate parameter code for copying"""
    config = st.session_state.config
    
    parameters_code = f"""
# MAIT Configuration Parameters
outcome_var = '{config['outcome_var']}'
columns_to_drop = {config['columns_to_drop']}
cat_features = {config['cat_features']}
models_to_include = {config['models_to_include']}
main_folder_name = '{config['main_folder_name']}'
data_split = {config['data_split']}
train_size_perc = {config['train_size_perc']}
feat_sel = {config['feat_sel']}
num_features_sel = {config['num_features_sel']}
hp_tuning = {config['hp_tuning']}
n_iter_hptuning = {config['n_iter_hptuning']}
cv_folds = {config['cv_folds']}
tun_score = '{config['tun_score']}'
GPU_avail = {config['GPU_avail']}
oversampling = {config['oversampling']}
scale_data = {config['scale_data']}
merged_rare_categories = {config['merged_rare_categories']}
rarity_threshold = {config['rarity_threshold']}
remove_outliers = {config['remove_outliers']}
"""
    
    st.code(parameters_code, language='python')
    st.info(" Copy the code above and paste it into your MAIT notebook")

if __name__ == "__main__":
    main()
