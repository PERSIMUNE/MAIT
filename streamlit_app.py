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
        Configure your parameters below and export a ready-to-run Python script for your pipeline.
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
        "Finalize and Export"
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
    elif page == "Finalize and Export":
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
            # Filter out outcome variable and columns to drop from available options
            available_for_cat = [col for col in st.session_state.available_columns 
                               if col != st.session_state.config['outcome_var'] 
                               and col not in st.session_state.config['columns_to_drop']]
            
            # Initialize widget state if not exists or validate current selection
            if "categorical_features" not in st.session_state:
                # Get current categorical features, ensuring they're still valid
                valid_cat_features = [f for f in st.session_state.config['cat_features'] if f in available_for_cat]
                st.session_state["categorical_features"] = valid_cat_features
            
            cat_features = st.multiselect(
                "Categorical Features:", 
                available_for_cat,
                key="categorical_features",
                help="Select all columns that contain categorical data (e.g., gender, smoking_status, race)"
            )
            
            # Update configuration from the widget
            st.session_state.config['cat_features'] = cat_features
            
            # Show current selection
            if cat_features:
                st.info(f"✅ Selected {len(cat_features)} categorical feature(s): {', '.join(cat_features)}")
        
        # Columns to drop
        if 'available_columns' in st.session_state:
            # Filter out outcome variable from available options
            available_for_drop = [col for col in st.session_state.available_columns 
                                if col != st.session_state.config['outcome_var']]
            
            # Initialize widget state if not exists or validate current selection
            if "columns_to_drop" not in st.session_state:
                # Get current columns to drop, ensuring they're still valid
                valid_cols_to_drop = [f for f in st.session_state.config['columns_to_drop'] if f in available_for_drop]
                st.session_state["columns_to_drop"] = valid_cols_to_drop
            
            columns_to_drop = st.multiselect(
                "Columns to Drop:", 
                available_for_drop,
                key="columns_to_drop",
                help="Select columns that should be excluded from the analysis"
            )
            
            # Update configuration from the widget
            st.session_state.config['columns_to_drop'] = columns_to_drop
            
            # Show current selection
            if columns_to_drop:
                st.info(f"🗑️ Will drop {len(columns_to_drop)} column(s): {', '.join(columns_to_drop)}")
    
    # Debug info section (collapsible)
    with st.expander("🔍 Current Configuration Status", expanded=False):
        col_debug1, col_debug2 = st.columns(2)
        
        with col_debug1:
            st.write("**Current Settings:**")
            st.write(f"• Data file: `{st.session_state.config['data_file']}`")
            st.write(f"• Outcome variable: `{st.session_state.config['outcome_var']}`")
            if 'available_columns' in st.session_state:
                st.write(f"• Total columns available: {len(st.session_state.available_columns)}")
        
        with col_debug2:
            st.write("**Feature Selection:**")
            st.write(f"• Categorical features: {st.session_state.config['cat_features']}")
            st.write(f"• Columns to drop: {st.session_state.config['columns_to_drop']}")
            if 'available_columns' in st.session_state:
                remaining_cols = len(st.session_state.available_columns) - len(st.session_state.config['columns_to_drop']) - (1 if st.session_state.config['outcome_var'] else 0)
                st.write(f"• Features for modeling: ~{remaining_cols}")
    
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
    st.markdown('<h2 class="section-header"> Generate Python Script & Setup Instructions</h2>', unsafe_allow_html=True)
    
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
        if st.button("💾 Save Executable Python File", type="primary"):
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
    """Generate a complete Python script with the current configuration"""
    try:
        # Generate the complete Python script
        python_content, error = generate_python_script(st.session_state.config)
        
        if error:
            st.error(f"⚠️ Error generating script: {error}")
            return
        
        # Create filename based on configuration
        output_filename = f"mait_pipeline_{st.session_state.config['main_folder_name']}.py"
        
        # Save the Python script
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(python_content)
        
        st.success(f"✅ Python script exported successfully!")
        st.info(f"📁 Saved to: {output_filename}")
        
        # Show configuration summary
        with st.expander("📋 Configuration Summary", expanded=False):
            config = st.session_state.config
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Configuration:**")
                st.write(f"• Data file: `{config['data_file']}`")
                st.write(f"• Outcome variable: `{config['outcome_var']}`")
                st.write(f"• Categorical features: {len(config['cat_features'])} selected")
                st.write(f"• Data split: {config['train_size_perc']:.0%} train / {1-config['train_size_perc']:.0%} test")
                
                st.write("**Model Configuration:**")
                models_list = [m.replace('_mdl', '') for m in config['models_to_include']]
                st.write(f"• Selected models: {', '.join(models_list)}")
                st.write(f"• Feature selection: {'Enabled' if config['feat_sel'] else 'Disabled'}")
                if config['feat_sel']:
                    st.write(f"• Features to select: {config['num_features_sel']}")
            
            with col2:
                st.write("**Processing Configuration:**")
                st.write(f"• Cross validation: {config['cv_folds']} folds")
                st.write(f"• Hyperparameter tuning: {'Enabled' if config['hp_tuning'] else 'Disabled'}")
                st.write(f"• GPU support: {'Yes' if config['GPU_avail'] else 'No'}")
                st.write(f"• CPUs for training: {config['n_cpu_model_training']}")
                
                st.write("**Analysis Options:**")
                st.write(f"• Survival analysis: {'Enabled' if config['survival_analysis'] else 'Disabled'}")
                st.write(f"• Regression analysis: {'Enabled' if config['regression_analysis'] else 'Disabled'}")
                st.write(f"• External validation: {'Enabled' if config['external_val'] else 'Disabled'}")
        
        # Download button
        st.download_button(
            label="⬇️ Download Python Script",
            data=python_content,
            file_name=output_filename,
            mime="text/plain",
            help="Download the exported Python script to run your MAIT pipeline"
        )
        
        # Instructions
        with st.expander("📖 Usage Instructions", expanded=True):
            st.markdown(f"""
**How to use your exported Python script:**

1. **Download the script** using the button above or find it in your current directory as `{output_filename}`

2. **Install required packages** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script directly**:
   ```bash
   python {output_filename}
   ```

4. **Or convert to Jupyter notebook** (if you prefer notebooks):
   ```bash
   # Install jupytext if needed: pip install jupytext
   jupytext --to notebook {output_filename}
   ```

5. **Check the results** in the `{config['main_folder_name']}` folder that will be created

**What the script contains:**
- ✅ All your configuration parameters pre-filled
- ✅ Complete MAIT pipeline code from the template
- ✅ Ready to run without manual configuration
- ✅ Compatible with both Python execution and Jupyter notebooks

**Note:** Make sure your data file `{config['data_file']}` is in the same directory as the script, or update the path in the script.
            """)
        
    except Exception as e:
        st.error(f"⚠️ Error generating script: {e}")

def generate_python_script(config):
    """Generate a complete Python script based on the template with user configuration injected."""
    try:
        # Read the Python template
        template_path = "Tutorials/mait_template.py"
        if not os.path.exists(template_path):
            return None, "Python template not found. Please ensure mait_template.py exists in the Tutorials directory."
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Create configuration replacements based on user input
        replacements = create_configuration_replacements(config)
        
        # Apply replacements to the template
        modified_content = template_content
        
        # Apply replacements line by line to ensure accuracy
        lines = modified_content.split('\n')
        modified_lines = []
        
        for line in lines:
            modified_line = line
            for old_value, new_value in replacements.items():
                if old_value in line and not line.strip().startswith('#'):
                    modified_line = line.replace(old_value, new_value)
                    break
            modified_lines.append(modified_line)
        
        modified_content = '\n'.join(modified_lines)
        
        # Add user configuration header
        header_comment = f'''# MAIT Pipeline - Exported by Streamlit Configuration Interface
# Exported on {time.strftime("%Y-%m-%d %H:%M:%S")}
# 
# This file contains your custom MAIT pipeline configuration.
# All parameters have been set according to your Streamlit interface selections.
# 
# CONFIGURATION SUMMARY:
# - Data file: {config['data_file']}
# - Outcome variable: {config['outcome_var']}
# - Selected models: {', '.join([m.replace('_mdl', '') for m in config['models_to_include']])}
# - Feature selection: {'Enabled' if config['feat_sel'] else 'Disabled'} ({config['num_features_sel']} features)
# - Data split: {config['train_size_perc']:.0%} training / {1-config['train_size_perc']:.0%} testing
# - Cross validation: {config['cv_folds']} folds
# - Hyperparameter tuning: {'Enabled' if config['hp_tuning'] else 'Disabled'}
# - GPU support: {'Yes' if config['GPU_avail'] else 'No'}
# - Output folder: {config['main_folder_name']}
#
# To run this script:
# 1. Ensure your data file is in the same directory
# 2. Install required packages: pip install -r requirements.txt
# 3. Run: python {config['main_folder_name']}_pipeline.py
# 
# ==================================================================================

'''
        
        final_content = header_comment + modified_content
        
        return final_content, None
        
    except Exception as e:
        return None, f"Error generating Python script: {str(e)}"

def create_configuration_replacements(config):
    """Create a dictionary of replacements to apply to the template"""
    
    replacements = {}
    
    # Data configuration
    replacements['mydata = pd.read_csv("combined_data_Azithromycin.csv")'] = f'mydata = pd.read_csv("{config["data_file"]}")  # Data file from Streamlit config'
    replacements['outcome_var = "azm_sr"'] = f'outcome_var = "{config["outcome_var"]}"  # Outcome variable from Streamlit config'
    replacements['cat_features = []'] = f'cat_features = {config["cat_features"]}  # Categorical features from Streamlit config'
    replacements['columns_to_drop = []'] = f'columns_to_drop = {config["columns_to_drop"]}  # Columns to drop from Streamlit config'
    
    # Model selection
    models_str = ', '.join([f'"{m}"' for m in config["models_to_include"]])
    replacements['models_to_include = ["QLattice_mdl", "NaiveBayes_mdl", "RandomForest_mdl", "LightGBM_mdl", "CatBoost_mdl", "LogisticRegression_mdl", "HistGBC_mdl"]'] = f'models_to_include = [{models_str}]  # Models from Streamlit config'
    
    # Results folder
    replacements["main_folder_name = 'results_Azithromycin'"] = f"main_folder_name = '{config['main_folder_name']}'  # Output folder from Streamlit config"
    
    # Class labels
    class_labels_str = str(config['class_labels_display'])
    replacements["class_labels_display = ['non-resistant', 'resistant']"] = f"class_labels_display = {class_labels_str}  # Class labels from Streamlit config"
    
    # Data split settings
    replacements['data_split = True'] = f'data_split = {config["data_split"]}  # Data split setting from Streamlit config'
    replacements['train_size_perc = 0.8'] = f'train_size_perc = {config["train_size_perc"]}  # Training size from Streamlit config'
    replacements['data_split_by_patients = False'] = f'data_split_by_patients = {config["data_split_by_patients"]}  # Patient split from Streamlit config'
    replacements['data_split_multi_strats = False'] = f'data_split_multi_strats = {config["data_split_multi_strats"]}  # Multi-strat from Streamlit config'
    replacements['already_split = False'] = f'already_split = {config["already_split"]}  # Already split from Streamlit config'
    
    # Feature selection
    replacements['feat_sel = True'] = f'feat_sel = {config["feat_sel"]}  # Feature selection from Streamlit config'
    replacements['num_features_sel = 30'] = f'num_features_sel = {config["num_features_sel"]}  # Number of features from Streamlit config'
    
    # External validation
    replacements['external_val = False'] = f'external_val = {config["external_val"]}  # External validation from Streamlit config'
    replacements['ext_val_demo = False'] = f'ext_val_demo = {config["ext_val_demo"]}  # External validation demo from Streamlit config'
    
    # Processing settings
    replacements['merged_rare_categories = True'] = f'merged_rare_categories = {config["merged_rare_categories"]}  # Merge rare categories from Streamlit config'
    replacements['rarity_threshold = 0.05'] = f'rarity_threshold = {config["rarity_threshold"]}  # Rarity threshold from Streamlit config'
    replacements['remove_outliers = False'] = f'remove_outliers = {config["remove_outliers"]}  # Remove outliers from Streamlit config'
    
    # Resource configuration
    replacements['GPU_avail = True'] = f'GPU_avail = {config["GPU_avail"]}  # GPU availability from Streamlit config'
    replacements['hp_tuning = True'] = f'hp_tuning = {config["hp_tuning"]}  # Hyperparameter tuning from Streamlit config'
    replacements['n_cpu_for_tuning = 20'] = f'n_cpu_for_tuning = {config["n_cpu_for_tuning"]}  # CPUs for tuning from Streamlit config'
    replacements['n_cpu_model_training = 20'] = f'n_cpu_model_training = {config["n_cpu_model_training"]}  # CPUs for training from Streamlit config'
    replacements['n_iter_hptuning = 10'] = f'n_iter_hptuning = {config["n_iter_hptuning"]}  # Tuning iterations from Streamlit config'
    
    # Cross validation
    replacements['cv_folds = 5'] = f'cv_folds = {config["cv_folds"]}  # CV folds from Streamlit config'
    replacements['test_only_best_cvmodel = True'] = f'test_only_best_cvmodel = {config["test_only_best_cvmodel"]}  # Test best model from Streamlit config'
    
    # Missing data handling
    replacements['exclude_highly_missing_columns = True'] = f'exclude_highly_missing_columns = {config["exclude_highly_missing_columns"]}  # Exclude missing columns from Streamlit config'
    replacements['exclude_highly_missing_rows = True'] = f'exclude_highly_missing_rows = {config["exclude_highly_missing_rows"]}  # Exclude missing rows from Streamlit config'
    
    # Analysis types
    replacements['survival_analysis = False'] = f'survival_analysis = {config["survival_analysis"]}  # Survival analysis from Streamlit config'
    replacements['regression_analysis = False'] = f'regression_analysis = {config["regression_analysis"]}  # Regression analysis from Streamlit config'
    
    # Demo settings (turn off for real use)
    replacements['demo_configs = True'] = 'demo_configs = False  # Demo configs disabled for real use'
    
    return replacements

def run_pipeline():
    """Provide clear instructions for running the pipeline manually"""
    st.info("📋 **Manual Setup Instructions** - Follow these steps to run your configured MAIT pipeline:")
    
    config = st.session_state.config
    
    # Step 1: Template guidance
    st.markdown("### Step 1: Using the Exported Script")
    st.markdown(f"""
    The exported Python script (`mait_pipeline_{config['main_folder_name']}.py`) contains:
    - ✅ Complete MAIT pipeline code
    - ✅ Your configuration parameters pre-filled
    - ✅ Ready to run without manual editing
    """)
    
    # Step 2: Execution instructions
    st.markdown("### Step 2: Run Your Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option A: Direct Python Execution**")
        st.code(f"""
# Navigate to your MAIT directory
cd /path/to/MAIT

# Run the exported script
python mait_pipeline_{config['main_folder_name']}.py
        """, language="bash")
    
    with col2:
        st.markdown("**Option B: Convert to Jupyter Notebook**")
        st.code(f"""
# Install jupytext if needed
pip install jupytext

# Convert to notebook
jupytext --to notebook mait_pipeline_{config['main_folder_name']}.py

# Open in Jupyter
jupyter notebook mait_pipeline_{config['main_folder_name']}.ipynb
        """, language="bash")
    
    # Step 3: Expected output
    st.markdown("### Step 3: Expected Output")
    st.markdown(f"""
    After running, you should see:
    - 📁 **Results folder**: `{config['main_folder_name']}/`
    - 📊 **Performance reports** with model comparisons
    - 📈 **Visualizations** (ROC curves, feature importance, etc.)
    - 🤖 **Trained models** saved for future use
    """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting", expanded=False):
        st.markdown("""
        **Common Issues:**
        
        1. **Missing packages**: Install requirements with `pip install -r requirements.txt`
        2. **Data file not found**: Ensure your data file is in the same directory as the script
        3. **Permission errors**: Check file permissions and disk space
        4. **Memory issues**: Reduce dataset size or increase system memory
        
        **Getting Help:**
        - Check the MAIT documentation on GitHub
        - Review the generated script comments for parameter explanations
        - Ensure all dependencies are properly installed
        """)

def save_configuration():
    """Save configuration to file"""
    try:
        config_path = f"configs/config_{st.session_state.config['main_folder_name']}.json"
        
        # Create configs directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(st.session_state.config, f, indent=2)
        
        st.success(f"✅ Configuration saved to: {config_path}")
        
        # Show download option
        config_json = json.dumps(st.session_state.config, indent=2)
        st.download_button(
            label="⬇️ Download Configuration File",
            data=config_json,
            file_name=f"mait_config_{st.session_state.config['main_folder_name']}.json",
            mime="application/json",
            help="Download the configuration file for future use"
        )
        
    except Exception as e:
        st.error(f"⚠️ Error saving configuration: {e}")

def copy_parameters_to_clipboard():
    """Generate parameter code for copying"""
    config = st.session_state.config
    
    st.subheader("📋 Copy Configuration Parameters")
    
    # Generate formatted configuration code
    parameters_code = f"""# MAIT Configuration Parameters
# Generated from Streamlit Interface on {time.strftime('%Y-%m-%d %H:%M:%S')}

# Data Configuration
data_file = '{config['data_file']}'
outcome_var = '{config['outcome_var']}'
cat_features = {config['cat_features']}
columns_to_drop = {config['columns_to_drop']}

# Model Selection
models_to_include = {config['models_to_include']}

# Output Settings
main_folder_name = '{config['main_folder_name']}'
class_labels_display = {config['class_labels_display']}

# Data Split Settings
data_split = {config['data_split']}
train_size_perc = {config['train_size_perc']}
data_split_by_patients = {config['data_split_by_patients']}
already_split = {config['already_split']}

# Feature Engineering
feat_sel = {config['feat_sel']}
num_features_sel = {config['num_features_sel']}
merged_rare_categories = {config['merged_rare_categories']}
rarity_threshold = {config['rarity_threshold']}
remove_outliers = {config['remove_outliers']}

# Training Parameters
hp_tuning = {config['hp_tuning']}
n_iter_hptuning = {config['n_iter_hptuning']}
cv_folds = {config['cv_folds']}
tun_score = '{config['tun_score']}'

# Resource Configuration
n_cpu_for_tuning = {config['n_cpu_for_tuning']}
n_cpu_model_training = {config['n_cpu_model_training']}
GPU_avail = {config['GPU_avail']}

# Advanced Options
external_val = {config['external_val']}
survival_analysis = {config['survival_analysis']}
regression_analysis = {config['regression_analysis']}
"""
    
    st.code(parameters_code, language='python')
    st.info("💡 **Tip:** Copy this entire block and paste it into any MAIT notebook to apply your configuration")
    
    # Also provide as downloadable file
    st.download_button(
        label="⬇️ Download as Python File",
        data=parameters_code,
        file_name=f"mait_config_{config['main_folder_name']}.py",
        mime="text/plain",
        help="Download the configuration as a Python file"
    )

if __name__ == "__main__":
    main()
