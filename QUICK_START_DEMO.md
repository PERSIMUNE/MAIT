# Quick Start Guide: Testing MAIT Streamlit App with Demo Data

This guide shows you how to quickly test the MAIT Streamlit configuration interface using the included demo dataset.

## Step 1: Start the Streamlit App

```bash
cd /path/to/MAIT
./run_streamlit.sh
```

Or manually:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Step 2: Configure with Demo Data

### Data Configuration Section

1. **Navigate to "Data Configuration"** in the sidebar
2. **Select "File Path"** option
3. **Click "Use Demo Data"** button (or manually enter: `demo_medical_data.csv`)
4. **Verify the data loads** - you should see a preview showing 50 patients with 14 columns

### Set Variables

Configure the following based on the demo data:

- **Outcome Variable**: `heart_disease`
- **Categorical Features**: Select `gender` and `smoking_status`
- **Columns to Drop**: Select `patient_id`

## Step 3: Configure Other Sections

### Model Selection
- **Choose 2-3 models** for quick testing (e.g., Random Forest, LightGBM, Logistic Regression)
- **Keep default analysis type** (binary classification)

### Feature Engineering
- **Enable Feature Selection**: Yes
- **Number of Features**: 10 (for quick testing)
- **Keep other defaults**

### Training Parameters
- **Hyperparameter Tuning**: Yes
- **Tuning Iterations**: 5 (for quick testing)
- **Cross Validation Folds**: 3 (for quick testing)
- **GPU Available**: Set based on your system

### Output Settings
- **Results Folder**: `heart_disease_demo`
- **Figure Format**: PNG
- **Class Labels**: `No Risk`, `At Risk`

## Step 4: Generate Setup Instructions

1. **Navigate to "Generate & Run"** section
2. **Review configuration summary** - should show:
   - Data File: demo_medical_data.csv
   - Outcome Variable: heart_disease
   - Selected Models: 2-3 models
   - Output Folder: heart_disease_demo

3. **Click "Get Setup Instructions"** button

## Step 5: Follow the Setup Instructions

The app will provide detailed instructions including:

### Manual Configuration Steps
1. **Copy template notebook** from `Tutorials/MAIT_Tutorial_Azithromycin_pub.ipynb`
2. **Rename it** to `MAIT_Pipeline_heart_disease_demo.ipynb`
3. **Find and replace** configuration values as shown in the side-by-side comparison
4. **Update specific settings**:
   - Data file path: `demo_medical_data.csv`
   - Outcome variable: `heart_disease`
   - Categorical features: `['gender', 'smoking_status']`
   - Columns to drop: `['patient_id']`

### Quick Copy Option
Use the "Quick Copy" expandable section to copy all parameters at once and paste them into the configuration cell of your notebook.

## Step 6: Run the MAIT Pipeline

1. **Open your configured notebook** in Jupyter or VS Code
2. **Run all cells** to execute the complete pipeline
3. **Monitor progress** and check results in the `heart_disease_demo` folder

## Expected Results

After running the pipeline, you should get:

### Generated Files
```
heart_disease_demo/
├── models/           # Trained ML models
├── figures/          # ROC curves, feature importance plots
├── reports/          # Performance metrics, model comparison
└── logs/            # Execution logs
```

### Key Insights
- **Model Performance**: Compare how different algorithms perform on cardiovascular risk prediction
- **Feature Importance**: See which factors (age, BMI, smoking, etc.) are most predictive
- **Risk Stratification**: Understand how well the models separate high-risk from low-risk patients

## Troubleshooting

### Common Issues
1. **File not found**: Ensure you're running from the MAIT root directory
2. **Missing columns**: Check that column names match exactly (case-sensitive)
3. **Model errors**: Start with fewer models if you encounter issues

### Performance Tips
- **Quick testing**: Use fewer CV folds (3) and tuning iterations (5)
- **Full analysis**: Increase to 5 CV folds and 10+ tuning iterations
- **Feature selection**: Start with 10 features, increase for better performance

## Demo Data Details

The `demo_medical_data.csv` contains realistic cardiovascular risk factors:

- **Demographics**: age, gender
- **Lifestyle**: smoking_status, exercise_frequency, alcohol_consumption, sleep_hours
- **Health Metrics**: bmi, cholesterol_level, stress_level
- **Medical History**: diabetes, hypertension, family_history
- **Outcome**: heart_disease (0 = no disease, 1 = disease)

This dataset is perfect for demonstrating MAIT's capabilities in medical AI and binary classification tasks.

## Next Steps

After successfully running the demo:

1. **Explore results** in the generated reports and visualizations
2. **Try different model combinations** to see performance differences
3. **Experiment with feature selection** settings
4. **Use your own data** by replacing the demo file path
5. **Save configurations** for reuse with similar datasets

This demo provides a complete end-to-end example of using MAIT for medical AI research!
