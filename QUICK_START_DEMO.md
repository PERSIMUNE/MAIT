# Try MAIT with the included dataset

This walkthrough shows how to configure and export a binary classification analysis. The interface prepares a Python script; it does not execute the analysis in the browser.

## 1. Open the configuration interface

Use https://maitconfig.streamlit.app/ or launch it locally in a separate Python environment:

```bash
python -m venv .venv-config
# macOS / Linux
source .venv-config/bin/activate
# Windows PowerShell: .venv-config\Scripts\Activate.ps1
python -m pip install -r streamlit_requirements.txt
python -m streamlit run streamlit_app.py
```

Run these commands from the MAIT repository directory. These packages install the configuration interface only.

## 2. Select the example data

In **Data Configuration**, choose **File Path** and **Use Demo Data**.

- Outcome: `heart_disease`
- Categorical features: `gender`, `smoking_status`
- Drop: `patient_id`

Check the preview before continuing. The included file has 50 rows; it is an interface demonstration, not evidence of clinical predictive performance.

## 3. Start with a small configuration

Select Logistic Regression and Random Forest. Set GPU availability to false and use one CPU for training and tuning. For a first attempt, disable hyperparameter tuning and feature selection and use three cross-validation folds. Keep survival analysis, regression analysis, and external validation disabled. Set the output folder to `heart_disease_demo`.

These settings reduce the initial workload. They are not recommendations for a final research analysis.

## 4. Export your script

Go to **Generate & Run**, review the summary, and click **Save Executable Python File**, followed by **Download Python Script**.

Save the script in your local MAIT directory alongside `demo_medical_data.csv`. A path selected in the online interface must also exist on the computer where you run the script.

## 5. Run in the analysis environment

Use the analysis environment described in [README](README.md#how-to-use), then run:

```bash
python mait_pipeline_heart_disease_demo.py
```

The interface environment above is insufficient for this step. The existing analysis installation instructions require a portability update: `requirements.txt` contains build-machine file paths and platform-specific packages. A clean installation and complete demo run have not yet been verified as part of this documentation update.

## 6. Review the results

Inspect the `heart_disease_demo` folder. Use the manual to interpret model comparisons, discrimination measures, and feature explanations. Check actual output filenames rather than expecting a fixed `models/figures/reports/logs` structure.

For this small demonstration, focus on understanding the workflow and outputs. A successful run does not establish clinical usefulness.

## If something goes wrong

Record your operating system, Python version, MAIT revision, selected settings, and the full error message. Open a GitHub issue without including confidential data. For a missing input file, check that you are running from the MAIT directory and the configured file exists there.
