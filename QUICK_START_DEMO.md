# Try MAIT with the included dataset

## License, third-party components, and intended use

MAIT's own source code is distributed under the MIT License; see [LICENSE](https://github.com/PERSIMUNE/MAIT/blob/main/LICENSE) for its terms, warranty disclaimer, and limitation of liability. The existing PERSIMUNE copyright notice is retained.

Third-party libraries, datasets, and other components retain their respective licenses and notices. MAIT's MIT License does not relicense those components or override their conditions. Some dependencies impose additional requirements or restrictions, including restrictions on commercial use or redistribution. See [THIRD_PARTY_NOTICES.md](https://github.com/PERSIMUNE/MAIT/blob/main/THIRD_PARTY_NOTICES.md) and the applicable component's license. This notice does not establish that every combination or use is license-compatible.

MAIT is intended for research and methodological exploration. It has not been validated for clinical decision-making, diagnosis, or treatment. Example results and successful software execution do not establish clinical safety, effectiveness, or suitability for a particular purpose.

Users should independently assess data quality, study design, model assumptions, validation, and applicable legal and ethical requirements before relying on results. No commitment to ongoing maintenance or support is made unless separately agreed in writing.

Developer attribution identifies contributions to the software; it does not imply endorsement or approval by a current or former employer or institution.


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
