# Run MAIT through the local app

The Streamlit app can configure and run binary-classification analyses and present the results in a dashboard. Analyses run on the computer hosting the app. The public configuration app can also open results archives; execution is enabled by the local launcher.

## Install and launch

From a clone of this repository, use Python 3.11:

```bash
python3.11 -m venv .venv-mait
source .venv-mait/bin/activate
python -m pip install -r requirements-local.txt
bash run_streamlit.sh
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv-mait
.venv-mait\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
$env:MAIT_ENABLE_RUNNER="1"
python -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

If Streamlit and the analysis dependencies are installed in different environments, set `MAIT_ANALYSIS_PYTHON` to the full path of the Python 3.11 analysis interpreter before launching. The app checks the interpreter and required packages before enabling **Run MAIT**.

## First analysis

1. In **Data Configuration**, try the bundled example or load your CSV. Select the outcome, categorical predictors, and columns to exclude.
2. Review model selection, preprocessing, and training settings. In **Run & Results**, **Use quick CPU settings** selects CatBoost and logistic regression, three CV folds, no tuning or feature selection, and five permutation repeats.
3. Select the **positive outcome class**. The app records this choice and encodes it as 1; the other class becomes 0.
4. Click **Run MAIT**. Progress and the live log update automatically. You can cancel a running analysis.
5. Explore **Performance**, **SHAP & interpretation**, **Data summaries**, and **Downloads**.

Each run has a separate temporary folder, dataset snapshot, and settings snapshot. Changing the configuration does not relabel a completed run. The output folder setting remains relevant to exported scripts; in-app runs always use their own folder and PNG figures, with up to two CPU workers.

The runner currently supports Logistic Regression, Random Forest, and CatBoost, with at least two models, a binary outcome, and a standard training/test split. Survival, regression, external validation, custom/grouped splits, other model families, and GPU settings remain available through script export and the full notebooks. The runner reports unsupported selections before starting.

## Results and reproducibility

The dashboard separates development cross-validation from held-out test performance. CatBoost's mixed-feature SHAP plot is labelled as development CV; final-model SHAP figures are labelled as test results. The dashboard reads saved outputs without refitting models.

Download the results archive before closing or restarting the app. It includes the prepared input dataset (outcome encoded as 0/1), settings, logs, fitted model, figures, tables, the original run script, and a portable `reproduce.py`. Extract the archive and run `python reproduce.py` in the same analysis environment to reproduce the workflow. The original source license is included.

You can reopen the archive in **Open saved results**. Existing MAIT Kaggle results archives are supported too. Archives are inspected without extracting their paths; model pickles and scripts are never loaded or executed by the viewer. Local temporary run files are not a durable run-history service.
