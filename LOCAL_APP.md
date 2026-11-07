# Run MAIT through the local app

The Streamlit app can configure and run classification, regression, and survival analyses and present the results in a dashboard. The local launcher enables execution on your computer. The public app configures analyses and opens results archives.

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

The local installation includes CatBoost and Feyn. **Run MAIT** supports CatBoost for classification. For QLattice, export a script and run it in the analysis environment. For LightGBM, use script export and install the additional package in that environment:

```bash
python -m pip install lightgbm
```

## First analysis

1. In **Data Configuration**, try the bundled example or load your CSV. Select the outcome, categorical predictors, and columns to exclude.
2. Open **Run & Results** and choose **Analysis type**.
   - **Binary classification** uses your classification settings. Choose the positive class. **Use quick CPU settings** selects CatBoost and logistic regression, three CV folds, and five permutation repeats.
   - **Regression** uses a numeric outcome and compares linear regression and random forest regression. Results include MAE, RMSE, R², and an observed-versus-predicted plot.
   - **Survival analysis** uses a follow-up time column and a binary event indicator. Choose the value that means the event occurred. The other value represents censoring. The app compares Cox proportional hazards and random survival forest and shows the concordance index and survival curves.
3. Click **Run MAIT**. Progress and the live log update automatically. You can cancel a running analysis.
4. Explore **Performance**, **Data summaries**, and **Downloads**. Classification includes SHAP figures. Regression and survival show held-out permutation importance for the original predictors.

Each run has a separate temporary folder and saves a snapshot of the data and settings. Regression and survival have their own predictor, split, and cross-validation controls in **Run & Results**. They use fixed model settings and fit preprocessing within each training fold. Use one row per independent observation. Custom splits and external validation are available in the full workflows.


## Results and reproducibility

The dashboard separates development cross-validation from held-out test performance. CatBoost's mixed-feature SHAP plot is labelled as development CV; final-model SHAP figures are labelled as test results. The dashboard reads saved outputs without refitting models.

Download the results archive before closing or restarting the app. It includes the prepared input dataset (class and event outcomes encoded as 0/1), settings, logs, fitted model, figures, tables, the original run script, and a portable `reproduce.py`. Extract the archive and run `python reproduce.py` in the same analysis environment to reproduce the workflow. The original source license is included.

You can reopen the archive in **Open saved results**. Existing MAIT Kaggle results archives are supported too. Archives are inspected without extracting their paths; model pickles and scripts are never loaded or executed by the viewer. Local temporary run files are not a durable run-history service.
