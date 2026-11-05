<span id="quickstart-minimal-configuration-guide"></span>

# Quick start

<div class="mait-guide-lead">
<p class="mait-eyebrow">YOUR FIRST ANALYSIS</p>
<p>Prepare your data, configure an analysis, and export a Python script. Start with the interactive app or adapt a worked tutorial.</p>
<a class="mait-primary-link" href="https://maitconfig.streamlit.app/">Open the configuration app <span aria-hidden="true">↗</span></a>
<a class="mait-secondary-link" href="Executed_tutorials.html">Browse tutorials →</a>
</div>

<div class="mait-guide-route" aria-label="Analysis workflow">
<div><span>01</span><strong>Prepare</strong><p>Choose your dataset and outcome.</p></div>
<div><span>02</span><strong>Configure</strong><p>Select models and analysis settings.</p></div>
<div><span>03</span><strong>Export &amp; run</strong><p>Run the script in your analysis environment.</p></div>
</div>

<span id="easy-configuration-with-streamlit-app"></span>

## Configure with the app

The [MAIT configuration app](https://maitconfig.streamlit.app/) guides you through data settings, model selection, feature engineering, training, and outputs. It prepares a Python script; run that script in your local MAIT analysis environment.

1. **Set up your data.** Upload a CSV to preview its columns, or enter a file path. Choose the outcome and categorical features, then exclude any columns you do not want to use.
2. **Choose your settings.** Work through the sections in the sidebar. Select models, review the data split, and set the training and output options.
3. **Export your analysis.** Open **Finalize and Export**, review the summary, and choose **Export Python script**, then **Download Python Script**. Save it in your MAIT directory and follow the setup instructions to run it.

For an example, choose **File Path → Use Demo Data**. Set the outcome to `heart_disease`, select `gender` and `smoking_status` as categorical features, and exclude `patient_id`.

<aside class="mait-guide-note"><strong>Using your own data?</strong> The data path in the exported script must exist on the computer where you run it. An upload previews the data in the app; keep a local copy for the analysis.</aside>

### Run the app locally

From a copy of the [MAIT repository](https://github.com/PERSIMUNE/MAIT), install the interface requirements in a Python environment, then launch it:

```bash
python -m pip install -r streamlit_requirements.txt
python -m streamlit run streamlit_app.py
```

These requirements install the configuration interface. Use the [analysis setup instructions](https://github.com/PERSIMUNE/MAIT#how-to-use) for running the exported pipeline.

<span id="manual-configuration-alternative-method"></span>

## Configure a tutorial manually

Prefer to work in a notebook? Copy a [worked tutorial](Executed_tutorials.html) and edit its configuration block. The examples below highlight settings to review; retain the tutorial’s other settings and imports.

<span id="general-settings-mandatory-for-all-tasks"></span>

### Data and output

Use column names exactly as they appear in your dataset. Separate categorical and continuous predictors, and exclude identifiers or variables that should not enter the model.

```python
main_folder_name = "results_my_project"
outcome_var = "target"
cat_features = ["race", "gender"]
continuous_features = ["age", "weight"]
columns_to_drop = ["patient_id"]
```

<span id="binary-classification-mandatory-parameters"></span>

### Binary classification

Define the negative and positive class labels and map them to the values used in your outcome column. In this example, the data are encoded as `0` and `1`; adapt the keys if your encoding differs.

```python
class_0 = "benign"
class_1 = "malignant"
class_label_dict = {0: class_0, 1: class_1}
```

Select the classification models in the tutorial’s model configuration. Start with a small selection to check the workflow before expanding your comparison.

### Training and test data

For an 80% training / 20% test split:

```python
data_split = True
train_size_perc = 0.8
already_split = False
```

For a predefined split, enable `already_split` and follow the [manual’s data-splitting instructions](MANUAL.html). For repeated observations, review patient-level splitting before training.

<span id="survival-analysis-mandatory-parameters"></span>

<span id="regression-mandatory-parameters"></span>

### Survival analysis and regression

Use a tutorial appropriate to your task and review these additional settings:

| Task | Settings to review | What they specify |
| --- | --- | --- |
| Survival analysis | `survival_analysis = True` | Enable the survival workflow. |
| | `time_to_event_column = "max_time_difference_days"` | Name of the time-to-event column; also check the tutorial’s event/outcome coding. |
| Regression | `regression_analysis = True` | Enable the regression workflow. |
| | `regression_outcome = "regression_outcome_var"` | Name of the continuous outcome column. |

<span id="resource-configuration"></span>

### Computing resources

Set CPU counts to suit your computer. Enable GPU use only when your hardware and the selected models support it.

```python
GPU_avail = False
n_cpu_model_training = 1
n_cpu_for_tuning = 1
```

<span id="additional-notes"></span>

## Continue your analysis

<div class="mait-guide-next">
<a href="Executed_tutorials.html"><strong>Explore worked examples →</strong><span>See the workflow and its outputs.</span></a>
<a href="MANUAL.html"><strong>Read the methods and settings →</strong><span>Understand configuration and model evaluation.</span></a>
</div>
