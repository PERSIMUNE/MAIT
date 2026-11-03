# QuickStart - minimal configuration guide

## License, third-party components, and intended use

MAIT's own source code is distributed under the MIT License; see [LICENSE](https://github.com/PERSIMUNE/MAIT/blob/main/LICENSE) for its terms, warranty disclaimer, and limitation of liability. The existing PERSIMUNE copyright notice is retained.

Third-party libraries, datasets, and other components retain their respective licenses and notices. MAIT's MIT License does not relicense those components or override their conditions. Some dependencies impose additional requirements or restrictions, including restrictions on commercial use or redistribution. See [THIRD_PARTY_NOTICES.md](https://github.com/PERSIMUNE/MAIT/blob/main/THIRD_PARTY_NOTICES.md) and the applicable component's license. This notice does not establish that every combination or use is license-compatible.

MAIT is intended for research and methodological exploration. It has not been validated for clinical decision-making, diagnosis, or treatment. Example results and successful software execution do not establish clinical safety, effectiveness, or suitability for a particular purpose.

Users should independently assess data quality, study design, model assumptions, validation, and applicable legal and ethical requirements before relying on results. No commitment to ongoing maintenance or support is made unless separately agreed in writing.

Developer attribution identifies contributions to the software; it does not imply endorsement or approval by a current or former employer or institution.


You can copy one of the tutorial notebooks found in this documentation and customize it for your project. The customization includes setting serveral parameters depending the type of the analysis.

## 🚀 Easy Configuration with Streamlit App

**NEW: Use our interactive configuration interface!**

Instead of manually setting parameters, you can use our Streamlit-based configuration interface:

- **Online version**: [https://maitconfig.streamlit.app/](https://maitconfig.streamlit.app/)
- **Local version**: Run `streamlit run streamlit_app.py` in your MAIT directory

The Streamlit app provides:
- ✅ Interactive parameter configuration with validation
- ✅ Real-time preview of your settings
- ✅ Export ready-to-run Python scripts
- ✅ Built-in help and documentation
- ✅ No manual coding required

Simply configure your pipeline through the web interface and download the generated Python script!

---

## Manual Configuration (Alternative Method)

## General Settings (Mandatory for All Tasks)
1. **Specify Main Directory**:
   - `main_folder_name = 'results_<ProjectName>'`

---

## Binary Classification (Mandatory Parameters)

1. **Outcome Variable**:
   - `outcome_var = "target"`

you must also specify the names of categorical variables (features) if available in your dataset, like `cat_features = ["race", "gender"]`.
Also if you need some variables not to be included like `columns_to_drop = ["irrelevant feature X", "irrelevant feature Y"]`, and also to make sure continuous variables are also defined like `continuous_features = ["age","weight"]`.

2. **Class Labels**:
   - Positive Class: `class_1 = "malignant"`
   - Negative Class: `class_0 = "benign"`

   also replace 0 and 1 here based on your dataset `class_label_dict = {0:class_0, 1:class_1}`, maybe in your dataset they're like -1 and 1 instead.

3. **Data Split**:
   - Enable/Disable: `data_split = True`

4. **Training Size**:
   - If `data_split = True`: 
     - `train_size_perc = 0.8`

5. **Already Split Data** (If Applicable):
   - `already_split = False`
   - If `True`, specify subsets:
     ```python
     testset = mydata[mydata['subset'] == 'Test']
     mydata = mydata[mydata['subset'] == 'Train']
     ```

6. **Classification Models**:
   - Specify at least two models:

---

## Survival Analysis (Mandatory Parameters)
1. **Enable Survival Analysis**:
   - `survival_analysis = True`

2. **Time-to-Event Column**:
   - `time_to_event_column = "max_time_difference_days"`

---

## Regression (Mandatory Parameters)
1. **Enable Regression Analysis**:
   - `regression_analysis = True`

2. **Outcome Variable**:
   - `regression_outcome = "regression_outcome_var"`

---

## Resource Configuration
1. **CPU/GPU Settings**:
   - Enable GPU (if available): `GPU_avail = True`
   - CPUs for Training: `n_cpu_model_training = 20`
   - CPUs for Hyperparameter Tuning: `n_cpu_for_tuning = 20`

---

## Additional Notes
- **Recommended**: Use the [Streamlit configuration app](https://maitconfig.streamlit.app/) for easier setup and automatic parameter validation
- Each task (classification, survival, regression) has specific mandatory parameters. Only include the settings relevant to the task you are running.
- Omitting any of the above parameters will result in errors or crashes during execution.
- For further details, refer to the full pipeline manual.
