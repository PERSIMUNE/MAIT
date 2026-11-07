"""Run controls and a read-only results dashboard for the Streamlit interface."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import pandas as pd
from PIL import Image
import streamlit as st
from mait.runner import check_environment, read_bundle, start_analysis, validate_analysis


def local_execution_enabled():
    return os.environ.get('MAIT_ENABLE_RUNNER') == '1'


def quick_settings(config):
    config.update(models_to_include=['CatBoost_mdl', 'LogisticRegression_mdl'],
                  hp_tuning=False, feat_sel=False, cv_folds=3, GPU_avail=False,
                  n_cpu_for_tuning=2, n_cpu_model_training=2, n_rep_feature_permutation=5,
                  data_split=True, train_size_perc=.8, data_split_by_patients=False,
                  data_split_multi_strats=False, already_split=False, external_val=False,
                  ext_val_demo=False, survival_analysis=False, regression_analysis=False,
                  demo_configs=False, shorten_feature_names=False)
    for key in list(st.session_state):
        if key.startswith('model_'):
            del st.session_state[key]


@st.cache_data(show_spinner=False, ttl=60)
def environment_status(python):
    return check_environment(python)


@st.cache_data(show_spinner=False)
def image_preview(content):
    with Image.open(io.BytesIO(content)) as image:
        if image.width * image.height > 60_000_000:
            raise ValueError('This figure is too large to preview; download it instead.')
        image = image.convert('RGB')
        image.thumbnail((1800, 1800))
        output = io.BytesIO()
        image.save(output, format='PNG')
        return output.getvalue()


@st.cache_data(show_spinner=False)
def table_preview(content, suffix):
    if suffix == '.csv':
        return pd.read_csv(io.BytesIO(content))
    return pd.read_excel(io.BytesIO(content))


@st.cache_data(show_spinner=False)
def unpack_results(content):
    return read_bundle(content)


def show_table(content, suffix):
    try:
        st.dataframe(table_preview(content, suffix), hide_index=True, width='stretch')
    except Exception:
        st.info('This table could not be previewed. Download the archive to inspect the original file.')


def show_figure(name, content):
    try:
        st.image(image_preview(content), width='stretch')
    except Exception as error:
        st.info(f'Preview unavailable: {error}')
    st.download_button('Download this figure', content, file_name=Path(name).name,
                       key='figure-download-' + name)


def results_dashboard(archive):
    try:
        artifacts = unpack_results(archive)
    except (ValueError, zipfile.BadZipFile, OSError) as error:
        st.error(f'Cannot open results: {error}')
        return
    by_name = {Path(name).name: content for name, content in artifacts.items()}
    def record(name):
        try:
            value = json.loads(by_name.get(name, b'{}'))
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}
    run = record('run.json')
    summary = record('analysis_summary.json')
    st.subheader('Analysis results')
    if run:
        details = [f"Run saved {run.get('created', '')}", f"Outcome: {run.get('outcome', '')}"]
        if run.get('positive_class') is not None:
            details.append(f"Event value: {run['positive_class']}" if run.get('analysis_type') == 'survival'
                           else f"Positive class: {run['positive_class']}")
        if run.get('duration_column'):
            details.append(f"Follow-up time: {run['duration_column']}")
        st.caption(' · '.join(details))
    interpretation = ('Feature importance' if summary.get('analysis_type') in {'regression', 'survival'}
                      else 'SHAP & interpretation')
    overview, shap, data, downloads = st.tabs(['Performance', interpretation, 'Data summaries', 'Downloads'])
    with overview:
        if summary:
            c1, c2, c3 = st.columns(3)
            c1.metric('Selected model', summary.get('selected_model', '—'))
            c2.metric('Development observations', summary.get('development_rows', '—'))
            c3.metric('Test observations', summary.get('test_rows', '—'))
            st.caption(summary.get('selection_rule', ''))
        metrics = sorted(n for n in artifacts if Path(n).name.startswith('test_performance_') and n.endswith('.csv'))
        if metrics:
            selected = st.selectbox('Held-out performance', metrics,
                                   format_func=lambda x: Path(x).stem.replace('test_performance_', ''), key='performance-table')
            show_table(artifacts[selected], '.csv')
        cv = next((n for n in artifacts if Path(n).name == 'aggregated_results_all.xlsx'), None)
        if cv:
            with st.expander('Compare cross-validation results', expanded=bool(summary.get('analysis_type'))):
                st.caption('Development-set cross-validation; these are separate from held-out test results.')
                show_table(artifacts[cv], '.xlsx')
        figures = sorted(n for n in artifacts if Path(n).name.startswith(('ROC_CM_', 'performance_')) and Path(n).suffix.lower() in {'.png', '.tif', '.tiff'})
        if figures:
            selected = st.selectbox('Performance figure', figures,
                                   format_func=lambda x: Path(x).stem.removeprefix('ROC_CM_').removeprefix('performance_'), key='performance-figure')
            show_figure(selected, artifacts[selected])
        if not metrics and not cv and not figures:
            st.info('This archive does not contain recognised performance tables or figures.')
    with shap:
        figures = [n for n in artifacts if any(x in Path(n).name.lower() for x in ['shap', 'interaction', 'fi_perm', 'fi_tree']) and Path(n).suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}]
        figures.sort(key=lambda n: (0 if '_withcats' in n else 1 if 'finalmodel' in n else 2, n))
        def description(name):
            label = Path(name).stem
            if label == 'SHAP_summary_plot_CB_withcats':
                return 'CatBoost: continuous + categorical features · development CV'
            if 'finalmodel' in label:
                return label.replace('SHAP_summary_plot_', '').replace('_', ' ') + ' · held-out test set'
            return label.replace('_', ' ')
        if figures:
            selected = st.selectbox('Interpretation figure', figures, format_func=description, key='shap-figure')
            if '_withcats' in selected:
                st.caption('Continuous values use a colour scale; categories use discrete colours and marker shapes. SHAP contributions describe the model, not causal effects.')
            show_figure(selected, artifacts[selected])
        else:
            st.info('No interpretation figures were found in this archive.')
    with data:
        tables = sorted(n for n in artifacts if Path(n).suffix.lower() in {'.csv', '.xlsx'} and not Path(n).name.startswith('test_performance_') and Path(n).name != 'input.csv')
        if tables:
            selected = st.selectbox('Data or feature summary', tables, format_func=lambda x: Path(x).name, key='data-table')
            show_table(artifacts[selected], Path(selected).suffix.lower())
        else:
            st.info('No data-summary tables were found.')
    with downloads:
        st.download_button('Download results archive', archive, 'mait-results.zip', 'application/zip', type='primary')
        st.caption('Includes the prepared dataset, reproduce.py, settings, logs, figures, tables, and fitted model. Model files are downloaded only; the dashboard does not load them.')
        if 'config.json' in by_name:
            st.download_button('Download run settings', by_name['config.json'], 'mait-config.json', 'application/json')
            with st.expander('Settings used for this run'):
                st.json(record('config.json'))
        log_name = 'run.log' if 'run.log' in by_name else 'pipeline_report.txt'
        if log_name in by_name:
            with st.expander('Run log and environment'):
                st.code(by_name[log_name].decode('utf-8', errors='replace')[-20000:], language='text')
        st.caption('Results stay associated with their saved settings even if you change the current configuration.')


@st.fragment(run_every='3s')
def monitor_job():
    job = st.session_state['analysis_job']
    if job.poll() != 'running':
        st.rerun()
    elapsed = int(time.monotonic() - job.started)
    st.info(f'MAIT is running · {elapsed // 60}m {elapsed % 60}s elapsed. You can review other configuration sections while it runs.')
    results = job.directory / 'results'
    if list(results.glob('SHAP_summary*')):
        st.caption('Model evaluation and interpretation outputs are being generated.')
    elif (results / 'summary_table_devset.xlsx').exists():
        st.caption('Data preparation has completed; models are being evaluated.')
    else:
        st.caption('Preparing data and initialising the analysis environment.')
    if st.button('Cancel analysis'):
        job.cancel()
        st.rerun()
    with st.expander('Live run log'):
        st.code(job.log_tail(), language='text')


def run_results_page():
    st.header('Run & explore')
    st.caption('Run classification, regression, or survival analysis locally, or open a saved MAIT results archive.')
    config = st.session_state.config
    data = st.session_state.get('analysis_data')
    job = st.session_state.get('analysis_job')
    running = job is not None and job.poll() == 'running'
    with st.expander('Run a configured analysis', expanded=not st.session_state.get('results_archive')):
        if data is None:
            st.info('Choose example data or upload a CSV in Data Configuration first.')
        else:
            st.write(f"**{st.session_state.get('dataset_name', 'Dataset')}** · {len(data):,} rows · {len(data.columns):,} columns")
            task = st.selectbox('Analysis type', ['classification', 'regression', 'survival'],
                                format_func=lambda value: {'classification': 'Binary classification',
                                                           'regression': 'Regression', 'survival': 'Survival analysis'}[value],
                                disabled=running)
            positive = None
            if task == 'classification':
                if st.button('Use quick CPU settings', disabled=running):
                    quick_settings(config)
                    st.rerun()
                target = config.get('outcome_var')
            else:
                config = {'analysis_type': task, 'data_split': True}
                target = st.selectbox('Event indicator' if task == 'survival' else 'Numeric outcome',
                                      list(data.columns), index=None, key=f'{task}-target')
                duration = None
                if task == 'survival':
                    duration = st.selectbox('Follow-up time', [c for c in data if c != target], index=None)
                    config['duration_column'] = duration
                candidates = [c for c in data if c not in {target, duration}]
                config['outcome_var'] = target
                excluded = st.multiselect('Exclude columns', candidates,
                                          default=[c for c in st.session_state.config['columns_to_drop'] if c in candidates],
                                          key=f'{task}-drop')
                config['columns_to_drop'] = excluded
                available = [c for c in candidates if c not in excluded]
                config['cat_features'] = st.multiselect(
                    'Categorical predictors', available,
                    default=[c for c in st.session_state.config['cat_features'] if c in available],
                    key=f'{task}-categorical')
                config['train_size_perc'] = st.slider('Development fraction', .5, .9, .8, .05, key=f'{task}-split')
                config['cv_folds'] = st.number_input('Cross-validation folds', 3, 10, 3, key=f'{task}-folds')
                st.caption('Compares Cox proportional hazards and random survival forest.' if task == 'survival'
                           else 'Compares linear regression and random forest regression.')
            if task != 'regression':
                classes = data[target].dropna().unique().tolist() if target in data.columns else []
                positive = st.selectbox('Event value' if task == 'survival' else 'Positive outcome class',
                                        classes, index=None, key='runner-positive' if task == 'classification' else 'runner-event',
                                        help='The selected value is encoded as 1. The other value becomes 0.')
            try:
                errors = validate_analysis(data, config, positive)
            except (ValueError, TypeError):
                errors = ['Some saved settings have invalid values. Review your configuration.']
            if errors:
                for error in errors:
                    st.warning(error)
            st.caption('In-app runs use up to two CPU workers and PNG figures. The exact run settings are saved with the results.')
            enabled = local_execution_enabled()
            python = os.environ.get('MAIT_ANALYSIS_PYTHON', sys.executable)
            ready, message = environment_status(python) if enabled else (False, 'Local execution is not enabled in this hosted instance.')
            if enabled:
                (st.success if ready else st.info)(message)
            else:
                st.info('To run MAIT on your computer, launch the app using the local instructions below. This hosted app can configure analyses and display saved results.')
            with st.expander('Local installation and launch'):
                st.code('python3.11 -m venv .venv-mait\nsource .venv-mait/bin/activate\npython -m pip install -r requirements-local.txt\nMAIT_ENABLE_RUNNER=1 python -m streamlit run streamlit_app.py', language='bash')
                st.caption('From the MAIT repository, use run_streamlit.sh to enable local execution. On Windows PowerShell, set $env:MAIT_ENABLE_RUNNER="1" before launching Streamlit.')
            if st.button('Run MAIT', type='primary', disabled=running or bool(errors) or not ready or not enabled):
                try:
                    st.session_state['analysis_job'] = start_analysis(data, config, positive, python)
                    st.session_state.pop('results_archive', None)
                    st.session_state.pop('harvested_job', None)
                    st.rerun()
                except (ValueError, OSError) as error:
                    st.error(str(error))
    job = st.session_state.get('analysis_job')
    if job:
        status = job.poll()
        if status == 'running':
            monitor_job()
        elif status == 'succeeded':
            if st.session_state.get('harvested_job') != str(job.directory):
                st.session_state['results_archive'] = job.bundle()
                st.session_state['harvested_job'] = str(job.directory)
            st.success('Analysis complete. Explore your results below.')
        elif status == 'cancelled':
            st.info('Analysis cancelled. You can update the settings and start a new run.')
        else:
            st.error('The analysis did not finish. Review the log, adjust the settings, and try again.')
            with st.expander('Failure details', expanded=True):
                st.code(job.log_tail(), language='text')
    with st.expander('Open saved results', expanded=not job):
        archive = st.file_uploader('MAIT results archive (.zip)', type=['zip'], key='results-upload')
        st.caption('Archives are processed by the computer hosting this app. Existing MAIT Kaggle results archives are also supported.')
        if archive:
            content = archive.getvalue()
            digest = hashlib.sha256(content).hexdigest()
            if st.session_state.get('imported_results_digest') != digest:
                try:
                    unpack_results(content)
                    st.session_state['results_archive'] = content
                    st.session_state['imported_results_digest'] = digest
                except (ValueError, zipfile.BadZipFile, OSError) as error:
                    st.error(str(error))
    if st.session_state.get('results_archive'):
        results_dashboard(st.session_state['results_archive'])
