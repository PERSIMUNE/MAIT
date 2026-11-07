"""Isolated local execution and portable, read-only result bundles for MAIT."""
import ast
from dataclasses import dataclass
from importlib.resources import files
import io
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

from .export import _configure

SUPPORTED_MODELS = {'LogisticRegression_mdl', 'RandomForest_mdl', 'CatBoost_mdl'}
MAX_BUNDLE_BYTES = 300 * 1024 * 1024


def validate_analysis(data, config, positive_class):
    errors = []
    target = config.get('outcome_var')
    if data is None or data.empty:
        return ['Load a dataset first.']
    if not data.columns.is_unique:
        return ['Column names must be unique.']
    if not all(isinstance(c, str) for c in data.columns):
        errors.append('Column names must be text.')
    if target not in data.columns:
        return errors + ['Select an outcome column in Data Configuration.']
    outcome = data[target]
    if outcome.isna().any():
        errors.append('Remove rows with missing outcomes before running.')
    classes = outcome.dropna().unique().tolist()
    if len(classes) != 2:
        errors.append('In-app execution currently requires a binary outcome with exactly two classes.')
    if positive_class not in classes:
        errors.append('Select the positive outcome class.')
    folds = int(config.get('cv_folds', 3))
    if not 3 <= folds <= 10:
        errors.append('Use between 3 and 10 cross-validation folds.')
    if len(classes) == 2 and outcome.value_counts().min() < max(10, 2 * folds):
        errors.append('Each outcome class needs at least max(10, twice the CV folds) observations for this workflow.')
    models = config.get('models_to_include', [])
    if len(set(models)) < 2:
        errors.append('Choose at least two models for MAIT’s model-comparison step.')
    if set(models) - SUPPORTED_MODELS:
        errors.append('Local execution currently supports Logistic Regression, Random Forest, and CatBoost. Export other model configurations.')
    for key in ['survival_analysis', 'regression_analysis', 'external_val', 'ext_val_demo',
                'already_split', 'data_split_by_patients', 'data_split_multi_strats', 'demo_configs']:
        if config.get(key):
            errors.append(f'Turn off {key.replace("_", " ")} for in-app binary classification; use export for this option.')
    if not config.get('data_split', True):
        errors.append('Enable a training/test split for the results dashboard.')
    fraction = float(config.get('train_size_perc', .8))
    if not .5 <= fraction <= .9:
        errors.append('Use a training fraction between 50% and 90% for in-app execution.')
    if config.get('GPU_avail'):
        errors.append('Turn off GPU support for the local CPU workflow.')
    dropped = config.get('columns_to_drop', [])
    categorical = config.get('cat_features', [])
    if (set(dropped) | set(categorical)) - set(data.columns):
        errors.append('Some selected columns are absent from this dataset; review Data Configuration.')
    if target in dropped or target in categorical or set(dropped) & set(categorical):
        errors.append('The outcome, excluded columns, and categorical predictors must not overlap.')
    predictors = [c for c in data.columns if c != target and c not in dropped]
    if not predictors:
        errors.append('Keep at least one predictor column.')
    import pandas as pd
    for column in predictors:
        if column not in categorical:
            try:
                values = pd.to_numeric(data[column].dropna(), errors='raise')
                import numpy as np
                if not np.isfinite(values).all():
                    errors.append(f'Replace infinite values in “{column}” before running.')
            except (ValueError, TypeError):
                errors.append(f'Mark “{column}” as categorical or convert it to numbers.')
    return errors


def configured_source(config, data_path, output_path):
    """Use literal values, replacing only initial configuration assignments."""
    source = files('mait').joinpath('data/pipeline.py.txt').read_text()
    names = {n.targets[0].id for n in ast.parse(source).body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    values = {k: repr(v) for k, v in config.items() if k in names}
    values.update(mydata=f'pd.read_csv({str(data_path)!r})', main_folder_name=repr(str(output_path)))
    if 'tun_score' in config:
        values['custom_scorer'] = repr(config['tun_score'])
    source = _configure(source, values)
    marker = '    # Plot the SHAP decision plot with only significant features'
    source = source.replace(marker, '    shap_values = np.asarray(shap_values, dtype=float)\n' + marker)
    return source


# These exports read variables produced by the original analysis; they do not refit models.
RESULT_EXPORT = '''
import json as _json
from pathlib import Path as _Path
for _model, _variable in {'LR':'results_df_LR', 'RF':'results_df_rf', 'CB':'results_df_CB'}.items():
    _table = globals().get(_variable)
    if isinstance(_table, pd.DataFrame):
        _table.to_csv(f'test_performance_{_model}.csv', index=False)
_summary = {'selected_model': type(selected_model).__name__,
            'development_rows': len(X_train), 'test_rows': len(y_test),
            'predictor_count': len(X_test.columns),
            'selection_rule': 'MAIT cross-validation comparison of MCC, ROC AUC, and PR AUC'}
_Path('analysis_summary.json').write_text(_json.dumps(_summary, indent=2))
'''


def check_environment(python_executable=None):
    python = str(python_executable or sys.executable)
    code = "import importlib.util,json,sys; names=['numpy','pandas','sklearn','scipy','shap','sksurv','matplotlib','seaborn','catboost','feyn','mrmr','survshap','openpyxl','imblearn','IPython','psutil','sympy','pip']; print(json.dumps({'version':list(sys.version_info[:2]),'missing':[x for x in names if importlib.util.find_spec(x) is None]}))"
    try:
        p = subprocess.run([python, '-c', code], capture_output=True, text=True, timeout=20)
        if p.returncode:
            return False, 'The analysis Python environment could not be started.'
        info = json.loads(p.stdout.strip().splitlines()[-1])
        if info['version'] != [3, 11]:
            return False, 'Use Python 3.11 for the validated local workflow.'
        if info['missing']:
            return False, 'Missing analysis packages: ' + ', '.join(info['missing'])
        return True, 'Analysis environment ready (Python 3.11).'
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False, 'Set MAIT_ANALYSIS_PYTHON to a working Python 3.11 analysis environment.'


@dataclass
class AnalysisJob:
    directory: Path
    process: subprocess.Popen
    started: float
    status: str = 'running'

    def poll(self):
        if self.status != 'running':
            return self.status
        code = self.process.poll()
        if code is None:
            return 'running'
        required = self.directory / 'results/analysis_summary.json'
        self.status = 'succeeded' if code == 0 and required.exists() else 'failed'
        return self.status

    def cancel(self):
        if self.poll() == 'running':
            if os.name == 'posix':
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    self.poll()
                    return
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(self.process.pid, signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait()
            self.status = 'cancelled'

    def log_tail(self):
        path = self.directory / 'run.log'
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 20000))
            return f.read().decode('utf-8', errors='replace')

    def bundle(self):
        if self.poll() != 'succeeded':
            raise ValueError('Only completed analyses can be exported as results.')
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in ['analysis.py', 'reproduce.py', 'input.csv', 'config.json', 'run.log', 'run.json', 'LICENSE.txt']:
                z.write(self.directory / name, name)
            for path in (self.directory / 'results').rglob('*'):
                if path.is_file() and not path.is_symlink():
                    z.write(path, path.relative_to(self.directory))
        return buffer.getvalue()


def start_analysis(data, config, positive_class, python_executable=None):
    errors = validate_analysis(data, config, positive_class)
    if errors:
        raise ValueError('\n'.join(errors))
    ready, message = check_environment(python_executable)
    if not ready:
        raise ValueError(message)
    root = Path(tempfile.mkdtemp(prefix='mait-analysis-'))
    target = config['outcome_var']
    labels = data[target].unique().tolist()
    negative = next(x for x in labels if x != positive_class)
    prepared = data.copy()
    prepared[target] = (prepared[target] == positive_class).astype(int)
    prepared.to_csv(root / 'input.csv', index=False)
    settings = dict(config)
    settings.update(class_labels_display=[str(negative), str(positive_class)],
                    n_cpu_model_training=min(2, max(1, int(config.get('n_cpu_model_training', 1)))),
                    n_cpu_for_tuning=min(2, max(1, int(config.get('n_cpu_for_tuning', 1)))),
                    fig_file_format='png', demo_configs=False)
    settings.setdefault('n_rep_feature_permutation', 5)
    (root / 'config.json').write_text(json.dumps(settings, indent=2, default=str))
    record = {'rows': len(data), 'columns': len(data.columns), 'outcome': target,
              'positive_class': str(positive_class), 'negative_class': str(negative),
              'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'models': settings['models_to_include']}
    (root / 'run.json').write_text(json.dumps(record, indent=2))
    source = configured_source(settings, root / 'input.csv', root / 'results')
    (root / 'analysis.py').write_text(source + '\n' + RESULT_EXPORT)
    portable = _configure(source, {
        'mydata': 'pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.csv"))',
        'main_folder_name': 'os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")',
    })
    (root / 'reproduce.py').write_text(portable + '\n' + RESULT_EXPORT)
    (root / 'LICENSE.txt').write_text(files('mait').joinpath('data/SOURCE_LICENSE.txt').read_text())
    environment = os.environ.copy()
    environment.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(root / 'matplotlib-cache'),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (root / 'run.log').open('w') as log:
        process = subprocess.Popen([str(python_executable or sys.executable), '-u', str(root / 'analysis.py')],
                                   cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name == 'posix'))
    return AnalysisJob(root, process, time.monotonic())


def read_bundle(content):
    """Inspect portable results without extracting files or loading model pickles."""
    result = {}
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        entries = z.infolist()
        if len(entries) > 500 or sum(x.file_size for x in entries) > MAX_BUNDLE_BYTES:
            raise ValueError('Archive is too large (maximum 500 files and 300 MB expanded).')
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in entry.filename or ':' in entry.filename:
                raise ValueError('Archive contains an unsafe file path.')
            if entry.flag_bits & 1:
                raise ValueError('Encrypted result archives are not supported.')
            if entry.is_dir():
                continue
            if entry.file_size > 50 * 1024 * 1024:
                raise ValueError('An individual result exceeds 50 MB.')
            # Models and scripts are retained for download only; never deserialized or executed.
            if path.suffix.lower() in {'.csv', '.xlsx', '.json', '.txt', '.log', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
                result[entry.filename] = z.read(entry)
    if not result:
        raise ValueError('No supported MAIT result files were found.')
    return result
