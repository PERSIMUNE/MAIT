import ast
import io
import tempfile
import subprocess
import sys
from unittest.mock import patch, Mock
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from mait.runner import AnalysisJob, configured_source, read_bundle, start_analysis, validate_analysis


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'age': range(40), 'group': ['A', 'B'] * 20,
                                  'outcome': ['no'] * 20 + ['yes'] * 20})
        self.config = {'outcome_var': 'outcome', 'cat_features': ['group'],
                       'models_to_include': ['CatBoost_mdl', 'LogisticRegression_mdl'],
                       'cv_folds': 3, 'data_split': True, 'train_size_perc': .8}

    def test_rejects_invalid_targets_and_predictor_roles(self):
        self.assertEqual(validate_analysis(self.data, self.config, 'yes'), [])
        self.assertTrue(validate_analysis(self.data, self.config, 'unknown'))
        self.assertTrue(validate_analysis(self.data, dict(self.config, cat_features=[]), 'yes'))
        self.assertTrue(validate_analysis(self.data, dict(self.config, columns_to_drop=['outcome']), 'yes'))
        self.assertTrue(validate_analysis(self.data, dict(self.config, regression_analysis=True), 'yes'))
        self.assertTrue(validate_analysis(self.data, dict(self.config, models_to_include=['CatBoost_mdl']), 'yes'))

    def test_literal_configuration_cannot_inject_python(self):
        malicious = 'target"; raise RuntimeError("injected") #'
        source = configured_source(dict(self.config, outcome_var=malicious), "a'quoted.csv", "results'quoted")
        tree = ast.parse(source)
        value = next(n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'outcome_var')
        self.assertEqual(ast.literal_eval(value), malicious)
        self.assertFalse(any(isinstance(n, ast.Raise) for n in tree.body))

    def test_unknown_outcome_fails_before_spawning_or_saving_data(self):
        with self.assertRaises(ValueError):
            start_analysis(self.data, self.config, 'unknown', '/not/a/python')

    def test_result_import_rejects_traversal_and_does_not_load_pickles(self):
        for name in ['../result.json', '/result.json', 'C:/result.json', 'a\\result.json']:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as z:
                z.writestr(name, '{}')
            with self.assertRaises(ValueError):
                read_bundle(buffer.getvalue())
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            z.writestr('selected_model.pkl', b'not-a-valid-pickle')
            z.writestr('analysis.py', 'raise RuntimeError("never run")')
            z.writestr('results/analysis_summary.json', '{"selected_model":"Test"}')
        loaded = read_bundle(buffer.getvalue())
        self.assertEqual(set(loaded), {'results/analysis_summary.json'})

    def test_runs_snapshot_data_and_settings_in_separate_directories(self):
        with patch('mait.runner.check_environment', return_value=(True, 'ready')), patch('mait.runner.subprocess.Popen', return_value=Mock()):
            first = start_analysis(self.data, self.config, 'yes')
            second = start_analysis(self.data, self.config, 'no')
        try:
            self.assertNotEqual(first.directory, second.directory)
            one = pd.read_csv(first.directory / 'input.csv')
            two = pd.read_csv(second.directory / 'input.csv')
            self.assertEqual(one.loc[0, 'outcome'], 0)
            self.assertEqual(two.loc[0, 'outcome'], 1)
            self.data.loc[0, 'age'] = 999
            self.config['cv_folds'] = 10
            self.assertNotEqual(pd.read_csv(first.directory / 'input.csv').loc[0, 'age'], 999)
            self.assertIn('"cv_folds": 3', (first.directory / 'config.json').read_text())
            self.assertIn('__file__', (first.directory / 'reproduce.py').read_text())
        finally:
            import shutil
            shutil.rmtree(first.directory)
            shutil.rmtree(second.directory)

    def test_cancel_stops_the_owned_process_and_cannot_export_success(self):
        import os, time
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=os.name == 'posix')
            job = AnalysisJob(Path(tmp), process, time.monotonic())
            job.cancel()
            self.assertIsNotNone(process.poll())
            self.assertEqual(job.poll(), 'cancelled')
            with self.assertRaises(ValueError):
                job.bundle()

    def test_large_archives_are_rejected_before_reading(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            for i in range(501):
                z.writestr(f'{i}.json', '{}')
        with self.assertRaises(ValueError):
            read_bundle(buffer.getvalue())


if __name__ == '__main__':
    unittest.main()
