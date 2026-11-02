import ast
import tempfile
import unittest
from pathlib import Path
from mait import export_demo
from mait.export import _configure


class ExportTests(unittest.TestCase):
    def test_export_preserves_analysis_and_uses_demo_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "a 'quoted' folder"
            script = export_demo(destination)
            tree = ast.parse(script.read_text())
            assignments = {node.targets[0].id: node.value for node in tree.body
                           if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
            self.assertEqual(ast.literal_eval(assignments['outcome_var']), 'heart_disease')
            self.assertFalse(ast.literal_eval(assignments['GPU_avail']))
            self.assertEqual(ast.literal_eval(assignments['n_cpu_model_training']), 1)
            self.assertEqual(ast.literal_eval(assignments['mydata'].args[0]), str(destination.resolve() / 'demo_medical_data.csv'))
            self.assertTrue((destination / 'demo_medical_data.csv').is_file())
            with self.assertRaises(FileExistsError):
                export_demo(destination)

    def test_bundled_template_matches_repository_source(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual((root / 'Tutorials/mait_template.py').read_bytes(),
                         (root / 'src/mait/data/pipeline.py.txt').read_bytes())

    def test_missing_assignment_fails_instead_of_silent_default(self):
        with self.assertRaises(ValueError):
            _configure('unrelated = 1\n', {'outcome_var': repr('heart_disease')})


if __name__ == '__main__':
    unittest.main()
