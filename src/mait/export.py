"""Prepare the original MAIT script with an explicit CPU demo configuration."""
import ast
from importlib.resources import files
from pathlib import Path

DEMO_SETTINGS = {
    "demo_configs": False,
    "cat_features": ["gender", "smoking_status"],
    "columns_to_drop": ["patient_id"],
    "outcome_var": "heart_disease",
    "models_to_include": ["LogisticRegression_mdl", "RandomForest_mdl"],
    "class_labels_display": ["No disease", "Disease"],
    "GPU_avail": False,
    "n_cpu_for_tuning": 1,
    "n_cpu_model_training": 1,
    "hp_tuning": False,
    "feat_sel": False,
    "cv_folds": 3,
    "survival_analysis": False,
    "regression_analysis": False,
}


def _configure(source, settings):
    """Replace only top-level assignments, failing if a setting is missing."""
    tree = ast.parse(source)
    replacements = []
    for name, value in settings.items():
        matches = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == name]
        if not matches:
            raise ValueError(f"Missing top-level assignment for {name}")
        # Later assignments are pipeline transformations, not configuration.
        node = matches[0]
        replacements.append((node.lineno - 1, node.end_lineno, f"{name} = {value}\n"))
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines)
    compile(result, "mait_demo.py", "exec")
    return result


def export_demo(directory):
    """Create a new demo directory; never overwrite an existing directory.

    Exports the original analysis script, data, and settings. Does not run analysis
    or install its dependencies. Returns the generated script path.
    """
    destination = Path(directory).expanduser().resolve()
    settings = {key: repr(value) for key, value in DEMO_SETTINGS.items()}
    settings["mydata"] = f"pd.read_csv({str(destination / 'demo_medical_data.csv')!r})"
    settings["main_folder_name"] = repr(str(destination / "results"))
    resources = files("mait").joinpath("data")
    source = _configure(resources.joinpath("pipeline.py.txt").read_text(), settings)
    destination.mkdir(parents=True, exist_ok=False)
    script = destination / "mait_demo.py"
    script.write_text(source, encoding="utf-8")
    (destination / "demo_medical_data.csv").write_bytes(
        resources.joinpath("demo_medical_data.csv").read_bytes())
    (destination / "README.txt").write_text(
        "MAIT demo exported from the original pipeline.\n"
        "Install the full MAIT analysis environment before running mait_demo.py.\n"
        "The package alone does not install analysis dependencies.\n"
        "This small example demonstrates software operation, not clinical validity.\n"
        "Citation: https://doi.org/10.48550/arXiv.2501.04547\n", encoding="utf-8")
    return script
