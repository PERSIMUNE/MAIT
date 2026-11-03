# MAIT — Medical Artificial Intelligence Toolbox

MAIT provides explainable machine-learning workflows for tabular research data. This alpha package offers a CPU classification demonstration using Logistic Regression and Random Forest, with evaluation tables, a saved model, and explanation plots.

## Install and run

```bash
python -m pip install "mait-pipeline[analysis]"
mait demo ./my-mait-demo
```

These commands apply after the PyPI release is published. For the trial release, install the MAIT wheel from TestPyPI separately from analysis dependencies sourced from PyPI. For a repository checkout, use `python -m pip install ".[analysis]"`.

Choose a new output directory. The demo command runs the pipeline in a separate process and saves its input data, configured script, and results there. It does not modify your notebook's working directory.

To export the script without running it:

```bash
mait export-demo ./my-export
```

Or in Python:

```python
from mait import export_demo
script = export_demo("./my-export")
```

## Scope and validation

The installed demo completed on Apple Silicon macOS with Python 3.11, including a fresh analysis-environment installation and installation of the uploaded TestPyPI wheel. Other platforms, model combinations, survival analysis, regression, and equivalence to historical versions remain unverified. This small demonstration does not establish clinical predictive performance.

The analysis extra still includes legacy dependencies that will be separated as MAIT becomes a modular library. LightGBM is excluded; its separate `lightgbm` extra also needs a native OpenMP runtime on macOS.

The packaged template is synchronized with the repository tutorial template and checked automatically. See the [source repository](https://github.com/PERSIMUNE/MAIT) for methods, tutorials, limitations, and release instructions.

## Citation

Ramtin Zargari Marandi, Anne Svane Frahm, Jens Lundgren, Daniel Dawson Murray, and Maja Milojevic. Medical artificial intelligence toolbox (MAIT): an explainable machine learning framework for binary classification, survival modelling, and regression analyses. 2025. https://doi.org/10.48550/arXiv.2501.04547
