# MAIT 0.1.0a2 release candidate

Install with the analysis dependencies and run the bundled CPU example:

```bash
python -m pip install "mait-toolbox[analysis]"
mait demo ./my-mait-demo
```

These registry commands apply after publication. Before publication, install from this checkout with `python -m pip install ".[analysis]"`.

The destination must be new. The command runs the original pipeline in a separate process and leaves its data, script, and results there. This is a small software demonstration, not evidence of clinical performance.

The analysis extra excludes LightGBM, whose native OpenMP requirement varies by platform. Install the `lightgbm` extra and its native runtime separately to use that model. Other unselected legacy dependencies are still included until the pipeline is modularized.

Validated scope is the bundled binary-classification example using Logistic Regression and Random Forest on Apple Silicon macOS/Python 3.11. Automated export checks run separately. Other model combinations, survival, regression, and other operating systems are not validated by this release.

## Publication checklist

- Confirm ownership and availability of the provisional PyPI name `mait-toolbox`.
- Build source and wheel distributions; check metadata.
- Upload to TestPyPI, install that wheel with analysis dependencies from PyPI, and run the demo.
- Publish the same reviewed candidate to PyPI under the maintainer's account.

Publishing requires an authenticated PyPI/TestPyPI account. No credentials belong in this repository.
