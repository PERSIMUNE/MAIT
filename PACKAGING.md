# MAIT package prototype

This first packaging step exposes the existing MAIT script through a small library and command-line interface. It builds on the original analysis code and exports explicit CPU settings for the included classification dataset.

This is an alpha prototype, not a complete migration of MAIT into a modular library. The distribution name `mait-toolbox` is provisional; it has not been reserved or published.

See [release instructions](RELEASE.md) for the new `mait demo` command and analysis extra.

## Install from this checkout

```bash
python -m pip install .
mait --version
mait export-demo ./my-mait-demo
```

Or use Python:

```python
from mait import export_demo
script = export_demo("./my-mait-demo")
```

The destination must be new. Existing files are never overwritten. Export does not require the analysis dependencies.

## Run the exported script

The full original MAIT analysis environment is still required. The package intentionally does not claim that the existing platform-specific requirements are portable. After that environment is available:

```bash
python my-mait-demo/mait_demo.py
```

Inputs and result paths are absolute, so invocation does not depend on your current directory. The script uses the original pipeline, with compatibility updates for optional LightGBM loading and scikit-learn scoring, including its process working-directory changes. It must run in a separate process rather than inside a caller's Python session.

## What comes next

1. Resolve and verify a portable analysis environment.
2. Complete the real demo run and record its outputs and runtime.
3. Extract reusable preprocessing, fitting, evaluation, and reporting functions; verify their results against the original pipeline.
4. Make survival and symbolic-model dependencies optional.
5. Connect the graphical interface to the validated library.

The bundled template is synchronized with `Tutorials/mait_template.py`; an automated check keeps the copy synchronized until the pipeline has one canonical packaged implementation.

## Local compatibility work

`requirements-analysis.txt` provides a candidate CPU environment with bounded versions. Full validation remains in progress. LightGBM is imported only when selected; selecting it on macOS still requires OpenMP. Target conversion and scorer calls have been updated for current library behaviour. No change to model-selection criteria is intended.
