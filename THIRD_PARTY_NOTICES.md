# Third-party components and licensing notices

Last checked: 3 November 2026. MAIT's MIT License applies to MAIT code, not to independently licensed dependencies or datasets. Dependencies installed by pip are separate distributions; their license files, attribution requirements, and applicable usage/redistribution conditions remain in effect. This inventory is not a finding of legal compatibility and is not a substitute for applicable license texts. Dataset terms must be checked separately for each dataset.

## Items requiring attention

- **Feyn 3.5.0:** both its metadata and bundled license identify CC-BY-NC-ND-4.0. Its bundled license credits Abzu ApS. Do not assume MAIT's MIT permission authorizes commercial use, modification, or redistribution of Feyn. Feyn is currently installed by the analysis extra even when QLattice is not selected. Separating this dependency is outstanding work; this notice does not fix that limitation. See https://pypi.org/project/feyn/ and https://creativecommons.org/licenses/by-nc-nd/4.0/ .
- **mrmr-selection 0.2.8:** installed metadata identifies GPLv3, while the LICENSE included in that exact installed distribution is MIT, copyright 2023 smazzanti. The current upstream repository also has an MIT license. This inconsistency is recorded rather than represented as definitively resolved. See https://github.com/smazzanti/mrmr/blob/main/LICENSE .

## Direct analysis dependencies

Versions below describe the local validation environment, not a lock on every user's installation. Metadata labels may be missing, ambiguous, or incomplete. Inspect the installed distribution's license files and upstream terms before redistribution.

| Component | Checked version | Metadata license label | Source |
| --- | --- | --- | --- |
| numpy | 1.26.4 | Copyright (c) 2005-2023, NumPy Developers. | https://pypi.org/project/numpy/ |
| pandas | 2.3.3 | BSD 3-Clause License | https://pypi.org/project/pandas/ |
| scikit-learn | 1.6.1 | BSD 3-Clause License | https://pypi.org/project/scikit-learn/ |
| scipy | 1.17.1 | Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. | https://pypi.org/project/scipy/ |
| shap | 0.47.2 | MIT License | https://pypi.org/project/shap/ |
| scikit-survival | 0.24.1 | GNU GENERAL PUBLIC LICENSE | https://pypi.org/project/scikit-survival/ |
| matplotlib | 3.11.2 | License agreement for matplotlib versions 1.3.0 and later | https://pypi.org/project/matplotlib/ |
| seaborn | 0.13.2 | BSD License | https://pypi.org/project/seaborn/ |
| joblib | 1.6.0 | BSD-3-Clause | https://pypi.org/project/joblib/ |
| catboost | 1.2.10 | Apache License, Version 2.0 | https://pypi.org/project/catboost/ |
| feyn | 3.5.0 | CC-BY-NC-ND-4.0 | https://pypi.org/project/feyn/ |
| mrmr-selection | 0.2.8 | GNU General Public License v3.0 | https://pypi.org/project/mrmr-selection/ |
| survshap | 0.4.2 | MIT License | https://pypi.org/project/survshap/ |
| openpyxl | 3.1.5 | MIT | https://pypi.org/project/openpyxl/ |
| sympy | 1.14.0 | BSD | https://pypi.org/project/sympy/ |
| imbalanced-learn | 0.14.2 | MIT License | https://pypi.org/project/imbalanced-learn/ |
| ipython | 9.18.1 | BSD-3-Clause | https://pypi.org/project/ipython/ |
| psutil | 7.2.2 | BSD-3-Clause | https://pypi.org/project/psutil/ |

LightGBM is an optional separate extra and is not part of this checked analysis environment. Consult https://github.com/microsoft/LightGBM/blob/master/LICENSE and the exact installed distribution. Transitive dependencies have their own terms and are not exhaustively audited here.
