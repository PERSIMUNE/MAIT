# How to use medical artificial intelligence toolbox (MAIT)

## License, third-party components, and intended use

MAIT's own source code is distributed under the MIT License; see [LICENSE](https://github.com/PERSIMUNE/MAIT/blob/main/LICENSE) for its terms, warranty disclaimer, and limitation of liability.

Third-party libraries, datasets, and other components retain their respective licenses and notices. MAIT's MIT License does not relicense those components or override their conditions. Some dependencies impose additional requirements or restrictions, including restrictions on commercial use or redistribution. See [THIRD_PARTY_NOTICES.md](https://github.com/PERSIMUNE/MAIT/blob/main/THIRD_PARTY_NOTICES.md) and the applicable component's license. This notice does not establish that every combination or use is license-compatible.

MAIT is intended for research and methodological exploration. It has not been validated for clinical decision-making, diagnosis, or treatment. Example results and successful software execution do not establish clinical safety, effectiveness, or suitability for a particular purpose.

Users should independently assess data quality, study design, model assumptions, validation, and applicable legal and ethical requirements before relying on results. No commitment to ongoing maintenance or support is made unless separately agreed in writing.




MAIT (medical artificial intelligence toolbox) is designed to streamline studies involving machine learning, with a particular focus on binary classification and explainable AI, while also extending its utility to precision medicine. It also allows the development of survival and regression models, alongside facilitating comparisons between machine learning and statistical models.

MAIT performs model evaluation and interpretation and presents the results in formats suitable for publication, including figures and tables. Moreover, MAIT offers essential data processing functionalities such as quality checks, imputation, scaling, exploration through association plots, and data splitting.

The data splitting feature is particularly advantageous for projects aiming to create prognostic or diagnostic tools, ensuring a portion of the dataset is reserved for external validation. Without data splitting, MAIT remains valuable for exploring associations, such as predictive feature discovery.

MAIT has many unique functionalities compared to existing pipelines and framwework that are explained in details in its documentation files (e.g., MANUAL) on GitHub and its upcoming paper.

For further details and practical examples, visit the MAIT repository on GitHub: https://github.com/PERSIMUNE/MAIT.

The development of MAIT is a result of extensive research and programming to provide a reliable open-access software for research. If you choose to utilize MAIT in your research, we kindly request that you cite it, enabling other researchers to discover and benefit from its capabilities.

Citation information is available on https://github.com/PERSIMUNE/MAIT

MAIT is developed by Ramtin Zargari Marandi, PhD (Postdoc researcher), at PERSIMUNE - CHIP Center of excellence for health, immunity and infections - Rigshospitalet, Copenhagen University Hospital, to conduct several machine learning studies. Affiliated researchers are Anne Svane Frahm, PhD, Maja Milojevic, PhD, and Daniel Murray, PhD.

The pipeline is based on well-stablished Python packages and frameworks such as scikit-learn. All the packages and their versions that are used in the pipeline are reported as part of the pipeline documentation. 

To utilize MAIT effectively, it is recommended to familiarize yourself with concepts and techniques in machine learning and statistical analyses. This foundational knowledge will enable you to understand MAIT's functionalities and leverage them optimally for your research objectives.
