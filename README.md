![Alt Text](https://github.com/PERSIMUNE/MAIT/blob/main/MAITlogo.gif)

# MAIT - medical artificial intelligence toolbox

## Introduction
Welcome to the MAIT repository! This pipeline, implemented in Python, is designed to streamline your machine learning workflows using Jupyter Notebooks. It is compatible with both Windows and Linux operating systems. This repository also includes several tutorial notebooks to help you get started quickly. You can also refer to the `MANUAL` of MAIT for documentation. To overview MAIT's unique features and capabilities, we highly recommend reading the [MAIT preprint](https://arxiv.org/abs/2501.04547) on arXiv. If you use MAIT in your research, please remember to cite it in your publications.

## How to Use

### Step 1: Clone the Repository
First, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/PERSIMUNE/MAIT.git
cd MAIT
```

### Step 2: Install Conda and Setup the Environment

There are different ways to create an environment to use MAIT.
Ensure that you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed. 

Create a new conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate mait_py3_10_9
```
Or if you prefer to use `pip` to install the required packages, you can create a conda environment and install the dependencies using `pip`:

```bash
conda create --name mait_py3_10_9 python=3.10.9
conda activate mait_py3_10_9
pip install -r requirements.txt
```
Also you can try an updated Python version, and also use mlflow (experimental).

```bash
conda create -n mait_env python=3.12 \
  numpy pandas scikit-learn scipy matplotlib seaborn joblib lightgbm catboost ipykernel \
  imbalanced-learn mlflow shap scikit-survival -c conda-forge
conda activate mait_env
pip install feyn mrmr_selection survshap openpyxl
conda install protobuf=3.20.* -c conda-forge
pip install --upgrade mlflow
```
### Step 3: Using Docker
A Docker image is available for this pipeline. You can build and run the Docker container using the `Dockerfile` provided in the repository. Here are the steps:

1. Build the Docker image (or use the one that is already available: https://hub.docker.com/r/danishdyna/mait_30092024gpu):

    ```bash
    docker build -t mait_py3_10_9:latest .
    ```

2. Run the Docker container where your MAIT pipeline files are located:

    ```bash
    docker run -it --rm -p 8888:8888 mait_py3_10_9:latest
    ```
    or do this
    ```bash
    docker run  --gpus all -p 8888:8888 -v "$(pwd):/app" -it danishdyna/mait_30092024gpu /bin/bash
    ```

Inside the container, you should activate the conda environment using `activate the conda mait_py3_10_9_30092024gpu` then run this to initiate Jupyter Notebook: `jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root`.

You can also refer to the `environment.yml` file to understand the dependencies and create your Docker environment. If you run the Docker container on an HPC server or a remote computer you can use http://your_HPC_server_address:8888/ to get access to the Jupyter Notebook to run MAIT.

## Tutorials
The repository includes several Jupyter Notebooks that serve as tutorials. These notebooks cover various aspects of the pipeline and demonstrate how to use different components effectively. Below you can find a list of available tutorials:

1. [Tutorial 1: Prediction of antimicrobial resistance for Azithromycin](https://github.com/PERSIMUNE/MAIT/blob/main/Tutorials/MAIT_Tutorial_Azithromycin_pub.html)
2. [Tutorial 2: Prediction of antimicrobial resistance for Ciprofloxacin](https://github.com/PERSIMUNE/MAIT/blob/main/Tutorials/MAIT_Tutorial_Ciprofloxacin_pub.html)
3. [Tutorial 3: Prediction of Dementia](https://github.com/PERSIMUNE/MAIT/blob/main/Tutorials/MAIT_Tutorial_Dementia_pub.html)
4. [Tutorial 4: Prediction of Breast Cancer](https://github.com/PERSIMUNE/MAIT/blob/main/Tutorials/MAIT_Tutorial_BreastCancer_pub.html)

## How to Cite
If you use this pipeline in your research, please cite it as follows:

```
@misc{marandi2025medicalartificialintelligencetoolbox,
      title={Medical artificial intelligence toolbox (MAIT): an explainable machine learning framework for binary classification, survival modelling, and regression analyses}, 
      author={Ramtin Zargari Marandi and Anne Svane Frahm and Jens Lundgren and Daniel Dawson Murray and Maja Milojevic},
      year={2025},
      eprint={2501.04547},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.04547}, 
}
```

## License
This pipeline is free to use for research purposes. Please ensure you follow the licenses of the individual packages used within this pipeline. For more details, refer to the `LICENSE` file in the repository.

---

We hope you find this pipeline useful for your machine learning projects. If you encounter any issues or have any questions, feel free to open an issue on GitHub.
![Alt Text](https://github.com/PERSIMUNE/MAIT/blob/main/MAIT_results_examples.gif)
