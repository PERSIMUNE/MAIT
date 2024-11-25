# Use NVIDIA's CUDA 12.5 base image with cuDNN support
FROM nvidia/cuda:12.5.0-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Install dependencies (if needed for your application)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -fsSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Update the Conda package manager
RUN conda update -n base -c defaults conda

# Copy the Conda environment YAML file to the working directory
COPY environment.yml .

# Create a Conda environment from the YAML file
RUN conda env create -f environment.yml

# Activate the Conda environment in the bash shell and set it in the environment PATH
SHELL ["/bin/bash", "-c"]
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" >> ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH
RUN ln -s /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin/jupyter /usr/local/bin/jupyter

# Install NVIDIA Container Toolkit to allow GPU access
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-container-toolkit

# Expose the port that Jupyter Notebook will run on
EXPOSE 8888

# Set the command to run Jupyter Notebook when the container starts
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
