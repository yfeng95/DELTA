#!/usr/bin/env bash

export CONDA_ENV_NAME=delta
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.9

# eval "$(conda shell.bash hook)"
# conda activate $CONDA_ENV_NAME
# pip install -r requirements.txt
conda env create -f environment.yml
conda activate $CONDA_ENV_NAME
pip install --upgrade PyMCubes
pip install trimesh
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
