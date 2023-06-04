Active Learning of Hydrogen Combustion Reaction.

## Installation and Dependencies


The developer installation is available and for that you need to first clone H2Combustion from this repository:

    git clone https://github.com/THGLab/H2Combustion_AL.git


This repository is built on top of the NewtonNet repository: https://github.com/THGLab/NewtonNet

We recommend using conda environment to install dependencies of this library.
Please install (or load) conda and then proceed with the following commands:

    conda create --name torch-gpu python=3.7
    conda activate torch-gpu
    conda install -c conda-forge numpy scipy pandas ase pyyaml tqdm scikit-learn attrs xlsxwriter
    conda install -c pytorch pytorch torchvision cudatoolkit=11.3
    conda install -c conda-forge plumed py-plumed
    pip install rmsd

This should take ~20 minute to install. Now, you can run combust and md modules anywhere on your computer as long as you are in the `torch-gpu` environment.

## OS requirement
This package is supported for macOS and Linux. The package has been tested on the following systems:
macOS: Big Sur (11.4)
Linux: x86_64 GNU/Linux


## Guidelines
- You can find several models inside the scripts directory that rely on the implemented modules in the combust and md library. 
The yaml file control setting of the model. Please modify parameters using the yaml files when retraining.

- The documentation of the modules are available at most cases. Please look up local classes or functions
and consult with the docstrings in the code.

- Some paths are hardcoded into the code that is specific to developer. Make sure to go through and change them before using the code.


## Demo
See script/demo/md_demo.py for using trained model to run md simulation. The expected output is included in script/demo/demo_output/