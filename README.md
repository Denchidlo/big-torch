# big-torch library

# Content

## Installation:
pip install -i https://test.pypi.org/simple/ bigtorch-denissimo


## Folders and files:
* .github/workflow - Here I'll create several actions like docker-image-publishing and library-publishing
* src/big_torch - My neural network library
* [src/some.ipynb](https://github.com/Denchidlo/big-torch/blob/master/src/some.ipynb) - Demo
* container - Folder which you'll build in order to use my lib on MNIST task 

### 1) Github actions (IN PROGRESS)
* publish-to-pypi.yml - Deploy big-torch to PyPi-Test

### 2) Big-troch lib

#### It has the following structure:
* core - Mix of different utilities I'm currently unable to gather in one logical module
* layers - All possible layers that can form my neural network. They share common interface [AbstractLayer, ParametrizedLayer], see [abstract.py](https://github.com/Denchidlo/big-torch/blob/master/src/big_torch/layers/abstract.py)
* model - Model builder
* train - Is represented by [optimization fabric](https://github.com/Denchidlo/big-torch/blob/master/src/big_torch/train/fabric.py) which can create your own custom optimization pipeline by using different [optimizers](https://github.com/Denchidlo/big-torch/blob/master/src/big_torch/train/updaters.py), [frame_generators](https://github.com/Denchidlo/big-torch/blob/master/src/big_torch/train/frame_generators.py) and some more utils

### 3) Container (IN PROGRESS)

This folder will provide you easy to launch docker container. You'll just need to specify model structure and credentials to neptune.ai 