# big-torch library

# Content

### 1) Big-troch lib

#### It has the following structure:
* core - Mix of different utilities I'm currently unable to gather in one logical module
* layers - All possible layers that can form my neural network. They share common interface [AbstractLayer, ParametrizedLayer], see [abstract.py](https://github.com/Denchidlo/big-torch/blob/master/src/big_torch/layers/abstract.py)
* model - Model builder
* train - Is represented by [optimization fabric](https://github.com/Denchidlo/big-torch/blob/master/src/big_torch/train/fabric.py) which can create your own custom optimization pipeline by using different [optimizers](https://github.com/Denchidlo/big-torch/blob/master/src/big_torch/train/updaters.py), [frame_generators](https://github.com/Denchidlo/big-torch/blob/master/src/big_torch/train/frame_generators.py) and some more utils
* preprocessing - Utility for data preprocessing