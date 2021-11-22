from os import name
import numpy as np
import math
from ..utils.registry import ModuleAggregator

initializer_registry = ModuleAggregator("weights_initialiser")


@initializer_registry.register(name="blank")
def blank_initializer(shape):
    return np.zeros(shape)


@initializer_registry.register(name="xavier_uniform")
def uniform_xavier_initializer(shape):
    return np.random.uniform(-1.0, 1.0, size=shape) * np.sqrt(6.0 / np.sum(shape))


@initializer_registry.register(name="xavier_normal")
def normal_xavier_initializer(shape):
    return np.random.randn(*shape) * np.sqrt(2.0 / np.sum(shape))


@initializer_registry.register(name="kaiming")
def kaiming_initializer(shape):
    return np.randn(*shape) * math.sqrt(2.0 / shape[0])
