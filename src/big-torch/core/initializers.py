import numpy as np
import math
from .utils import ModuleAggregator

initializer_registry = ModuleAggregator()

@initializer_registry.register(name='xavier_uniform')
def uniform_xavier_initializer(shape):
    return np.random.uniform(-1., 1., size=shape) * np.sqrt(6. / np.sum(shape))

@initializer_registry.register(name='xavier_normal')
def normal_xavier_initializer(shape):
    return np.random.randn(*shape) * np.sqrt(2. / np.sum(shape))

@initializer_registry.register(name='kaiming')
def kaiming_initializer(shape):
    return np.randn(*shape) * math.sqrt(2. / shape[0])