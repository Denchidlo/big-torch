import numpy as np

from .abstract import AbstractLayer, layer_registry
from .layer_mixins import DependecyCallMixin


@layer_registry.register('concat')
class Concat(AbstractLayer, DependecyCallMixin):
    output_names = ['y']

    def __init__(self, shape) -> None:
        self.shape = shape

    def _fwd_pass(self, X):
        return np.sum(X, axis=0), None

    def _bwd_pass(self, X, d_out):
        return (d_out, d_out), None
