import numpy as np

from ..core.computational_graph import LayerComponent
from .abstract import AbstractLayer, layer_registry
from .layer_mixins import DependecyCallMixin


@layer_registry.register('placeholder')
class Placeholder(AbstractLayer):
    output_names = ['y']

    def __init__(self, shape) -> None:
        self.shape = shape
        self._component = None

    def __call__(self):
        if getattr(self, '_component', None) == None:
            self._component = LayerComponent(self)
            self._component.dependencies = self._component.outputs
            return self._component
        else:
            return self._component

    def _fwd_pass(self, X):
        return X, None

    def _bwd_pass(self, X, d_out):
        return d_out, None
