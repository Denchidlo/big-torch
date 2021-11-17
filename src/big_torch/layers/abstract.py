from abc import abstractmethod
import numpy as np

from ..utils.registry import ModuleAggregator

layer_registry = ModuleAggregator(registry_name='layers')


class AbstractLayer:
    def __init__(self, shape) -> None:
        self.shape = shape

    @abstractmethod
    def _fwd_pass(self, X):
        raise NotImplementedError()

    @abstractmethod
    def _bwd_pass(self, X, d_out):
        raise NotImplementedError()


class ParametrizedLayer(AbstractLayer):
    def __init__(self, shape) -> None:
        super().__init__(shape)

    @abstractmethod
    def blank(self):
        raise NotImplementedError()

    @abstractmethod
    def get_context(self):
        raise NotImplementedError()

    @abstractmethod
    def change(self, step, eta):
        raise NotImplementedError()

    @abstractmethod
    def average(self, gradients_list):
        raise NotImplementedError()

    @abstractmethod
    def apply(self, func):
        raise NotImplementedError()

    @abstractmethod
    def binary_operation(lhs, rhs, operation):
        raise NotImplementedError()
