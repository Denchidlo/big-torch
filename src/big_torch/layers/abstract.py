from abc import abstractmethod
import numpy as np

class AbstractLayer():
    def __init__(self, shape) -> None:
        self.shape = shape

    @abstractmethod
    def _fwd_prop(self, X):
        raise NotImplementedError()

    @abstractmethod
    def _bckwd_prop(self, X, d_out):
        raise NotImplementedError()


class ParametrizedLayer(AbstractLayer):
    def __init__(self, shape) -> None:
        super().__init__(shape)

    def change(self, step, eta):
        raise NotImplementedError()