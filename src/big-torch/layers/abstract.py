from abc import abstractmethod
import numpy as np

class AbstractLayer():
    @abstractmethod
    def _fwd_prop():
        raise NotImplementedError()

    @abstractmethod
    def _bckwd_prop():
        raise NotImplementedError()

class ParametrizedLayer(AbstractLayer):
    pass
