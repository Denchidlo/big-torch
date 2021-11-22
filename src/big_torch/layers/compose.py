import numpy as np

from .abstract import AbstractLayer, layer_registry
from .layer_mixins import DependecyCallMixin, GradientStacker1DMixin


@layer_registry.register("concat")
class Concat(AbstractLayer, DependecyCallMixin, GradientStacker1DMixin):
    output_names = ["y"]

    def __init__(self, shape) -> None:
        self.shape = shape

    def _fwd_pass(self, X):
        return np.sum(X, axis=0), X

    def _bwd_pass(self, X, d_out):
        return (d_out for _ in range(len(X))), None


@layer_registry.register("mul-concat")
class MultiplicationConcat(AbstractLayer, DependecyCallMixin, GradientStacker1DMixin):
    output_names = ["y"]

    def __init__(self, shape) -> None:
        self.shape = shape

    def _fwd_pass(self, X):
        return np.prod(X, axis=0), X

    def _bwd_pass(self, X, d_out):
        _len = len(X)
        _d_out = [
            np.prod(np.array(X)[np.arange(_len) != idx], axis=0) * d_out for idx in range(_len)
        ]
        return _d_out, None
