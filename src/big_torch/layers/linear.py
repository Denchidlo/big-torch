import numpy as np

from ..preprocessing.initializers import initializer_registry
from .abstract import AbstractLayer, layer_registry, ParametrizedObject
from .layer_mixins import DependecyCallMixin, GradientStacker1DMixin


@layer_registry.register("linear")
class Dense(
    AbstractLayer, ParametrizedObject, DependecyCallMixin, GradientStacker1DMixin
):
    output_names = ["y"]

    def __init__(self, shape, kernel_initializer="xavier_normal", b_initial=0):
        self.shape = shape

        self.W = initializer_registry[kernel_initializer](shape)
        self.b = b_initial * np.ones((1, shape[1]))

    def _fwd_pass(self, X):
        return X.dot(self.W) + self.b, X

    def _bwd_pass(self, X, d_out):
        grad_W = X.T.dot(d_out)
        grad_b = np.mean(d_out, axis=0)
        grad_in = d_out.dot(self.W.T)

        return grad_in, [grad_W, grad_b]

    def change(self, step, eta):
        self.W -= eta * step[0]
        self.b -= eta * step[1]
        return self

    def blank(self):
        return Dense(self.shape, kernel_initializer="blank", b_initial=0)

    def get_context(self):
        return self.W, self.b

    def average(self, gradients_list):
        w_list = [el[0] for el in gradients_list]
        b_list = [el[1] for el in gradients_list]
        return np.mean(w_list, axis=0), np.mean(b_list, axis=0)

    def apply(self, func, context=None):
        c_W, c_b = None, None if context is None else context
        self.W = func(self.W, c_W)
        self.b = func(self.b, c_b)
        return self

    def context_binary_operation(self, lhs, rhs, operation):
        w_res = operation(lhs[0], rhs[0])
        b_res = operation(lhs[1], rhs[1])
        return w_res, b_res
