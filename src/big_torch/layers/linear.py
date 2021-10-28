from .abstract import AbstractLayer, layer_registry, ParametrizedLayer
from ..core.warnings import restrict_parallel
from ..preprocessing.initializers import initializer_registry
import numpy as np


class LinearLayer(ParametrizedLayer):
    def __init__(self, shape, w_init="xavier_normal", b_initial=0):
        self.shape = shape

        # TODO: Make initalization more flexible
        self.W = initializer_registry[w_init](shape)
        self.b = b_initial * np.ones((1, shape[1]))

    def _fwd_prop(self, X):
        return X.dot(self.W) + self.b, X

    def _bckwd_prop(self, X, d_out):
        grad_W = X.T.dot(d_out)
        grad_b = np.mean(d_out, axis=0)
        grad_in = d_out.dot(self.W.T)

        return grad_in, [grad_W, grad_b]

    def change(self, step, eta):
        self.W -= eta * step[0]
        self.b -= eta * step[1]

    def average(self, gradients_list):
        w_list = [el[0] for el in gradients_list]
        b_list = [el[1] for el in gradients_list]
        return np.mean(w_list, axis=0), np.mean(b_list, axis=0)
