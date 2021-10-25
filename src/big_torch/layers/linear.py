from .abstract import ParametrizedLayer
from ..core.warnings import restrict_parallel
from ..core.initializers import initializer_registry
import numpy as np



class LinearLayer(ParametrizedLayer):
    def __init__(self, shape, w_init='xavier_normal', b_initial=0):
        self.shape = shape
        
        # TODO: Make initalization more flexible
        self.W = initializer_registry[w_init](shape)
        self.b = b_initial * np.ones((1, shape[1]))

        self.params = [self.W, self.b]

    def _fwd_prop(self, X):
        return X.dot(self.W) + self.b, X

    def _bckwd_prop(self, X, d_out):
        grad_W = X.T.dot(d_out) / d_out.dtype(d_out.shape[0])
        grad_b = np.mean(d_out, axis=0, keepdims=True)
        grad_in = d_out.dot(self.W.T)

        return grad_in, (grad_W, grad_b)

    def change(self, step, eta):
        self.W -= eta * step[0]
        self.b -= eta * step[1]