from .abstract import ParametrizedLayer
from ..core.warnings import restrict_parallel
from ..core.initializers import initializer_registry
import numpy as np



class LinearLayer(ParametrizedLayer):
    def __init__(self, shape, w_init='xavier_normal', b_initial=0):
        self.shape = shape
        
        # TODO: Make initalization more flexible
        self.W = initializer_registry[w_init](shape)
        self.b = b_initial * np.ones((shape[0], 1))

        self.params = [self.W, self.b]

    def _fwd_prop(self, X):
        return self.W.dot(X) + self.b, X

    def _bckwd_prop(self, X, d_out):
        grad_W = d_out.dot(X)
        grad_b = d_out
        grad_in = self.W.T.dot(d_out)

        return grad_in, (grad_W, grad_b)
