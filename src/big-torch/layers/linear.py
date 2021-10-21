from .abstract import ParametrizedLayer
from ..core.warnings import restrict_parallel
from ..core.initializers import initializer_registry
import numpy as np



class LinearLayer(ParametrizedLayer):
    def __init__(self, shape, w_init='xavier_normal', b_initial=0):
        self.shape = shape
        
        # TODO:
        # - Add initalization more flexible
        self.W = initializer_registry[w_init](shape)
        self.b = b_initial * np.ones((shape[0], 1))

        self.params = [self.W, self.b]

    @restrict_parallel
    def _fwd_prop(self, X):
        self.X = X
        return self.W.dot(X) + self.b

    @restrict_parallel
    def _bckwd_prop(self, dOut):
        grad_W = dOut.dot(self.X.T)
        grad_b = dOut
        grad_in = self.W.T.dot(dOut)

        return grad_in, (grad_W, grad_b)