from .abstract import ParametrizedLayer
from ..core.warnings import restrict_parallel
import numpy as np



class LinearLayer(ParametrizedLayer):
    def __init__(self, in_size, out_size):
        self.W = np.random.randn(in_size, out_size)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]
        self.gradW = None
        self.gradB = None
        self.gradInput = None

    @restrict_parallel
    def _fwd_prop(self, X):
        self.X = X
        return X.dot(self.W) + self.b

    @restrict_parallel
    def _bckwd_prop(self, grad_out):
        self.gradW = self.X.T.dot(grad_out)
        self.gradB = np.mean(grad_out, axis=0)
        self.gradInput = grad_out.dot(self.W.T)
        return self.gradInput, [self.gradW, self.gradB]