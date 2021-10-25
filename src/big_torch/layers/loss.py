import numpy as np
from .abstract import AbstractLayer

class CrossEntropy(AbstractLayer):
    def _fwd_prop(self, result_tuple):
        """
            Warning: Do not call it implicitly

            Params: 
                result_tuple - array like collection of (predicted, true results)
        """
        y = result_tuple[1]
        y_hat = result_tuple[0]

        losses = - (y * np.log(y_hat))

        return np.mean(losses), y_hat

    def _bckwd_prop(self, X, d_out):
        """
            Warning: Do not call it implicitly

            Params: 
                * X - prediction vector
                * d_out - vector of true results
        """
        return (d_out * np.power(X, -1)) / X.shape[0], None