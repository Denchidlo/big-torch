import numpy as np

from ..abstract import AbstractLayer
from ..layer_mixins import DependecyCallMixin
from ...preprocessing.sequences import mask_all_equal_value


class RNNLossWrapper1D(AbstractLayer, DependecyCallMixin):
    def __init__(self, loss_layer, mask_creator=mask_all_equal_value(np.NaN, axis=2), agg=np.mean) -> None:
        self.layer = loss_layer
        self.output_names = self.layer.output_names
        self.mask_creator = mask_creator
        self.agg = agg

    def _fwd_pass(self, X):
        y_predict, y_true = X
        assert y_true.shape == y_predict.shape, f'Forecast and original shape are not equal true:{y_true.shape}, predict:{y_predict.shape}'

        batch_size, seq_size, _ = y_true.shape
        _loss = np.empty((batch_size, seq_size))

        for window in range(seq_size):
            _y_slice = y_true[:, window, :]
            _y_hat_slice = y_predict[:, window, ]

            _loss_slice, _ = self.layer._fwd_pass((_y_hat_slice, _y_slice))
            _loss[:, window] = _loss_slice

        return self.agg(_loss), y_predict

    def _bwd_pass(self, X, d_out):
        y_true, y_predict = d_out, X
        assert y_true.shape == y_predict.shape, f'Forecast and original shape are not equal true:{y_true.shape}, predict:{y_predict.shape}'

        _, seq_size, _ = y_true.shape
        _d_out = np.empty(y_true.shape)

        for window in range(seq_size):
            _y_slice = y_true[:, window, :]
            _y_hat_slice = y_predict[:, window, ]

            _d_out_slice, _ = self.layer._bwd_pass(_y_hat_slice, _y_slice)
            _d_out[:, window, :] = _d_out_slice

        return _d_out * self.mask_creator(y_true), y_predict


class RNNActivationWrapper1D(AbstractLayer, DependecyCallMixin):
    def __init__(self, loss_layer) -> None:
        self.layer = loss_layer
        self.output_names = self.layer.output_names

    def _fwd_pass(self, X):
        _, seq_size, _ = X.shape
        _out = None
        _context = None

        for window in range(seq_size):
            _x_slice = X[:, window, :]

            _out_slice, _context_slice = self.layer._fwd_pass(_x_slice)
            if _context is None:
                _, _out_vec_shape = _out_slice.shape
                _, _c_vec_shape = _context_slice.shape
                _out = np.empty((*X.shape[:-1], _out_vec_shape))
                _context = np.empty((*X.shape[:-1], _c_vec_shape))
            _context[:, window, :] = _context_slice
            _out[:, window, :] = _out_slice

        return _out, _context

    def _bwd_pass(self, X, d_out):
        _, seq_size, _ = X.shape
        _d_in = None

        for window in range(seq_size):
            _x_slice = X[:, window, :]
            _d_out_slice = d_out[:, window, :]

            _d_in_slice, _ = self.layer._bwd_pass(_x_slice, _d_out_slice)
            if _d_in is None:
                _, _d_in_vec_shape = _d_in_slice.shape
                _d_in = np.empty((*X.shape[:-1], _d_in_vec_shape))
            _d_in[:, window, :] = _d_in_slice

        return _d_in, None
