import numpy as np


from ... import layers as l
from ...layers.layer_mixins import ParamManagerMixin, DependecyCallMixin, GradientStacker1DMixin
from ..abstract import AbstractLayer
from ...core.computational_graph import ComputationalGraph


class RNNBase1D(AbstractLayer, ParamManagerMixin, DependecyCallMixin, GradientStacker1DMixin):
    """
        RNNBase1D - reccurent layer class handler

        Attention: RNNBase1D support 1 tensor for x -> (IN) and y -> (OUT). Thus x and y objects are 1d objects

        Graphical structure:
            x -> RNNBase1D(states_initial; graph) -> y
    """

    def __init__(self, shape, *args, **kwargs) -> None:
        self.shape = shape
        self.layer, self._parametrized_layers = self._create_block(
            shape, *args, **kwargs)

    @staticmethod
    def _create_block(shape, *args, **kwargs) -> ComputationalGraph:
        raise NotImplementedError()

    def _unpack_input(self, x):
        raise NotImplementedError()

    def _fwd_pass(self, X):
        X, states_initial = self._unpack_input(X)

        # Currently you can only work with (batch_size, sequence_size, vector_size) tensor
        shape = self.shape
        x_shape, s_shape, y_shape = shape['input'], shape['state'], shape['output'] if isinstance(
            shape, dict) else shape
        batch_size, sequence_length, vector_size = X.shape

        sequence_meta = []
        out = np.empty((batch_size, sequence_length, y_shape))

        _state_slice = states_initial
        for current_window in range(sequence_length):
            _x_slice = X[:, current_window, :]
            _out_slice, _grad_context = self.layer._fwd_pass(
                [_x_slice, *_state_slice])
            _y_slice, *_state_slice = _out_slice
            sequence_meta.append(_grad_context)
            out[:, current_window, :] = _y_slice
            if isinstance(self.layer, ComputationalGraph):
                self.layer.clear_execution_context()

        return out, sequence_meta

    def _bwd_pass(self, X, d_out):
        Y, _d_out_states_initial = self._unpack_input(d_out)

        shape = self.shape
        x_shape, s_shape, y_shape = shape['input'], shape['state'], shape['output'] if isinstance(
            shape, dict) else shape
        batch_size, sequence_length, vector_size = Y.shape

        param_gradient_list = []
        _in_grad = np.empty((batch_size, sequence_length, y_shape))

        current_window = sequence_length - 1
        _d_out_states = _d_out_states_initial
        for _ in range(sequence_length):
            _d_out_slice = Y[:, current_window, :]
            (_in_grad_slice, *_d_out_states), _param_gradients = self.layer._bwd_pass(
                X[current_window], [_d_out_slice, *_d_out_states])
            param_gradient_list.append(_param_gradients)
            _in_grad[:, current_window, :] = _in_grad_slice

        params_grads = self.layer.average(param_gradient_list)

        return _in_grad, params_grads

    def get_layers(self):
        return self.layer.get_layers()
