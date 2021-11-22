from .shared import RNNBase1D
from ...preprocessing.initializers import initializer_registry
from ...layers.abstract import layer_registry
from ...core.computational_graph import ComputationalGraph
from ... import layers as l


class SimpleRNN(RNNBase1D):
    output_names = ['y']

    def __init__(self, shape, state_initializer='blank', nonlinearity='tanh', lin_kwargs={}) -> None:
        self.initializer = initializer_registry[state_initializer]
        self.nonlinearity = nonlinearity
        super().__init__(shape, nonlinearity=nonlinearity, lin_kwargs=lin_kwargs)

    def _unpack_input(self, x):
        s_shape = self.shape['state'] if isinstance(
            self.shape, dict) else self.shape[1]
        batch_size, _, _ = x.shape

        return x, (self.initializer((batch_size, s_shape)),)

    @staticmethod
    def _create_block(shape, nonlinearity, lin_kwargs={}) -> ComputationalGraph:
        x_shape, s_shape, y_shape = shape['input'], shape['state'], shape['output'] if isinstance(
            shape, dict) else shape

        _nonlinear_layer_type = layer_registry[nonlinearity]

        _in_x = l.variables.Placeholder(x_shape)()
        _in_s = l.variables.Placeholder(s_shape)()

        _in_x_to_compose = l.linear.Dense(
            (x_shape, s_shape), **lin_kwargs)(_in_x)
        _in_s_to_compose = l.linear.Dense(
            (s_shape, s_shape), **lin_kwargs)(_in_s)

        _compose_out = l.compose.Concat((s_shape, s_shape))(
            [_in_x_to_compose['y'], _in_s_to_compose['y']])
        _out_s = _nonlinear_layer_type((s_shape, s_shape))(_compose_out)

        _out_s_to_lin_y = l.linear.Dense((s_shape, y_shape))(_out_s)
        _out_y = _nonlinear_layer_type((y_shape, y_shape))(_out_s_to_lin_y)

        computational_graph = ComputationalGraph(
            input_components=[_in_x, _in_s],
            output_components=[_out_y, _out_s]
        )
        computational_graph.unroll()

        return computational_graph, computational_graph._parametrized_layers


class LSTM(RNNBase1D):
    output_names = ['y']

    def __init__(self, shape, state_initializer='blank', nonlinearity={'gate': 'sigmoid', 'cell': 'tanh', 'out': 'tanh'}, lin_kwargs={}) -> None:
        self.initializer = initializer_registry[state_initializer]
        self.nonlinearity = nonlinearity
        super().__init__(shape, nonlinearity=nonlinearity, lin_kwargs=lin_kwargs)

    def _unpack_input(self, x):
        s_shape = self.shape['state'] if isinstance(
            self.shape, dict) else self.shape[1]
        batch_size, _, _ = x.shape

        return x, (self.initializer((batch_size, s_shape)), self.initializer((batch_size, s_shape)))

    @staticmethod
    def _create_block(shape, nonlinearity, lin_kwargs={}) -> ComputationalGraph:
        x_shape, c_shape, y_shape = shape['input'], shape['state'], shape['output'] if isinstance(
            shape, dict) else shape

        assert c_shape == y_shape, f'Cell and output gates should be of the same length'

        _nonlinear_layer_type_gate = layer_registry[nonlinearity['gate']]
        _nonlinear_layer_type_cell = layer_registry[nonlinearity['cell']]
        _nonlinear_layer_type_out = layer_registry[nonlinearity['out']]

        # Input section
        _in_x = l.variables.Placeholder(x_shape)()
        _in_c = l.variables.Placeholder(c_shape)()
        _in_h = l.variables.Placeholder(y_shape)()

        # Output gate section
        _x_to_output_lin = l.linear.Dense((x_shape, c_shape))(_in_x)
        _h_to_output_lin = l.linear.Dense((y_shape, c_shape))(_in_h)
        _output_lin = l.compose.Concat((c_shape, c_shape))([
            _x_to_output_lin['y'], _h_to_output_lin['y']
        ])
        _output_gate = _nonlinear_layer_type_gate(
            (c_shape, c_shape))(_output_lin)

        # Forget gate section
        _x_to_forget_lin = l.linear.Dense((x_shape, c_shape))(_in_x)
        _h_to_forget_lin = l.linear.Dense((y_shape, c_shape))(_in_h)
        _forget_lin = l.compose.Concat((c_shape, c_shape))([
            _x_to_forget_lin['y'], _h_to_forget_lin['y']
        ])
        _forget_gate = _nonlinear_layer_type_gate(
            (c_shape, c_shape))(_forget_lin)

        # Input gate section
        _x_to_input_lin = l.linear.Dense((x_shape, c_shape))(_in_x)
        _h_to_input_lin = l.linear.Dense((y_shape, c_shape))(_in_h)
        _input_lin = l.compose.Concat((c_shape, c_shape))([
            _x_to_input_lin['y'], _h_to_input_lin['y']
        ])
        _input_gate = _nonlinear_layer_type_gate(
            (c_shape, c_shape))(_input_lin)

        # Candidate cell section
        _x_to_candidate_cell_lin = l.linear.Dense((x_shape, c_shape))(_in_x)
        _h_to_candidate_cell_lin = l.linear.Dense((y_shape, c_shape))(_in_h)
        _candidate_cell_lin = l.compose.Concat((c_shape, c_shape))([
            _x_to_candidate_cell_lin['y'], _h_to_candidate_cell_lin['y']
        ])
        _candidate_cell_gate = _nonlinear_layer_type_cell(
            (c_shape, c_shape))(_candidate_cell_lin)

        _cell_add_component = l.compose.MultiplicationConcat((c_shape, c_shape))([
            _input_gate['y'], _candidate_cell_gate['y']
        ])

        _cell_old_component = l.compose.MultiplicationConcat((c_shape, c_shape))([
            _in_c['y'], _forget_gate['y']
        ])

        _out_cell = l.compose.Concat((c_shape, c_shape))([
            _cell_old_component['y'], _cell_add_component['y']
        ])

        _out_cell_to_h = _nonlinear_layer_type_out(
            (c_shape, c_shape))(_out_cell)
        _out_h = l.compose.MultiplicationConcat((c_shape, c_shape))([
            _output_gate['y'], _out_cell_to_h['y']
        ])

        computational_graph = ComputationalGraph(
            input_components=[_in_x, _in_c, _in_h],
            output_components=[_out_h, _out_cell, _out_h]
        )
        computational_graph.unroll()

        return computational_graph, computational_graph._parametrized_layers
