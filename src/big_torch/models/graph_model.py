from typing import Iterable
from .shared import BasicModelParams
from ..core.computational_graph import ComputationalComponent, ComputationalGraph, LayerComponent
from ..layers.loss import layer_registry, AbstractLayer


class GraphModel:
    """
        Attention: GraphModel is running does not support parallezation.
        Multithreading can be added later, after trasfer main computational modules into C/C++
    """
    _ModelParams = BasicModelParams

    class _ModelInstance:
        def __init__(self, model) -> None:
            self.computational_graph = model.graph
            self.model = model

        def _forward(self, X):
            graph = self.computational_graph
            graph.clear_execution_context()
            result = graph._fwd_pass(X if len(graph.input_components) > 1 else [X])
            return result if len(graph.outputs) > 1 else result[0]

        def _back_propogate(self, y):
            graph = self.computational_graph
            outputs = graph._fetch_execution_context('inputs', graph.outputs)

            y = [y] if len(graph.output_components) == 1 else y

            def _calculate_loss():
                l0_vals, l1_vals = [], []
                for loss, y_true, y_hat in zip(self.model.loss_functions, y, outputs):
                    l0, _ = loss._fwd_pass((y_hat, y_true))
                    l0_vals.append(l0)
                    l1, _ = loss._bwd_pass(y_hat, y_true)
                    l1_vals.append(l1)

                return l0_vals, l1_vals

            l0_vals, l1_vals = _calculate_loss()
            graph._bwd_pass(l1_vals)

            return l0_vals[0], graph._execution_context['param_grad']

        @staticmethod
        def train_step(args):
            self, x, y = args
            self._forward(x)
            loss, gradient_context = self._back_propogate(y)

            return loss, gradient_context

    def __init__(self, inputs, outputs) -> None:
        self.graph = None
        self.loss_functions = None


        self.inputs = inputs
        self.outputs = outputs

    def compile(self, loss):
        if self.graph is None:
            self.loss_functions = loss if isinstance(loss, Iterable) else [loss]
            
            self.graph = ComputationalGraph(
                input_components=self.inputs,
                output_components=self.outputs
            )
            self.graph.excluded_id = [loss.id for loss in self.loss_functions]
            self.graph.unroll()

            self.params = GraphModel._ModelParams(self.graph._parametrized_layers)
        else:
            raise UserWarning(
                'Compilation terminated. Recompilation of present model can lead to undefined behaviour')

    def ask_oracle(self, X, y, n_jobs):
        return self._ModelInstance.train_step(
            (self._ModelInstance(self), X, y)
        )

    def predict(self, X):
        worker = self._ModelInstance(self)
        return worker._forward(X)

    def eval(self, x, y, metrics=[]):
        instance = self._ModelInstance(self)
        y_hat = instance._forward(x)
        y, y_hat = [y], [y_hat] if len(self.graph.output_components) == 1 else tuple(y, y_hat)

        l0_vals = []
        metrics = [metric(y_hat[0], y[0]) for metric in metrics] if len(metrics) != 0 else []

        for loss, y_true, y_hat in zip(self.loss_functions, y, y_hat):
            l0, _ = loss._fwd_pass((y_hat, y_true))
            l0_vals.append(l0)

        return l0_vals[0], metrics
