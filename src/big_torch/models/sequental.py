import numpy as np

from ..layers.abstract import ParametrizedLayer
from .shared import BasicModelParams, open_pool_session, close_pool_session, POOL


class Sequental:
    _ModelParams = BasicModelParams

    class _ModelInstance:
        def __init__(self, fabric):
            self.common = fabric
            self.bprop_context = []
            self.gradient_context = []

        def _forward(self, X):
            self.bprop_context = []
            for layer in self.common.layers:
                X, bprop = layer._fwd_pass(X)
                self.bprop_context.append(bprop)

            self.last_predict = X
            return X

        def _back_propogate(self, y):
            # Loss function gradient init
            # Note that it's the only place were we call loss function
            loss, predict = self.common.loss_function._fwd_pass(
                (self.last_predict, y))
            d_out, _ = self.common.loss_function._bwd_pass(predict, y)

            idx = len(self.common.layers)
            for layer in reversed(self.common.layers):
                idx -= 1
                d_out, grad = layer._bwd_pass(self.bprop_context[idx], d_out)

                if not grad is None:
                    self.gradient_context.append(grad)

            return loss, self.gradient_context

        @staticmethod
        def train_step(args):
            self, x, y = args
            self._forward(x)
            loss, gradient_context = self._back_propogate(y)

            return loss, gradient_context

    def __init__(self) -> None:
        self.layers = []
        self.loss_function = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss_function = loss

    def compile(self):
        self._parametrised_layers = [
            layer for layer in self.layers if isinstance(layer, ParametrizedLayer)
        ]
        self.params = self._ModelParams(self._parametrised_layers)

    def ask_oracle(self, X, y, n_jobs=1):

        if n_jobs <= 0:
            n_jobs = 4

        if n_jobs != 1:
            payload = [
                [self._ModelInstance(self) for _ in range(n_jobs)],
                np.array_split(X, n_jobs),
                np.array_split(y, n_jobs),
            ]
            payload = list(zip(*payload))

            global POOL
            results = POOL.map(
                self._ModelInstance.train_step, iterable=payload, chunksize=1
            )

            loss = np.mean([el[0] for el in results])
            grads = self.params._avg_grads([el[1] for el in results])
        else:
            loss, grads = self._ModelInstance.train_step(
                (self._ModelInstance(self), X, y)
            )

        return loss, grads

    def predict(self, X):
        worker = self._ModelInstance(self)
        return worker._forward(X)

    def eval(self, x, y, metrics=None):
        y_hat = self._ModelInstance(self)._forward(x)
        loss, _ = self.loss_function._fwd_pass((y_hat, y))
        metrics = [metric(y_hat, y) for metric in metrics] if len(metrics) != 0 else []
        return loss, metrics
