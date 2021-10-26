from ..layers.abstract import ParametrizedLayer

class Model:

    class _ModelInstance:
        def __init__(self, fabric):
            self.common = fabric
            self.bprop_context = []
            self.gradient_context = []

        def _forward(self, X):
            self.bprop_context = []
            for layer in self.common.layers:
                X, bprop = layer._fwd_prop(X)
                self.bprop_context.append(bprop)

            self.last_predict = X
            return X

        def _back_propogate(self, y):
            # Loss function gradient init
            # Note that it's the only place were we call loss function
            loss, predict = self.common.loss_function._fwd_prop((self.last_predict, y))
            d_out, _ = self.common.loss_function._bckwd_prop(predict, y)
            
            idx = len(self.common.layers)
            for layer in reversed(self.common.layers):
                idx -= 1
                d_out, grad = layer._bckwd_prop(self.bprop_context[idx], d_out)
                
                if not grad is None: 
                    self.gradient_context.append(grad)
            
            return loss, self.gradient_context

        def train_step(self, x, y, show_metrics=False):
            """
                Parameters:
                 * x - Flatten vector x
                 * y - Target vector
            """
            self._forward(x)
            loss, gradient_context = self._back_propogate(y)

            return loss, gradient_context

    class _ModelParams:
        def __init__(self, layers) -> None:
            self.layers = layers

        def transform(self, gradients, eta):
            for idx, layer in enumerate(reversed(self.layers)):
                layer.change(gradients[idx], eta)
                
    def __init__(self) -> None:
        self.layers = []
        self.loss_function = None
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss_function = loss

    def build(self):
        self._parametrised_layers = [layer for layer in self.layers if isinstance(layer, ParametrizedLayer)]
        self.params = self._ModelParams(self._parametrised_layers)

    def ask_oracul(self, X, y, n_jobs=None, level=2):
        worker = self._ModelInstance(self)
        if level == 2:
            return worker.train_step(X, y)

    def predict(self, X):
        worker = self._ModelInstance(self)
        return worker._forward(X)