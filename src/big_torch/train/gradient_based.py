from ..core.utils import ModuleAggregator

opitmizer_registry = ModuleAggregator()

class GradientDecent():
    def __init__(self) -> None:
        pass

    def train(self, X, y, model, eta, eps, max_iter=None):
        _eps = 10*eps
        max_iter = abs(max_iter)
        n_iter = 0

        prev_l0 = None

        while  _eps is None or (_eps < eps) or n_iter <= max_iter:
            l0, l1 = model.ask_oracul(X, y)
            print(f"Epoch 1 - loss: {l0}")

            model.transform(l1, eta)

            _eps = abs(prev_l0 - l0) if prev_l0 != None else 2*eps
            prev_l0 = l0
            n_iter += 1

        return model
