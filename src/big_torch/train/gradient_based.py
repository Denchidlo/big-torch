from ..core.utils import ModuleAggregator

opitmizer_registry = ModuleAggregator()

class GradientDecent():
    def __init__(self) -> None:
        pass

    def train(self, model, x_train, y_train, x_val=None, y_val=None, eta=1e-4, eps=1e-5, max_iter=None):
        _eps = 10*eps
        max_iter = abs(max_iter)
        n_iter = 0

        prev_l0 = None
        

        while  True:
            l0, l1 = model.ask_oracul(x_train, y_train, level=2)
            n_iter += 1
            print(f"Epoch {n_iter} - loss: {l0}")

            model.params.transform(l1, eta)

            _eps = abs(prev_l0 - l0) if prev_l0 != None else 2*eps
            prev_l0 = l0

            if (_eps < eps) or n_iter > max_iter:
                break

        return model
