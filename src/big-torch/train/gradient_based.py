from ..core.utils import ModuleAggregator

opitmizer_registry = ModuleAggregator()

class GradientDesent():
    def __init__(self) -> None:
        pass

    def train(self, X, y, model, eta, eps, max_iter=None):
        _eps = None
        n_iter = 0

        prev_l0 = None

        while (_eps < eps) or n_iter <= max_iter:
            l0, l1 = model.calculate_all(X, y)

            # TODO: Rewrite all when I'll create model <-> optimizer interaction interface
            gradient_step = -eta * (model.params_mean(l1))
            model.change_params_for(gradient_step)

            _eps = abs(prev_l0 - l0) if prev_l0 != None else 2*eps
            prev_l0 = l0

        return model
