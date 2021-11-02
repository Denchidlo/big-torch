from ..core.utils import ModuleAggregator


optimizator_registry = ModuleAggregator('optimizator_registry')


class Optimizator:
    def fit_transform(self, model, x, y, learn_info, n_jobs=1):
        raise NotImplementedError()


@optimizator_registry.register('gradient_decent')
class GradientDecent(Optimizator):
    def __init__(self, eta) -> None:
        self.eta = eta

    def fit_transform(self, model, x, y, learn_info, n_jobs=1):
        l0, l1 = model.ask_oracle(x, y, n_jobs=n_jobs)
        model.params.transform(l1, self.eta)

        learn_info["l0"] = l0


@optimizator_registry.register('momentum')
class ClassicMomentum(Optimizator):
    def __init__(self, eta, gamma) -> None:
        self.eta = eta
        self.gamma = gamma
        self._momentum = None

    def fit_transform(self, model, x, y, learn_info, n_jobs=1):
        if self._momentum is None:
            self._momentum = [layer.blank() for layer in reversed(model.params.layers)]
        l0, l1 = model.ask_oracle(x, y, n_jobs=n_jobs)
        learn_info["l0"] = l0
        self._momentum = [
            layer.self_mul(self.gamma).change(step_grad, -1 * self.eta) 
            for layer, step_grad in zip(self._momentum, l1)
        ]
        model.params.transform(
            [layer.get_context() for layer in self._momentum],
            1
        )
         

@optimizator_registry.register('nesterov')
class NesterovMomentum(Optimizator):
    def __init__(self, eta, gamma) -> None:
        self.eta = eta
        self.gamma = gamma
        self._momentum = None

    def fit_transform(self, model, x, y, learn_info, n_jobs=1):
        if self._momentum is None:
            self._momentum = [layer.blank() for layer in reversed(model.params.layers)]
        model.params.transform(
            [layer.get_context() for layer in self._momentum],
            1
        )
        l0, l1 = model.ask_oracle(x, y, n_jobs=n_jobs)
        learn_info["l0"] = l0
        self._momentum = [
            layer.self_mul(self.gamma).change(step_grad, -1 * self.eta) 
            for layer, step_grad in zip(self._momentum, l1)
        ]