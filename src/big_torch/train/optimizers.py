from ..core.utils import ModuleAggregator


optimizator_registry = ModuleAggregator('optimizator_registry')


class Optimizator:
    def fit_transform(self, model, x, y, n_jobs):
        raise NotImplementedError()


@optimizator_registry.register('gradient_decent')
class GradientDecent(Optimizator):
    def __init__(self, eta) -> None:
        self.eta = eta

    def fit_transform(self, model, x, y, learn_info, n_jobs=1):
        l0, l1 = model.ask_oracle(x, y, n_jobs=n_jobs)
        model.params.transform(l1, self.eta)

        learn_info["l0"] = l0
