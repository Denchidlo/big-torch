from ..utils.registry import ModuleAggregator
import numpy as np


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
            self._momentum = [layer.blank()
                              for layer in reversed(model.params.layers)]
        l0, l1 = model.ask_oracle(x, y, n_jobs=n_jobs)
        learn_info["l0"] = l0
        self._momentum = [
            layer.apply(lambda x, _: x *
                        self.gamma).change(step_grad, -1 * self.eta)
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
            self._momentum = [layer.blank()
                              for layer in reversed(model.params.layers)]
        model.params.transform(
            [layer.get_context() for layer in self._momentum],
            1
        )
        l0, l1 = model.ask_oracle(x, y, n_jobs=n_jobs)
        learn_info["l0"] = l0
        self._momentum = [
            layer.apply(lambda x, _: x *
                        self.gamma).change(step_grad, -1 * self.eta)
            for layer, step_grad in zip(self._momentum, l1)
        ]


@optimizator_registry.register('Adagrad')
class Adagrad(Optimizator):
    def __init__(self, eta, eps=1e-8) -> None:
        self.eta = eta
        self.eps = eps
        self._cumulative_grad = None

    def fit_transform(self, model, x, y, learn_info, n_jobs=1):
        if self._cumulative_grad is None:
            self._cumulative_grad = [layer.blank()
                                     for layer in reversed(model.params.layers)]

        l0, l1 = model.ask_oracle(x, y, n_jobs=n_jobs)
        grads = []

        for grad_context, layer in zip(l1, self._cumulative_grad):
            grads.append(layer.context_binary_operation(
                grad_context,
                layer.get_context(),
                lambda lhs, rhs: (lhs / (np.sqrt(rhs) + self.eps))
            ))

        for grad_context, layer in zip(l1, self._cumulative_grad):
            layer.apply(lambda x, context: x + context**2, grad_context)

        model.params.transform(grads, self.eta)

        learn_info["l0"] = l0


@optimizator_registry.register('Adadelta')
class Adadelta(Optimizator):
    def __init__(self, damp, eps=1e-8) -> None:
        self.eps = eps
        self.damp = damp
        self._cumulative_grad = None
        self._delta_grad = None

    def fit_transform(self, model, x, y, learn_info, n_jobs=1):
        if self._cumulative_grad is None:
            self._cumulative_grad = [layer.blank()
                                     for layer in reversed(model.params.layers)]
            self._delta_grad = [layer.blank()
                                for layer in reversed(model.params.layers)]

        l0, l1 = model.ask_oracle(x, y, n_jobs=n_jobs)
        grads = []

        for grad_context, cumulative_layer, delta_layer in zip(l1, self._cumulative_grad, self._delta_grad):
            grads.append(cumulative_layer.context_binary_operation(
                delta_layer.context_binary_operation(
                    delta_layer.get_context(),
                    grad_context,
                    lambda lhs, rhs: np.sqrt(lhs + self.eps) * rhs
                ),
                cumulative_layer.get_context(),
                lambda lhs, rhs: (lhs / (np.sqrt(rhs) + self.eps))
            ))

        for grad_context, cumulative_layer, delta_layer, delta_grads in zip(l1, self._cumulative_grad, self._delta_grad, grads):
            delta_layer.apply(lambda x, context: (
                self.damp * x) + (1 - self.damp) * (context ** 2), delta_grads)
            cumulative_layer.apply(lambda x, context: (
                self.damp * x) + (1 - self.damp) * (context ** 2), grad_context)

        model.params.transform(grads, self.eta)

        learn_info["l0"] = l0
