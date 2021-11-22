import numpy as np

from .abstract import ParametrizedObject, GradientInputResolver


class DependecyCallMixin:
    """
    DependencyCallMixin:

    This class provides you an easy way to add dependencies to model layer
    Note that dependencies are provided in order they're sent to _fwd_pass

    Example:
    >>> activation_layer = SomeActivationLayerClass(args, kwargs)
    >>> activation_component = activation_layer([dependency_1, dependency_2])
    """

    def __call__(self, dependencies):
        if getattr(self, "_component", None) == None:
            from ..core.computational_graph import LayerComponent

            self._component = LayerComponent(self)
            self._component.subscribe(dependencies)
            return self._component
        else:
            raise RuntimeError("Attempt to instantiate component twice")


class ParamManagerMixin(ParametrizedObject):
    """
    ParamManagerMixin:

    This mixin extends your complex, non-atomic layers and components and allows
    you to easily manage your params.

    Class interface:

        Note:
            If you use it you should take into account that now your object
            or layer is treated as an atomic ParametrizedObject

        Requirements:
            object._parametrized_layers -> list of ParametrizedObjects
            (see big_torch/layers/abstract)

        Function result form:
            Iterable[layer_context| layers] | None
    """

    def get_context(self):
        return [layer.get_context() for layer in reversed(self._parametrized_layers)]

    def apply(self, func, context=None):
        for idx, layer in reversed(self._parametrized_layers):
            layer.apply(func, context=context[idx])

    def blank(self):
        return [layer.blank() for layer in reversed(self._parametrized_layers)]

    def average(self, gradients_list):
        return [
            layer.average([sublist[idx] for sublist in gradients_list])
            for idx, layer in enumerate(reversed(self._parametrized_layers))
        ]

    def change(self, step, eta):
        return [
            layer.change(substep, eta)
            for layer, substep in zip(reversed(self._parametrized_layers), step)
        ]

    def context_binary_operation(self, lhs, rhs, operation):
        return [
            layer.context_binary_operation(lhs_item, rhs_item, operation)
            for lhs_item, rhs_item, layer in zip(
                lhs, rhs, reversed(self._parametrized_layers)
            )
        ]


class GradientStacker1DMixin(GradientInputResolver):
    def handle_multiple_inputs(self, gradients):
        return np.sum(gradients, axis=0)

    def get_handler(self, output_idx):
        if output_idx > len(self.output_names):
            raise IndexError("Output index is out of layer bounadries")
        return self.handle_multiple_inputs
