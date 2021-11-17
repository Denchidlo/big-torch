from typing import Iterable
from ..core.computational_graph import LayerComponent

class DependecyCallMixin():
    """
        DependencyCallMixin:

        This class provides you an easy way to add dependencies to model layer
        Note that dependencies are provided and form in this order X in _fwd_pass

        Example:
        >>> activation_layer = SomeActivationLayerClass(args, kwargs)
        >>> activation_component = activation_layer([dependency_1, dependency_2])
    """

    def __call__(self, dependencies):
        if getattr(self, '_component', None) == None:
            self._component = LayerComponent(self)
            self._component.subscribe(dependencies)
            return self._component
        else:
            raise RuntimeError('Attempt to instantiate component twice')
