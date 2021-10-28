from big_torch.core.utils import ModuleAggregator

# Registry organazing
from .layers.abstract import layer_registry
from .preprocessing.cross_validation import metric_registry
from .preprocessing.initializers import initializer_registry
from .preprocessing.scalers import scaler_registry
from .train.frame_generators import generator_registry
from .train.optimizers import optimizator_registry

registry = ModuleAggregator('big-torch_registry')
registry.register('layers')(layer_registry)
registry.register('metrics')(metric_registry)
registry.register('scalers')(scaler_registry)
registry.register('generators')(generator_registry)
registry.register('optimizators')(optimizator_registry)
registry.register('initializers')(initializer_registry)
