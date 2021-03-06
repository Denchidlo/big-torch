from ..utils.registry import ModuleAggregator
from typing import Any
import numpy as np
from numpy.random.mtrand import choice


generator_registry = ModuleAggregator("frame_generators")


class FrameGenerator:
    """
    Train DataFrame Generator:

    Usage:

        >>> frame_generator = FrameGenerator(train_x, train_y)

        >>> x_batch, y_batch = next(frame_generator)
    """

    def __init__(self, train_x, train_y) -> None:
        self.train_x = train_x
        self.train_y = train_y

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError()


@generator_registry.register("basic")
class BasicGenerator(FrameGenerator):
    def __init__(self, train_x, train_y) -> None:
        super().__init__(train_x, train_y)

    def __next__(self):
        return self.train_x, self.train_y


@generator_registry.register("random_batch")
class StochasticBatchGenerator(FrameGenerator):
    def __init__(self, train_x, train_y, batch_size) -> None:
        super().__init__(train_x, train_y)
        self.generator = np.random.default_rng()
        self.total = train_x.shape[0]
        self.batch_size = batch_size

    def __next__(self):
        choice = self.generator.choice(self.total, self.batch_size, replace=False)

        return self.train_x[choice, :], self.train_y[choice, :]


@generator_registry.register("random_element")
class RandomElementGenerator(FrameGenerator):
    def __init__(self, train_x, train_y) -> None:
        super().__init__(train_x, train_y)
        self.total = train_x.shape[0]

    def __next__(self):
        choice = np.random.randint(0, self.total)
        return np.array([self.train_x[choice]]), np.array([self.train_y[choice]])
