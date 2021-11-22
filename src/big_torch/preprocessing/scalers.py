import numpy as np
from ..utils.registry import ModuleAggregator

scaler_registry = ModuleAggregator(registry_name="scalers")


@scaler_registry.register("min-max")
class MinMaxScaler:
    def __init__(self) -> None:
        self.min = None
        self.max = None

    def fit(self, x, axis=None):
        if axis is None:
            self.min = x.min()
            self.max = x.max()
        else:
            raise NotImplementedError(
                "Scaler is currently does not support axis wise scaling"
            )

    def fit_transform(self, x, axis=None, dtype=np.float64):
        if axis is None:
            self.fit(x, axis=axis)
            x = x.copy().astype(dtype)
            x -= self.min
            x /= self.max - self.min
            return x
        else:
            raise NotImplementedError(
                "Scaler is currently does not support axis wise scaling"
            )


@scaler_registry.register("standart")
class StandartScaler:
    def __init__(self) -> None:
        self.std = None
        self.mean = None

    def fit(self, x, axis=None):
        if axis is None:
            self.std = x.std()
            self.mean = x.mean()
        else:
            raise NotImplementedError(
                "Scaler is currently does not support axis wise scaling"
            )

    def fit_transform(self, x, axis=None, dtype=np.float64):
        if axis is None:
            self.fit(x, axis=axis)
            x = x.copy().astype(dtype)
            x -= self.mean
            x /= self.std
            return x
        else:
            raise NotImplementedError(
                "Scaler is currently does not support axis wise scaling"
            )
