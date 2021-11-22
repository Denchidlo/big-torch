import numpy as np


def mask_all_equal_value(value=np.NaN, axis=2):
    def _inner(tensor):
        return np.any(tensor != value, axis=axis, keepdims=True)

    return _inner
