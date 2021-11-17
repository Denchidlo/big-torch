import numpy as np
from ..utils.registry import ModuleAggregator

metric_registry = ModuleAggregator('metrics')


def one_hot_encoded(y, num_class):
    n = y.shape[0]
    onehot = np.zeros((n, num_class), dtype="int32")
    for i in range(n):
        idx = y[i]
        onehot[i][idx] = 1
    return onehot


def train_test_split(a, b, test_size):
    size = a.shape[0]

    assert a.shape[0] == b.shape[0], 'Size of left and right hand indicies should be equal'
    assert size > test_size, 'test_size out of range'

    choice = np.random.default_rng().choice(size, test_size, replace=False)
    mask = np.array([i for i in range(size)])
    mask = np.isin(mask, choice)
    return a[mask == 0], b[mask == 0], a[mask == 1], b[mask == 1]


@metric_registry.register('accuracy')
def class_accuracy(y_true, y_pred):
    # both are one hot encoded
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
