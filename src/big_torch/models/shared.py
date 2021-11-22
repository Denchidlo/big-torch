import multiprocessing as mp

POOL = None


def open_pool_session(n_jobs):
    global POOL
    POOL = mp.Pool(n_jobs).__enter__()


def close_pool_session():
    global POOL
    POOL.__exit__(None, None, None)
    POOL = None


class BasicModelParams:
    def __init__(self, layers) -> None:
        self.layers = layers

    def transform(self, gradients, eta):
        for idx, layer in enumerate(reversed(self.layers)):
            layer.change(gradients[idx], eta)

    def _avg_grads(self, gradients_list):
        resulting = []

        for idx, layer in enumerate(reversed(self.layers)):
            resulting.append(layer.average([el[idx] for el in gradients_list]))

        return resulting
