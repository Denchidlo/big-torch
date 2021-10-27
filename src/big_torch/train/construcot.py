from ..core.utils import ModuleAggregator
from ..models.model import open_pool_session, close_pool_session
from .frame_generators import BasicGenerator

opitmizer_registry = ModuleAggregator()


class OptimizatonFabric:
    def __init__(
        self,
        gen=BasicGenerator,
        updater=None,
        generator_cfg={},
        updater_cfg={},
        callbacks=[],
    ) -> None:
        self.generator_cfg = generator_cfg
        self.updater_cfg = updater_cfg
        self.callbacks = callbacks
        self.gen = gen
        self.updater = updater

    def train(
        self,
        model,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        eps=1e-5,
        max_iter=None,
        n_jobs=1,
    ):
        _eps = 10 * eps
        max_iter = abs(max_iter)
        n_iter = 0
        prev_l0 = None

        frame_generator = self.gen(x_train, y_train, **self.generator_cfg)
        optimizer = self.updater(**self.updater_cfg)

        if n_jobs != 1:
            open_pool_session(n_jobs)

        learning_information = {
            "x_val": x_val,
            "y_val": y_val,
            "model": model,
        }

        while True:
            n_iter += 1
            x_frame, y_frame = next(frame_generator)

            optimizer.fit_transform(
                model, x_frame, y_frame, learning_information, n_jobs
            )

            l0 = learning_information['l0']

            print(f"Epoch {n_iter} - loss: {l0}")

            _eps = abs(prev_l0 - l0) if prev_l0 != None else 2 * eps
            prev_l0 = l0

            if (_eps < eps) or (n_iter > max_iter and max_iter != None):
                break

        if n_jobs != 1:
            close_pool_session()

        return model
