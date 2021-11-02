from ..models.model import open_pool_session, close_pool_session
from .optimizers import optimizator_registry
from .frame_generators import BasicGenerator, generator_registry


class OptimizatonFabric:
    def __init__(
        self,
        generator=BasicGenerator,
        optimizer=None,
        generator_cfg={},
        optimizer_cfg={},
        callbacks=[],
    ) -> None:
        self.generator_cfg = generator_cfg
        self.optimizer_cfg = optimizer_cfg
        self.callbacks = callbacks
        self.generator = generator_registry[generator]
        self.optimizer = optimizator_registry[optimizer]

    def train(
        self,
        model,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        min_eps=1e-5,
        max_iter=None,
        n_jobs=1,
        verbose=1,
    ):
        eps = 10 * min_eps
        max_iter = abs(max_iter)
        epoch = 0
        prev_l0 = None

        # TODO: Add aggregator
        frame_generator = self.generator(
            x_train, y_train, **self.generator_cfg)
        optimizer = self.optimizer(**self.optimizer_cfg)

        if n_jobs != 1:
            open_pool_session(n_jobs)

        learning_information = {
            'x_val': x_val,
            'y_val': y_val,
            'model': model,
            'verbose': verbose,
            'callbacks': self.callbacks
        }

        for x_frame, y_frame in frame_generator:
            epoch += 1

            optimizer.fit_transform(
                model, x_frame, y_frame, learning_information, n_jobs
            )

            learning_information['epoch'] = epoch
            learning_information['x_train'] = x_frame
            learning_information['y_train'] = y_frame
            l0 = learning_information['l0']

            for callback in self.callbacks:
                callback.call(learning_information)

            eps = abs(prev_l0 - l0) if prev_l0 != None else 2 * min_eps
            prev_l0 = l0

            if (eps < min_eps) or (epoch > max_iter and max_iter != None):
                break

        if n_jobs != 1:
            close_pool_session()

        return model, learning_information
