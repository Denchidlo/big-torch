from ..preprocessing.cross_validation import metric_registry
from ..utils.registry import ModuleAggregator

callback_registry = ModuleAggregator("callbacks")


class TrainCallback:
    def call(self, learning_info):
        raise NotImplementedError()


@callback_registry.register("epoch_info")
class EpochInfo(TrainCallback):
    def __init__(self, metrics={}, validate=False, period=1) -> None:
        self.metric_names = metrics.keys()
        self.metrics = [metric_registry[metric] for metric in metrics.values()]
        self.period = period
        self.validate = validate
        self.init_meta = False

    def call(self, learning_info):
        if self.init_meta == False:
            learning_info["epoch_info"] = {
                "train_loss": [],
                "train_metrics": {name: [] for name in self.metric_names},
                "val_loss": [],
                "val_metrics": {name: [] for name in self.metric_names},
                "period": self.period,
            }
            self.init_meta = True

        if learning_info["epoch"] % self.period == 1 or self.period == 1:
            model = learning_info["model"]
            epoch = learning_info["epoch"]
            v_loss, v_metric = None, None

            t_loss, t_metric = model.eval(
                learning_info["x_train"], learning_info["y_train"], self.metrics
            )

            if self.validate:
                v_loss, v_metric = model.eval(
                    learning_info["x_val"], learning_info["y_val"], self.metrics
                )

            learning_info["epoch_info"]["train_loss"].append(t_loss)
            learning_info["epoch_info"]["val_loss"].append(v_loss)

            for idx, name in enumerate(self.metric_names):
                learning_info["epoch_info"]["train_metrics"][name].append(t_metric[idx])
                learning_info["epoch_info"]["val_metrics"][name].append(v_metric[idx])

            if learning_info["verbose"] > 0:
                print(
                    f"Epoch {epoch}: [Train] loss={t_loss} | metric={t_metric} [Test] loss={v_loss} | metric={v_metric}"
                )
