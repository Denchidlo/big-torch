class TrainCallback:
    def call(self, learning_info):
        raise NotImplementedError()


class EpochInfo(TrainCallback):
    def __init__(self, metric=None, validate=False, period=100) -> None:
        self.metric = metric
        self.validate = validate
        self.period = period
        self.init_meta = False

    def call(self, learning_info):
        if self.init_meta == False:
            learning_info['epoch_info'] = {
                'train_loss': [],
                'train_metrics': [],
                'val_loss': [],
                'val_metrics': [],
                'period': self.period
            }
            self.init_meta = True
        
        if learning_info['epoch'] % self.period == 1:
            model = learning_info['model']
            epoch = learning_info['epoch']
            v_loss, v_metric = None, None

            t_loss, t_metric = model.eval(learning_info['x_train'], learning_info['y_train'], self.metric)

            if self.validate:
                v_loss, v_metric = model.eval(learning_info['x_val'], learning_info['y_val'], self.metric)

            learning_info['epoch_info']['train_loss'].append(t_loss)
            learning_info['epoch_info']['train_metrics'].append(t_metric)
            learning_info['epoch_info']['val_loss'].append(v_loss)
            learning_info['epoch_info']['val_metrics'].append(v_metric)

            if learning_info['verbose'] > 0:
                print(f'Epoch {epoch}: [Train] loss={t_loss} | metric={t_metric} [Test] loss={v_loss} | metric={v_metric}')