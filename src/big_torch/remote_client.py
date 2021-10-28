from .preprocessing.cross_validation import train_test_split, one_hot_encoded

from .train.callbacks import callback_registry
from .preprocessing.scalers import scaler_registry
from .layers.abstract import layer_registry

from .train import OptimizatonFabric
from .models import Model

import numpy as np
import json


class RemoteClient():
    def __init__(self, cfg_file) -> None:
        with open(cfg_file, 'r+') as reader:
            self.cfg = json.load(reader)
        self._context = {}

    def preprocessing_stage(self):
        preprocessing_cfg = self.cfg['preprocessing']
        from mnist import MNIST
        mndata = MNIST(preprocessing_cfg['path'])

        x, y = mndata.load_training()
        x = np.array(x)
        y = np.array(y)

        if preprocessing_cfg['scaler'] != None:
            x = scaler_registry[preprocessing_cfg['scaler']]().fit_transform(x)

        y = one_hot_encoded(y, 10)

        t_x, t_y, v_x, v_y = train_test_split(
            x, y, test_size=preprocessing_cfg['test_size'])

        self._context['train_x'] = t_x
        self._context['train_y'] = t_y
        self._context['val_x'] = v_x
        self._context['val_y'] = v_y

    def compile_model(self):
        model_cfg = self.cfg['model']

        net = Model()
        self._context['net'] = net

        for layer in model_cfg['layers']:
            net.add_layer(layer_registry[layer['type']](**layer['cfg']))

        loss_layer = model_cfg['loss']
        net.set_loss(layer_registry[loss_layer['type']](**loss_layer['cfg']))

        net.build()

    def instantiate_callback(self, callback_cfg):
        callback = callback_registry[callback_cfg['type']](
            **callback_cfg['cfg'])
        return callback

    def build_session_stage(self):
        session_cfg = self.cfg['run_session']['fabric']

        kwargs = {
            'generator': session_cfg['generator'],
            'optimizer': session_cfg['optimizer'],
            'generator_cfg': session_cfg['generator_cfg'],
            'optimizer_cfg': session_cfg['optimizer_cfg'],
            'callbacks': [self.instantiate_callback(callback_cfg) for callback_cfg in session_cfg['callbacks']],
        }

        self._context['fabric'] = OptimizatonFabric(**kwargs)

    def run_session_stage(self):
        model = self._context['net']
        model, train_info = self._context['fabric'].train(
            model,
            self._context['train_x'],
            self._context['train_y'],
            x_val=self._context['val_x'],
            y_val=self._context['val_y'],
            **self.cfg['run_session']['train_kwargs']
        )

    def run(self):
        self.preprocessing_stage()
        self.compile_model()

        self.build_session_stage()
        return self.run_session_stage()
