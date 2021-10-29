from big_torch.remote_client import RemoteClient
import neptune.new as neptune
import pickle
import json


def to_pickle(obj, path):
    with open(path, 'wb') as writer:
        pickle.dump(obj, writer)

def to_json(obj, path):
    with open(path, 'wb') as writer:
        json.dump(obj, writer)

class NeptuneClient():
    def __init__(self, credentials) -> None:
        self.credentials = credentials
        
    def process_experiment(self, model_cfg):
        with open(model_cfg, 'r+') as reader:
            cfg = json.load(reader)

        print('Model is started')

        client = RemoteClient(model_cfg)
        model, learning_info = client.run()

        print('Model was succesfully finished')

        run = neptune.init(
            run=cfg['neptune_meta']['run'] if cfg['neptune_meta'].get('run', None) != None else None,
            tags=cfg['neptune_meta']['tags'] if cfg['neptune_meta'].get('tags', None) != None else None,
            **self.credentials
        )
        print('Initialize neptune session')

        run['model/config'] = cfg['model']
        run['model/layers'] = cfg['model']['layers']
        run['model/loss/type'] = cfg['model']['loss']['type']
        run['model/loss/config'] = cfg['model']['loss']['cfg']
        run['model/frame_generator/type'] = cfg['run_session']['fabric']['generator']
        run['model/frame_generator/config'] = cfg['run_session']['fabric']['generator_cfg']
        run['model/optimizer/type'] = cfg['run_session']['fabric']['optimizer']
        run['model/optimizer/config'] = cfg['run_session']['fabric']['optimizer_cfg']

        run['session/args'] = cfg['run_session']['train_kwargs']
        

        del learning_info['model']
        del learning_info['x_val']
        del learning_info['y_val']
        del learning_info['x_train']
        del learning_info['y_train']

        epoch_info = learning_info['epoch_info']
        epochs_total = len(epoch_info['train_loss'])

        mapper = {
            'train_loss': 'metrics/train/loss',
            'val_loss': 'metrics/test/loss',
            'train_metrics': 'metrics/train/accuracy',
            'val_metrics': 'metrics/test/accuracy',
        }

        valid_metrics = []
        for metric in mapper.keys():
            if epoch_info[metric][0] != None:
                valid_metrics.append(metric)

        period = epoch_info['period']

        for metric, neptune_key in mapper.items():
            series = epoch_info[metric]
            
            for idx in range(epochs_total):
                run[neptune_key].log(series[idx], step=(idx*period + 1))

        run['dump/experiment/config'].upload_files(model_cfg)
        
        to_pickle(model, './model_dump.pkl')
        run['dump/model'].upload_files('./model_dump.pkl')

        print('Neptune session finished')
        run.stop()


        