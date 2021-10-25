# GLOBAL PARAMETERS
DATASETS = ['mnist', 'cifar']
TRAINERS = {'fedavg': 'FedAvgTrainer', 
            'fdu': 'FDUTrainer',
            'fdu_no_wait': 'FDU_nowait_Trainer',
            'sgd': 'SGD'
}
OPTIMIZERS = TRAINERS.keys()


class ModelConfig(object):
    def __init__(self):
        pass

    def __call__(self, dataset, model):
        dataset = dataset.split('_')[0]
        if dataset == 'mnist':
            return {'input_shape': (1, 28, 28), 'num_class': 10}
        elif dataset == 'cifar':
            return {'input_shape': (3, 32, 32), 'num_class': 10}
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))


MODEL_PARAMS = ModelConfig()