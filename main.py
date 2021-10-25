import numpy as np
import argparse
import importlib
import torch
import os

from src.utils.worker_utils import read_data
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS


def read_options():
    parser = argparse.ArgumentParser()


    # General setting
    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=64)
    parser.add_argument('--local_step',
                        help='number of steps when clients train on data;',
                        type=int,
                        default=5)
    parser.add_argument('--participation_level',
                        help='lower bound for the participation probabilities. (e.g., participation_level = 4 means the participation probabilities of the devices are in [0.4, 1]). ',
                        type=int,
                        default=1)
    parser.add_argument('--participation_pattern',
                        help='participation probabilities pattern: random or adversarial',
                        type=str,
                        choices=['random','adversarial'],
                        default='random')
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.1)

    # Algorithm Speicific setting
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round (only for FedAvg)',
                        type=int,
                        default=10)
    parser.add_argument('--importance_sampling',
                        action='store_true',
                        default=False,
                        help='whether to perform importance sampling (only for SGD) (default: False)')


    # dataset and models
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_0_equal_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='logistic')
    
    # other training settings
    parser.add_argument('--gpu',
                        action='store_true',
                        default=False,
                        help='use gpu (default: False)')
    parser.add_argument('--noprint',
                        action='store_true',
                        default=False,
                        help='whether to print inner result (default: False)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--num_user',
                        help='number of users',
                        default = '',
                        type=str)
    parser.add_argument('--dirichlet',
                    help='dirichlet parameter',
                    default = '',
                    type=str)

    parser.add_argument('--result_dir',
                        help='result dir',
                        type=str,
                        default='result')
    
                        
    parsed = parser.parse_args()
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()

    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])

    # read data
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # Add model arguments
    options.update(MODEL_PARAMS(dataset_name, options['model']))

    # Load selected trainer
    trainer_path = 'src.trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, trainer_class, dataset_name, sub_data


def main():
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data = read_options()

    train_path = os.path.join('./data', dataset_name + options['dirichlet'], 'data', 'train')
    test_path = os.path.join('./data', dataset_name  + options['dirichlet'], 'data', 'test')
    avail_prob_file  = os.path.join('./data', dataset_name + options['dirichlet'], 'data' ,'avail_prob_{}_{}.pkl'.format(options['participation_level'], options['participation_pattern']))

    # `dataset` is a tuple like (cids, groups, train_data, test_data)
    all_data_info = read_data(train_path, test_path, avail_prob_file, sub_data)

    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info, options['result_dir'])
    trainer.train()


if __name__ == '__main__':
    main()
