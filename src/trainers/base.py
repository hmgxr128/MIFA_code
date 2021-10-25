import numpy as np
import torch
import time
import json
from src.models.client import Client
from src.models.worker import Worker
import time
import os


class Logger():
    def __init__(self, file):
        self.file = file
        self.out_dict = {}

    def log(self, round_i ,new_dict):
        self.out_dict[round_i]= new_dict 

    def dump(self):
        with open(self.file,'w') as f:
            f.write(json.dumps(self.out_dict,indent=2))


class BaseTrainer(object):
    def __init__(self, options, dataset, model=None, optimizer=None, result_dir='results'):
        self.worker = Worker(model, optimizer, options)
        print('>>> Activate a worker for training')

        self.options = options
        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.all_train_data_num = 0
        self.clients = self.setup_clients(dataset)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round'] # total number of communication rounds
        self.clients_per_round = options['clients_per_round'] # useful for fedavg.

        # Initialize system metrics
        self.print_result = not options['noprint']
        self.latest_model = self.worker.get_flat_model_params()

        # logger 
        hash_tag = hash(time.time())
        hash_tag = str(hash_tag)
        hash_tag = os.path.join(result_dir, hash_tag)
        os.makedirs(hash_tag)
        self.logger = Logger(os.path.join(hash_tag,'log.json'))
        with open(os.path.join(hash_tag, 'options.json'), 'w') as f:
            json.dump(options, f, indent=2)


    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def setup_clients(self, dataset):
        """Instantiates clients based on given train and test data directories

        Returns:
            all_clients: List of clients
        """
        users, groups, train_data, test_data, avail_prob_dict = dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            c = Client(user_id, group, avail_prob_dict[user],train_data[user], test_data[user], self.batch_size, self.worker)
            all_clients.append(c)
        return all_clients

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def get_avail_clients(self, seed=1):
        """Selects num_clients clients weighted by number of samples from possible_clients

        Args:
            seed: random seed
            the availabity of clients is determined by another procedure. 
            
        Return:
            list of available clients.
        """
        np.random.seed(seed * (self.options['seed'] + 1))
        avail_client_list = []
        for c in self.clients:
            p = c.available_probability
            coin = np.random.rand()
            if coin < p:
                avail_client_list.append(c)
        return avail_client_list

    def local_train(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training, used for logging only
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_train()
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% |".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['loss'], stat['acc']*100, ))

            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def evaluate_train(self, **kwargs):
        return self.base_evaluate(eval_on_train=True)

    def evaluate_test(self, **kwargs):
        return self.base_evaluate(eval_on_train=False)

    def base_evaluate(self, eval_on_train ,**kwargs):
        """
            Evaluate results on training data/test data.
        """
        num_samples = 0
        total_loss = 0
        total_correct = 0
        for c in self.clients:
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Evaluate locally
            if eval_on_train:
                return_dict = c.evaluate_train(**kwargs)
            else:
                return_dict = c.evaluate_test(**kwargs)

            num_samples += return_dict["num_samples"]
            total_loss += return_dict["total_loss"]
            total_correct += return_dict["total_correct"]


        ave_loss = total_loss / num_samples
        acc = total_correct / num_samples

        return ave_loss, acc