from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
import torch


class FedAvgTrainer(BaseTrainer):
    def __init__(self, options, dataset, result_dir='results'):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        super(FedAvgTrainer, self).__init__(options, dataset, model, self.optimizer, result_dir)
        self.clients_per_round = min(options['clients_per_round'], len(self.clients))

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        
        true_round = 0

        avail_clients = set()
        selected_clients = np.random.choice(self.clients, self.clients_per_round, replace=False).tolist()
        set_selected_clients = set([c.cid for c in selected_clients])

        for round_i in range(self.num_round):

            print("round", round_i)
            
            new_clients = self.get_avail_clients(seed=round_i)
            avail_clients = avail_clients.union([c.cid for c in new_clients])
            if set_selected_clients.issubset(avail_clients):  # repeated query each device until devices in the selected subset are all available
                # Solve minimization locally
                solns, stats = self.local_train(true_round, selected_clients)

                # Update latest model
                self.latest_model = self.aggregate(solns)
                self.optimizer.inverse_prop_decay_learning_rate(true_round + 1)
                
                train_loss, train_acc = self.evaluate_train()
                test_loss, test_acc = self.evaluate_test()
                out_dict = {'train_loss': train_loss, 'train_acc':train_acc,'test_loss': test_loss, 'test_acc':test_acc}
                print("training loss & acc",train_loss, train_acc )
                print("test loss & acc", test_loss, test_acc)
                self.logger.log(round_i ,out_dict)
                self.logger.dump()
                true_round += 1

                avail_clients = set()
                selected_clients = np.random.choice(self.clients, self.clients_per_round, replace=False).tolist()
                set_selected_clients = set([c.cid for c in selected_clients])

    def aggregate(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """
        averaged_solution = torch.zeros_like(self.latest_model)
        for _, local_solution in solns:
            averaged_solution += local_solution
        averaged_solution /= len(solns)
        return averaged_solution.detach()