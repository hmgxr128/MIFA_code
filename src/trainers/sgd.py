from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
import torch


class SGD(BaseTrainer):
    def __init__(self, options, dataset, result_dir='results'):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        self.importance_sampling = options['importance_sampling']
        super(SGD, self).__init__(options, dataset, model, self.optimizer, result_dir)

    def train(self):

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):

            print("round", round_i)
            
            selected_clients = self.get_avail_clients(seed=round_i)
        
            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)

            # Update latest model
            if self.importance_sampling:
                self.latest_model = self.aggregate_is(selected_clients ,solns)
            else:
                self.latest_model = self.aggregate_normal(selected_clients ,solns)
            self.optimizer.inverse_prop_decay_learning_rate(round_i + 1)
            
            train_loss, train_acc = self.evaluate_train()
            test_loss, test_acc = self.evaluate_test()
            out_dict = {'train_loss': train_loss, 'train_acc':train_acc,'test_loss': test_loss, 'test_acc':test_acc}
            print("training loss & acc",train_loss, train_acc )
            print("test loss & acc", test_loss, test_acc)
            self.logger.log(round_i ,out_dict)
            self.logger.dump()

    def aggregate_is(self, clients, solns):
        """
            Importance Sampling scheme.
            Assume that each client returns w_i, its available probabilty = p_i 
                New model = w_0 +  1/N  \sum_{i=1}^N [ w_i  - w_0 ] / p_i * I{Available}
                        = w_0 +  1/N  \sum_{i available} [ w_i  - w_0 ] / p_i 
        """
        averaged_solution = torch.zeros_like(self.latest_model)
        for c, (_, weights) in zip(clients, solns):
            p = c.available_probability
            local_update = weights - self.latest_model
            averaged_solution += local_update / p
        averaged_solution = self.latest_model + averaged_solution/len(self.clients)
        return averaged_solution.detach()

    def aggregate_normal(self, clients, solns):
        """
            Normal scheme. ignores the available probabilty.
            Assume that there is S clients and each client returns w_i, 
                New model =   \sum_{i available} [ w_i ] / S
        """
        averaged_solution = torch.zeros_like(self.latest_model)
        for _, local_solution in solns:
            averaged_solution += local_solution
        averaged_solution /= len(solns)
        return averaged_solution.detach()