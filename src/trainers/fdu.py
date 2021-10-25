from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
import torch


class FDUTrainer(BaseTrainer):
    def __init__(self, options, dataset, result_dir='results'):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        
        self.update_table = []  # store previous updates

        super(FDUTrainer, self).__init__(options, dataset, model, self.optimizer, result_dir)

    def initialize_update_table(self):
        solns, _ = self.local_train(0, self.clients)
        self.update_table = [ (1/self.optimizer.get_current_lr()) * (solns[i][1] - self.latest_model) for i in range(len(self.clients))]


    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        table_initialized = False 

        truth_round = 0
        init_avail_device = set()

        for round_i in range(self.num_round):
            print("round", round_i)
            selected_clients = self.get_avail_clients(seed=round_i)

            if table_initialized:
                # Solve minimization locally
                solns, stats = self.local_train(truth_round, selected_clients)

                # Update table
                for (idx, c) in enumerate(selected_clients):
                    self.update_table[c.cid] = 1/self.optimizer.get_current_lr() * (solns[idx][1] - self.latest_model)

                self.aggregate()
                self.optimizer.inverse_prop_decay_learning_rate(truth_round + 1)
                
                train_loss, train_acc = self.evaluate_train()
                test_loss, test_acc = self.evaluate_test()
                out_dict = {'train_loss': train_loss, 'train_acc':train_acc,'test_loss': test_loss, 'test_acc':test_acc}
                print("training loss & acc",train_loss, train_acc )
                print("test loss & acc", test_loss, test_acc)
                self.logger.log(round_i, out_dict)
                self.logger.dump()
                truth_round += 1
            else:
                init_avail_device = init_avail_device.union(set([c.cid for c in selected_clients]))
                if len(init_avail_device) == len(self.clients):
                    table_initialized = True
                    self.initialize_update_table()

    def aggregate(self):
        self.latest_model = self.latest_model + self.optimizer.get_current_lr() * sum(self.update_table) / len(self.clients)
