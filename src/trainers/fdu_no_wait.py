from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.optimizers.gd import GD
import torch


class FDU_nowait_Trainer(BaseTrainer):
    def __init__(self, options, dataset, result_dir='results'):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        
        self.update_table = []  # store previous updates

        super(FDU_nowait_Trainer, self).__init__(options, dataset, model, self.optimizer, result_dir)

    def initialize_update_table(self):
        self.update_table = [ torch.zeros_like(self.latest_model) for i in range(len(self.clients))]


    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        self.initialize_update_table()

        for round_i in range(self.num_round):
            print("round", round_i)
            selected_clients = self.get_avail_clients(seed=round_i)

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)

            # Update table
            for (idx, c) in enumerate(selected_clients):
                self.update_table[c.cid] = 1/self.optimizer.get_current_lr() * (solns[idx][1] - self.latest_model)

            self.aggregate()
            self.optimizer.inverse_prop_decay_learning_rate(round_i + 1)
            
            train_loss, train_acc = self.evaluate_train()
            test_loss, test_acc = self.evaluate_test()
            out_dict = {'train_loss': train_loss, 'train_acc':train_acc,'test_loss': test_loss, 'test_acc':test_acc}
            print("training loss & acc",train_loss, train_acc )
            print("test loss & acc", test_loss, test_acc)
            self.logger.log(round_i, out_dict)
            self.logger.dump()
            round_i += 1
            
                    

    def aggregate(self):
        self.latest_model = self.latest_model + self.optimizer.get_current_lr() * sum(self.update_table) / len(self.clients)
