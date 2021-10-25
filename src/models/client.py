import time
from torch.utils.data import DataLoader


class Client(object):
    """Base class for all local clients

    Outputs of gradients or local_solutions will be converted to np.array
    in order to save CUDA memory.
    """
    def __init__(self, cid, group, available_probability, train_data, test_data, batch_size, worker):
        self.cid = cid
        self.group = group
        self.worker = worker
        self.available_probability = available_probability

        self.train_data = train_data
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    def get_model_params(self):
        """Get model parameters"""
        return self.worker.get_model_params()

    def set_model_params(self, model_params_dict):
        """Set model parameters"""
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        self.worker.set_flat_model_params(flat_params)

    def local_train(self, **kwargs):
        """Solves local optimization problem

        Returns:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2. Statistic Dict contain
                2.1: bytes_write: number of bytes transmitted
                2.2: comp: number of FLOPs executed in training process
                2.3: bytes_read: number of bytes received
                2.4: other stats in train process
        """

        local_solution, worker_stats = self.worker.local_train(self.train_dataloader, **kwargs)

        stats = {'id': self.cid}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats
    
    def evaluate_train(self, **kwargs):
        return self.worker.evaluate(self.train_dataloader, **kwargs)

    def evaluate_test(self, **kwargs):
        return self.worker.evaluate(self.test_dataloader, **kwargs)
