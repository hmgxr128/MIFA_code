from src.utils.torch_utils import get_state_dict, get_flat_params_from, set_flat_params_to
import torch.nn as nn
import torch

criterion = nn.CrossEntropyLoss()

class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """
    def __init__(self, model, optimizer, options):
        # Basic parameters
        self.model = model
        self.optimizer = optimizer
        self.local_step = options['local_step']
        self.gpu = options['gpu'] if 'gpu' in options else False

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, _ in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def local_train(self, train_dataloader, **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """
        self.model.train()
        train_loss = train_acc = train_total = 0
        # local_counter = 0

        for _ in range(self.local_step): # train K epochs.
            for (x, y) in train_dataloader:

                # if local_counter >= self.local_step:
                #     break
                # local_counter+=1

                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                pred = self.model(x)

                loss = criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

            local_solution = self.get_flat_model_params()
            param_dict = {"norm": torch.norm(local_solution).item(),
                        "max": local_solution.max().item(),
                        "min": local_solution.min().item()}
            return_dict = {"loss": train_loss/train_total,
                        "acc": train_acc/train_total}
            return_dict.update(param_dict)
        return local_solution, return_dict

    def evaluate(self, dataloader, **kwargs):
        """
            Evaluate model on a dataset
        Args:
            dataloader: DataLoader class in Pytorch

        Returns
            Test loss and Test acc
        """
        self.model.eval()

        test_loss = test_acc = test_total = 0

        with torch.no_grad():
            for (x, y) in dataloader:
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                pred = self.model(x)
                loss = criterion(pred, y)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                test_loss += loss.item() * y.size(0)
                test_acc += correct
                test_total += target_size

        return_dict = {"num_samples": test_total,
                        "total_loss": test_loss,
                       "total_correct": test_acc}
        return return_dict