import torch
import numpy as np
import pickle
import os
import torchvision

cpath = os.path.dirname(os.path.abspath(__file__))
from gen_avail_prob import gen_avail_prob_adversarial_dirichlet

NUM_USER = 100
SAVE = True
DIRICHLET_PARAMETER = 0.1
NUM_TRAIN = 50000
NUM_TEST = 10000
NUM_CLASSES = 10


np.random.seed(16)

class ImageDataset(object):
    def __init__(self, images, labels):
        if isinstance(images, torch.Tensor):
            self.data = images.numpy()
        else:
            self.data = images
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels
        self.label_count = []

    def __len__(self):
        return len(self.target)
    
    def Shuffle(self):
        train_example_indices = []
        for k in range(NUM_CLASSES):
            # Select all indices where the train label is k
            train_label_k = np.where(self.target == k)[0]
            np.random.shuffle(train_label_k)  
            train_example_indices.append(train_label_k)
            self.label_count.append(train_label_k.shape[0])
        print(self.label_count)
        return train_example_indices




def Gen_dirichlet():
    train_multinomial_vals = []
    for i in range(NUM_USER):
        proportion = np.random.dirichlet(DIRICHLET_PARAMETER  * np.ones(10,))
        train_multinomial_vals.append(proportion)
    train_multinomial_vals =  np.array(train_multinomial_vals)
    return train_multinomial_vals


def main(dataset):
    assert dataset in ['cifar', 'mnist'], 'dataset not supported!'
    dataset_dir = os.path.join(cpath, dataset)
    if dataset == 'mnist':
        # Get MNIST data, normalize, and divide by level
        print('>>> Get MNIST data.')
        trainset = torchvision.datasets.MNIST(dataset_dir, download=True, train=True)
        testset = torchvision.datasets.MNIST(dataset_dir, download=True, train=False)

        train_d = ImageDataset(trainset.data, trainset.targets)
        test_d = ImageDataset(testset.data, testset.targets)
    if dataset == 'cifar':
        # Get CIFAR data, normalize, and divide by level
        print('>>> Get CIFAR10 data.')
        trainset = torchvision.datasets.CIFAR10(dataset_dir, download=False, train=True)
        testset = torchvision.datasets.CIFAR10(dataset_dir, download=False, train=False)

        train_d = ImageDataset(trainset.data, trainset.targets)
        test_d = ImageDataset(testset.data, testset.targets)
    
    train_example_indices = train_d.Shuffle()
    test_example_indices = test_d.Shuffle()

    train_count = np.zeros(10).astype(int)
    total_count_train = 0


    train_multinomial_vals = Gen_dirichlet()
    test_multinomial_vals = np.copy(train_multinomial_vals)
    train_examples_per_user = int(NUM_TRAIN / NUM_USER)
    test_examples_per_user = int(NUM_TEST / NUM_USER)

    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    class_info = []

    for user in range(NUM_USER):
        user_class_count = np.zeros(NUM_CLASSES).astype(int)
        for i in range(train_examples_per_user):
            # count the number of training samples for each class
            sampled_label = np.argwhere(np.random.multinomial(1, train_multinomial_vals[user]) == 1)[0][0]
            current_sample = train_example_indices[sampled_label][train_count[sampled_label]]
            train_X[user].append(train_d.data[current_sample])
            train_y[user].append(sampled_label)
            train_count[sampled_label] += 1
            user_class_count[sampled_label] += 1
            total_count_train += 1
            if train_count[sampled_label] == train_d.label_count[sampled_label] and total_count_train < NUM_TRAIN:
                train_multinomial_vals[:, sampled_label] = 0
                train_multinomial_vals = train_multinomial_vals / np.sum(train_multinomial_vals, axis = 1)[:, None]
        print("the number of training samples held by user {}: {}".format(user, np.sum(user_class_count)))
        # return the class that the client has the most data samples
        class_info.append(np.argmax(user_class_count))

        total_count_test = 0
        test_count = np.zeros(NUM_CLASSES).astype(int)

    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]



    for user in range(NUM_USER):
        user_class_count = np.zeros(NUM_CLASSES).astype(int)
        for i in range(test_examples_per_user):
            # count the number of test samples for each class
            sampled_label = np.argwhere(np.random.multinomial(1, test_multinomial_vals[user]) == 1)[0][0]
            current_sample = test_example_indices[sampled_label][test_count[sampled_label]]
            test_X[user].append(test_d.data[current_sample])
            test_y[user].append(sampled_label)
            test_count[sampled_label] += 1
            user_class_count[sampled_label] += 1
            total_count_test += 1
            if test_count[sampled_label] == test_d.label_count[sampled_label] and total_count_test < NUM_TEST:
                test_multinomial_vals[:, sampled_label] = 0
                test_multinomial_vals = test_multinomial_vals / np.sum(test_multinomial_vals, axis = 1)[:, None]
        print("the number of test samples held by user {}: {}".format(user, np.sum(user_class_count)))


    # Setup directory for train/test data
    print('>>> Set data path for {}.'.format(dataset))
    train_path = '{}/{}/data/train/all.pkl'.format(cpath,dataset + 'dirichlet' + str(DIRICHLET_PARAMETER))
    test_path = '{}/{}/data/test/all.pkl'.format(cpath,dataset + 'dirichlet' + str(DIRICHLET_PARAMETER))
 
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

     # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}


    # Setup 100 users
    for i in range(NUM_USER):
        uname = i
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data
    if SAVE:
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile)

        print('>>> Save data.')

    return class_info


if __name__ == '__main__':
    for dataset in ['mnist', 'cifar']:
        class_info = main(dataset)
        for i in range(1, 10): # for minimum participation rate from 0.1 to 0.9
            gen_avail_prob_adversarial_dirichlet(list(range(NUM_USER)), \
            0.1 * i, 1, '{}/{}/data/avail_prob_{}_adversarial.pkl'.format(cpath,dataset + 'dirichlet' + str(DIRICHLET_PARAMETER), i), class_info)