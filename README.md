# Fast Federated Learning in the Presence of Arbitrary Device Unavailability

This repository is the official implementation of [Fast Federated Learning in the Presence of Arbitrary Device Unavailability](https://arxiv.org/abs/2106.04159). 

We study federated learning algorithms under arbitrary device unavailability and show our proposed MIFA avoids excessive latency induced by inactive devices and achieves minimax optimal convergence rates.

Our code is adapted from the [code](https://github.com/lx10077/fedavgpy) for paper [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/abs/1907.02189).

## Data Preparation

To generate non-iid data with with each device holding samples of only two classes, run this command.

```shell
python data/generate_equal.py
```

To use Dirichlet allocation for data partitioning, run this command:

```shell
python data/gen_dirich.py
```

Modify the variable \$DIRICHLET_PARAMETER in ```data/gen_dirich.py``` to specify the concentration parameter $\alpha$.

## Training

To train the $\ell_2$ regularized logistic regression model on MNIST (each device holding samples of two classes), run this command:

```shell
bash run_mnist_logit.sh
```

To train the LeNet model on CIFAR10  (each device holding samples of two classes), run this command:

```shell
bash run_cifar_cnn.sh
```

To train the LeNet model on CIFAR10 partitioned using Dirichlet allocation, run this command:

```bash
bash run_dirich_cifar_cnn.sh
```

You can modify the shell script to change the experiment setup. The meaning of each variable is listed as follows: 

- \$NUM_USER: the total number of devices
- \$S is the number of participating devices each round for FedAvg with device sampling.  
- \$T: the total number of communication rounds
- \$K: the number of local epochs
- \$B: batch size
- \$device: GPU id
- \$SEED: random seed
- \$NET: the model, should be set as "logistic" or "cnn"
- \$WD: weight decay
- \$PARTICIPATION: the minimum participation rate is 0.1*\$PARTICIPATION, should be set as 1-9
- $ALGO: the algorithm
- \$RESULT_DIR: the directory for experiment logs

During training, the logs will be saved under the directory specified by the user. For each run, the folder is named as the hash of the starting time. Each folder contains two files, i.e. ```options.json``` and ```log.json```. The former records the experiment setup and the latter records the training loss, training accuracy, test loss and test accuracy. 

## Visualization

To visualize the training curves, run this command: 

```shell
python plot.py $LOG_DIR $PLOT_OPTION $DATASET
```

The usage of variables is listed as follows: 

- \$LOG_DIR: the directory for experiment logs
- \$PLOT_OPTION: should be in $\{0, 1, 2, 3\}$, corresponding to training loss, training accuracy, test loss and test_accuracy.
- $DATASET: should be 'cifar' or 'mnist'.

Example:

```
python plot.py results_cifar_cnn 3 cifar
```

## Results

The results are listed in Section 7 of the main text and G of the appendix in  this paper [Fast Federated Learning in the Presence of Arbitrary Device Unavailability](https://arxiv.org/abs/2106.04159). 

## Dependency

Pytorch = 1.0.0

numpy = 1.16.3

matplotlib = 3.0.0