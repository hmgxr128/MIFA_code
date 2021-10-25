# experiment set 
DATASET=cifar_all.pkl
DIRICHLET_PARAMETER=dirichlet0.15
NUM_USER=100
S=100
T=2000
K=2
B=100
device=4
SEED=2
NET=cnn
WD=0.001
PARTICIPATION=1
PATTERN=adversarial

RESULT_DIR=results_cifar_cnn_dirichlet_"$DIRICHLET_PARAMETER"

# our proposed algorithm
LR=0.1
S=100
ALGO=fdu
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD --dirichlet $DIRICHLET_PARAMETER --num_user $NUM_USER&


# sgd with Importance sampling
LR=0.1
ALGO=sgd
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD --importance_sampling --dirichlet $DIRICHLET_PARAMETER --num_user $NUM_USER &
sleep 1

# sgd without importance sampling  
LR=0.1
ALGO=sgd
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD --dirichlet $DIRICHLET_PARAMETER --num_user $NUM_USER &
sleep 1


# fedavg(100/100)
LR=0.1
ALGO=fedavg
S=100
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD --dirichlet $DIRICHLET_PARAMETER --num_user $NUM_USER&
sleep 1

# fedavg(50/100)
LR=0.09
ALGO=fedavg
S=50
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD --dirichlet $DIRICHLET_PARAMETER --num_user $NUM_USER &
sleep 1


# fedavg(25/100)
LR=0.11
ALGO=fedavg
S=25
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD --dirichlet $DIRICHLET_PARAMETER --num_user $NUM_USER&
sleep 1

# fedavg(10/100)
LR=0.09
ALGO=fedavg
S=10
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD --dirichlet $DIRICHLET_PARAMETER --num_user $NUM_USER&
sleep 1

