# experiment set 
RESULT_DIR=results_mnist_logit_p1

DATASET=mnist_all.pkl
S=100
T=200
K=2
B=100
LR=0.01
device=2
SEED=5
NET=logistic
WD=0.001
PARTICIPATION=1
PATTERN=adversarial

# our proposed algorithm
ALGO=fdu
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &


# sgd with Importance sampling
ALGO=sgd
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD --importance_sampling &
sleep 1

# sgd without importance sampling 
ALGO=sgd
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &
sleep 1


# fedavg(10/100)
ALGO=fedavg
S=10
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &
sleep 1

# fedavg(25/100)
ALGO=fedavg
S=25
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &
sleep 1


# fedavg(50/100)
ALGO=fedavg
S=50
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &
sleep 1


# fedavg(100/100)
ALGO=fedavg
S=100
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &
sleep 1


