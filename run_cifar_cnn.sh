# experiment set 
RESULT_DIR=results_cifar_cnn

DATASET=cifar_all.pkl
S=100
T=2000
K=2
B=100
device=5
SEED=5
NET=cnn
WD=0.001
PARTICIPATION=1
PATTERN=adversarial

# our proposed algorithm
ALGO=fdu
LR=0.08
S=100
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &


# sgd with Importance sampling
LR=0.08
ALGO=sgd
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD --importance_sampling &
sleep 1

# sgd without importance sampling
LR = 0.08
ALGO=sgd
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &
sleep 1


# fedavg(50)
LR=0.05
ALGO=fedavg
LR=0.05
S=50
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &
sleep 1


# fedavg(100)
ALGO=fedavg
S=100
LR=0.06
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &
sleep 1

# fedavg(25/100)
ALGO=fedavg
S=25
LR=0.06
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &
sleep 1


# fedavg(10/100)
ALGO=fedavg
S=10
LR=0.05
python main.py --result_dir $RESULT_DIR --participation_level $PARTICIPATION --participation_pattern $PATTERN --gpu --dataset $DATASET --clients_per_round $S --num_round $T --local_step $K --batch_size $B --lr $LR --device $device --seed $SEED --model $NET --algo $ALGO  --noprint --wd $WD &
sleep 1


