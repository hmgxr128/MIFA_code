import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
from matplotlib.ticker import ScalarFormatter

COLOR_MAP = {
    'FedAvg (S=100)': 'red',
    'FedAvg (S=10)': '#FF6EB4',
    'FedAvg (S=25)': '#6E6E6E',
    'FedAvg (S=50)': 'forestgreen',
    'FedAvg-IS': 'cornflowerblue',
    'biased FedAvg': 'darkorange',
    'MIFA': '#CD00CD',
}



def opt_reader(options):
    """
        Give Option a better algorithm name. used for plotting
    """
    if options['algo']=='fedavg':
        return 'FedAvg (S='+str(options['clients_per_round'])+')'
    if options['algo']=='sgd':
        if not options['importance_sampling']:
            return 'biased FedAvg'
        else:
            return 'FedAvg-IS'
    if options['algo']=='fdu':
        return 'MIFA'
    if options['algo']=='fdu_no_wait':
        return 'MIFA (0)'

def data_preprocess(options, logs):
    if 'fedavg' in options['algo']:
        time_steps = sorted([int(x) for x in logs.keys()])
        for i in range(min(time_steps) + 1, options['num_round']):
            if i not in time_steps:
                logs[str(i)] = logs[str(i-1)]



def beautiful_print_report(report_dict):
    for algo, res in report_dict.items():
        gap = res['acc'] - report_dict['MIFA']['acc']
        print("{}   acc:{:.2f}  std:{:.2f}  gap:{:.2f} \n".format(algo, res['acc'], res['std'], gap))


def smooth_curve(y_arr, n=2):
    length = y_arr.shape[0]
    ret = np.zeros(length)
    ret[:n] = y_arr[:n]
    ret[-n:] = y_arr[-n:]
    for idx in range(n, length - n):
        ret[idx] = np.mean(y_arr[(idx - n) : (idx + n + 1)])
    return(ret)
        

ROOT_DIR = sys.argv[1]

"""
assume that the result dir is organized as
results/experiment_id/log.json
results/experiment_id/options.json
"""
experiment_list = []
for roots, dirs, files  in os.walk(ROOT_DIR):
    if 'options.json' in files:
        experiment_list.append(roots)
print("exp list length", len(experiment_list))

log = defaultdict(list)
for exp_id in experiment_list:
    log_file = os.path.join(exp_id, 'log.json')
    opt_file = os.path.join(exp_id, 'options.json')
    with open(log_file) as f:
        log_dict = json.load(f)

    with open(opt_file) as f:
        opt_dict = json.load(f)



    del opt_dict['seed']
    del opt_dict['result_dir']
    del opt_dict['device']

    log[repr(opt_dict)].append(log_dict) #  group experiments of the same setting but different random seeds
    data_preprocess(opt_dict, log_dict)





avail_plots = ['train_loss','train_acc','test_loss','test_acc']
target = avail_plots[int(sys.argv[2])]
print('plotting ', target)


fig, ax = plt.subplots(figsize = (10, 8))
handler = {}
myfontdict2 = {'family':'Times New Roman','size':20}



report_dict = defaultdict(dict)
for options, logs in log.items():
    options = eval(options)
    
    total_round = options['num_round']
    total_round = 2000
    first_times = [min( [int(x) for x in log_dict.keys()]  ) for log_dict in logs]
    max_first_time = max(first_times)

    time_steps = range(max_first_time + 1, total_round + 1)
    alg_name = opt_reader(options)


    ys = [[log_dict[str(t-1)][target] for t in time_steps] for log_dict in logs]
    ys = np.array(ys)
    if target=='test_acc':
        ys =  ys*100
    y_mean = np.mean(ys,0)
    y_std = np.std(ys,0)
    alg_name = opt_reader(options)
    color= COLOR_MAP[alg_name]
    y_mean = smooth_curve(y_mean, n=2)
    y_std = smooth_curve(y_std, n=2)
    handler[alg_name] = plt.plot(time_steps, y_mean, color)[0]

    report_dict[alg_name]['acc'] = y_mean[-1]
    report_dict[alg_name]['std'] = y_std[-1]
    plt.fill_between(time_steps, y_mean-y_std, y_mean+y_std, alpha=0.3, color =color)

report_dict = dict(sorted(report_dict.items(), key=lambda item: item[1]['acc'], reverse=True))
beautiful_print_report(report_dict)

assert sys.argv[3] in ['cifar', 'mnist']
dataset = sys.argv[3]

if target=='train_loss':
    plt.xlabel("communication round", myfontdict2)
    plt.ylabel("training loss", myfontdict2)
    plt.yscale('log')
    # scaling for MNIST
    if dataset ==  'mnist':
        plt.ylim(0.4,1.8)
        ax.set_yticks([0.4,0.6,0.8,1,1.2,1.4])
    
        ax.yaxis.set_major_formatter(ScalarFormatter())
        #ax.yaxis.set_minor_formatter(ScalarFormatter())
    else:
        plt.ylim(1.8,2.59)
        ax.set_yticks([2.0,2.1,2.2,2.3,2.4])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
    label = sorted(handler.keys())
    handler_content = [handler[x] for x in label]
    myfontdict2 = {'family':'Times New Roman','size':20} 
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    #plt.legend(handler_content, label, prop=myfontdict2)
    plt.savefig("./plot/new_{}_{}_{}_train_loss.pdf".format(dataset,part,net), bbox_inches='tight')

if target=='test_acc':
    plt.xlabel("communication round", myfontdict2)
    plt.ylabel("test acc (%)", myfontdict2)
    plt.yscale('linear')
    if dataset ==  'mnist':
        plt.ylim(50,91)
    else:
        plt.ylim(15, 38)
        plt.yticks([15, 20, 25, 30, 35])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    label = sorted(handler.keys())
    handler_content = [handler[x] for x in label]
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    #plt.legend(handler_content, label, prop=myfontdict2)
    plt.savefig("./plot/{}_{}.pdf".format(dataset, avail_plots[int(sys.argv[2])]), bbox_inches='tight')

