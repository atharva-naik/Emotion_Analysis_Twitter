import os
import json
import subprocess
from subprocess import call
from multiprocessing import Pool
import random
# subprocess.run(['python','train.py'])
ENCODER = "bert"
#SEED = [random.randint(0,100), random.randint(0,100), random.randint(0,100)]
SEED = [42]
print(SEED)
process_1 = "python train.py --exp_name p{} --gpu_id {} --use_gpu --encoder {} --data_dir data/ --load_pickle dataset.pkl --save_dir saved_models --lr {} --batch_size {} --save_policy loss --activation {} --optim {} --l2 --wd {} {} --use_dropout --dropout_rate {} --epochs 5 --seed {}"

process_2 = "python train.py --exp_name p{} --gpu_id {} --use_gpu --encoder {} --data_dir data/ --load_pickle dataset.pkl --save_dir saved_models_{}_2 --lr {} --batch_size {} --save_policy loss --activation {} --optim {} --l2 --wd {} {} --use_dropout --dropout_rate {} --epochs 10 --seed {} --use_hierarchy"

process_3 = "python train.py --exp_name p{} --gpu_id {} --use_gpu --encoder {} --data_dir data/ --load_pickle dataset.pkl --save_dir saved_models_{}_3 --lr {} --batch_size {} --save_policy loss --activation {} --optim {} --l2 --wd {} {} --use_dropout --dropout_rate {} --epochs 10 --seed {} --use_hierarchy --use_connection --use_successive_reg --successive_reg_delta {}"

def run(args) :
    subprocess.call(process_1.format(*args).split(), stdout=subprocess.PIPE)
    
i = 0
model_stats = []
LR = [1e-5, 2e-5, 5e-5]
BATCH_SIZE = [16, 32]
#POCHS = ["2","3","4 --use_scheduler"]
#ACTIVATION = ['tanh', 'bce']
OPTIM = ['adamw']
WD = [0.01]
EMPATH = ['--use_empath','']
DROPOUT = [0.2]

for seed in SEED :
    for empath in EMPATH :
        for lr in LR :
            for batch_size in BATCH_SIZE :
                for dropout in DROPOUT:
                    for optim in OPTIM :
                        for wd in WD :
                            print(f"Running instances {i+1} {i+2} out of {2*len(SEED)*len(EMPATH)*len(BATCH_SIZE)*len(OPTIM)*len(WD)*len(LR)*len(DROPOUT)}")  
                            args_list = [(i+1,0,ENCODER,ENCODER,lr,batch_size,"bce",optim,wd,empath,dropout, seed),
                                        (i+2,1,ENCODER,ENCODER,lr,batch_size,"tanh",optim,wd,empath,dropout, seed)]
                            p = Pool(processes=2)
                            p.map(run, args_list)
                            with open(f"saved_models/p{i+1}/test.json","r") as f :
                                model_stats.append(json.load(f))
                            with open(f"saved_models/p{i+2}/test.json","r") as f :
                                model_stats.append(json.load(f))
                            i+=2

curr_min = 1e8
min_ind = ''
for i, stats in enumerate(model_stats):
    print('p'+str(i)+'Loss is =',stats['Total test Loss'])
    if curr_min > stats['Total test Loss']:
        curr_min = stats['Total test Loss']
        min_ind = 'p'+str(i)
print("best is:",min_ind, "with loss of:",curr_min)
