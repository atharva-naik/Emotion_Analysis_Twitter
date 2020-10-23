import os
import json
import subprocess
from subprocess import call
from multiprocessing import Pool
import random

ENCODER = "bert"
save_dir = "bert_mtl"
SEED = [42]
print(SEED)

process = "python -W ignore train.py --exp_name p{} --gpu_id {} --use_gpu --encoder {} --data_dir data/ --load_pickle dataset.pkl --save_dir "+ save_dir +" --lr {} --batch_size {} --save_policy loss --activation {} --optim adamw --l2 --wd 0.01 {} --use_dropout --dropout_rate 0.2 --epochs 5 --seed {}"
def run(args) :
    subprocess.call(process.format(*args).split(), stdout=subprocess.PIPE)
    
i = 0
model_stats = []
LR = ["1e-5", "2e-5 --use_scheduler", "5e-5 --use_scheduler"]
BATCH_SIZE = [16, 32]
EMPATH = ['--use_empath','']


for seed in SEED :
    for empath in EMPATH :
        for lr in LR :
            for batch_size in BATCH_SIZE :
                print(f"Running instances {i+1} {i+2} out of {2*len(SEED)*len(EMPATH)*len(BATCH_SIZE)*len(LR)}")  
                args_list = [(i+1,0,ENCODER,lr,batch_size,"bce",empath, seed),
                            (i+2,1,ENCODER,lr,batch_size,"tanh",empath, seed)]
                p = Pool(processes=2)
                p.map(run, args_list)
                with open(f"{save_dir}/p{i+1}/test.json","r") as f :
                    model_stats.append(json.load(f))
                with open(f"{save_dir}/p{i+2}/test.json","r") as f :
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
