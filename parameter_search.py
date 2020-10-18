import os
import json
import subprocess
from subprocess import call

# subprocess.run(['python','train.py'])
SEED = ['10','20','30','40','50']
DROPOUT = ['0','0.1','0.2','0.3']

i = 0
model_stats = []
for seed in SEED:
    for dropout in DROPOUT:
        i+=1
        subprocess.call(['python','test.py','--exp_name','p'+str(i),'--seed',seed,'--dropout_rate',dropout])

        with open(f"saved_models/p{i}/test.json","r") as f :
            model_stats.append(json.load(f))
        f.close()

curr_min = 100000
min_ind = ''
for i, stats in enumerate(model_stats):
    print('p'+str(i)+'Loss is =',stats['Total test Loss'])
    if curr_min > stats['Total test Loss']:
        curr_min = stats['Total test Loss']
        min_ind = 'p'+str(i)
print("best is:",min_ind, "with loss of:",curr_min)