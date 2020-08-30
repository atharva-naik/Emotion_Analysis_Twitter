import os
import ast
import math
import seaborn as sns
import pandas as pd 
import matplotlib.plt as plt

# read tsv file having the predictions of classifier
# example:
# {'id': 1145,
#  'uid': 1221082002490712064,
#  'date': 'Sat Jan 25 14:46:50 +0000 2020',
#  'text': 'coronavirus threat',
#  'categories': "['Anxious', 'Surprise', 'Official report']"}
data = pd.read_csv('./panacea_new_latest.tsv', sep='\t').drop_duplicates(subset='uid').to_dict('records')
# TODO make changes to drop duplicates part

# convert dates to serial number of day in the year  
months = {'Jan':0,'Feb':31,'Mar':60,'Apr':91,'May':121,'Jun':152, 'Jul':182} 
for i, item in enumerate(data):
    item['day_number'] = months[item['date'].split()[1]] + int(item['date'].split()[2])

sorted_data = sorted(data, key=lambda k: k['day_number']) 
