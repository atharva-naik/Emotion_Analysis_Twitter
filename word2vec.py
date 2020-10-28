import os
import gensim
import pandas as pd
from gensim.models import Word2Vec
from normalize_tweets import normalizeTweet

SIZE = 200
WINDOWS = 5
MIN_COUNT = 1
WORKERS = 4

target_isos = ['gb','it','de']
for iso in target_isos:
    print("iso being processed is: {}".format(iso))
    f = open('Panacea/tweet_texts_{}.txt'.format(iso),"r")
    sentences = []
    for i, line in enumerate(f):
        line = normalizeTweet(line)
        if len(line.split()) > 0:
            sentences.append(line.split())
        if i%1000 == 0:
            print(i)
    f.close()
    model = Word2Vec(sentences, size=SIZE, window=WINDOWS, min_count=MIN_COUNT, workers=WORKERS)
    model.save(f'w2v_{iso}_200.bin')