import os
import re
import json
import math
import torch
import codecs
import random
import pickle
import argparse
import tokenizer 
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch.nn as nn
from scipy import stats
from empath import Empath
from emoji import demojize
import torch.optim as optim
from scipy.stats import pearsonr
from nltk.tokenize import TweetTokenizer
from create_features_v2 import clean_tweets
from normalize_tweets import normalizeTweet
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, f1_score, label_ranking_average_precision_score, hamming_loss, jaccard_score

torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="bert")
parser.add_argument('--use_gpu', action="store_true")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--encoder', type=str, default="bertweet")
parser.add_argument('--data_dir',type=str, default="data/")
parser.add_argument('--save_dir',type=str, default="saved_models")
parser.add_argument('--load_pickle', type=str, default=None)
parser.add_argument('--save_pickle', action="store_true")
parser.add_argument('--use_empath', action="store_true")
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--save_policy', type=str, default="loss")
parser.add_argument('--activation', type=str, default="tanh")
parser.add_argument('--optim', type=str, default="adam")
parser.add_argument('--l2', action="store_true")
parser.add_argument('--wd', type=float, default=0.01)
parser.add_argument('--use_scheduler', action="store_true")
parser.add_argument('--use_dropout', action="store_true")
parser.add_argument('--dropout_rate', type=float, default=0.2)
parser.add_argument('--VAD_wt', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--seed', type=int, default=40)

parser.add_argument('--use_hierarchy', action="store_true")
parser.add_argument('--use_successive_reg', action="store_true")
parser.add_argument('--use_connection', action="store_true")
parser.add_argument('--successive_reg_delta', type=float, default=0.1)

args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

USE_GPU = args.use_gpu
GPU_ID = args.gpu_id
EXP_NAME = args.exp_name
SAVE_DIR = args.save_dir
if not os.path.exists(os.path.join(SAVE_DIR,EXP_NAME)) :
    os.makedirs(os.path.join(SAVE_DIR,EXP_NAME))
DATA_DIR = args.data_dir
if not os.path.exists(SAVE_DIR) :
    raise Exception("Incorrect path to dataset")

LOAD_PICKLE = args.load_pickle
if LOAD_PICKLE is not None and not os.path.exists(LOAD_PICKLE) :
    raise Exception("Incorrect path to dataset")
SAVE_PICKLE = args.save_pickle


ENCODER = args.encoder
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
EMPATH = args.use_empath
ACTIVATION = args.activation
USE_SCHEDULER = args.use_scheduler
THRESHOLD = 0.33 if ACTIVATION == 'tanh' else 0.5
OUTPUT_FN = nn.Tanh() if ACTIVATION == 'tanh' else nn.Sigmoid()
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() and USE_GPU else "cpu"
SAVE_POLICY = args.save_policy
OPTIM = args.optim
L2 = args.l2
WD = args.wd
USE_DROPOUT = args.use_dropout
DROPOUT_RATE = args.dropout_rate
VAD_wt = args.VAD_wt

HIERARCHY = args.use_hierarchy
SUCCESSIVE_REG = args.use_successive_reg
USE_CONNECTION = args.use_connection
SUCCESSIVE_REG_DELTA = args.successive_reg_delta

params = {
        "USE_GPU " : USE_GPU,
        "EXP_NAME" : EXP_NAME,
        "SAVE_DIR" : SAVE_DIR,
        "DATA_DIR" : DATA_DIR,
        "ENCODER" : ENCODER,
        "EPOCHS" : EPOCHS,
        "BATCH_SIZE" : BATCH_SIZE,
        "LR" : LR,
        "EMPATH" : EMPATH,
        "ACTIVATION" : ACTIVATION,
        "USE_SCHEDULER" : USE_SCHEDULER,
        "THRESHOLD" : THRESHOLD,
        "SAVE_POLICY" : SAVE_POLICY,
        "OPTIM" : OPTIM,
        "L2" : L2,
        "WD" : WD,
        "USE_DROPOUT" : USE_DROPOUT,
        "DROPOUT_RATE" : DROPOUT_RATE,
        "DEVICE": DEVICE,
        "LOAD_PICKLE": LOAD_PICKLE,
        "SAVE_PICKLE": SAVE_PICKLE,
        "VAD_wt": VAD_wt,
        "HIERARCHY" : HIERARCHY,
        "SUCCESSIVE_REG" : SUCCESSIVE_REG,
        "USE_CONNECTION" : USE_CONNECTION,
        "SUCCESSIVE_REG_DELTA" : SUCCESSIVE_REG_DELTA,
}
print(json.dumps(params, indent=4))
with open(f"{SAVE_DIR}/{EXP_NAME}/hp.json","w") as fin :
    json.dump(params, fin, indent=4)

class LexiconFeatures() :
    def __init__(self):
        self.lexicon = Empath()

    def tokenize(self, text):
        text = [str(w) for w in tokenizer(text)]
        return text

    def get_features(self, text):
        features = list(self.lexicon.analyze(text,normalize=True).values())
        features = torch.as_tensor([features])
        return(features)

    def parse_sentences(self, sentences) :
        temp = []
        for i in tqdm(range(len(sentences))):
            sent = sentences[i]
            temp.append(self.get_features(sent))
        temp = torch.cat(temp, dim=0)
        print("liwc features: {}".format(temp.shape))
        return temp

class DatasetModule(Dataset) :
    def __init__(self, PATH, category) :
        self.data = pd.read_csv(PATH).to_dict(orient="records")
        self.category = category
        if ENCODER == 'bert' :
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        else :
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        if self.category == 'emobank' :
            print("Loading VAD...")
            self.get_emobank()
        else :
            print("Loading Senwave...")
            self.get_senwave()

    def get_emobank(self) :
        self.sentences = []
        self.targets = []
        self.emotions = ["V","A","D"]
        for i in tqdm(range(len(self.data))) :
            item = self.data[i]
            clean_tweet = clean_tweets(item['text'])
            if clean_tweet == "" : continue
            self.sentences.append(clean_tweet)
            self.targets.append(torch.tensor([float(item[k]) for k in self.emotions], dtype=torch.float))
        self.encode()
        if EMPATH :
            self.lexicon_features = LexiconFeatures().parse_sentences(self.sentences)
        print("Dataset size: {}".format(len(self.sentences)))
        self.emobank = True

    def get_senwave(self) :
        self.sentences = []
        self.targets = []
        self.targets_one_hot = []
        self.emotions = ["Thankful","Anxious","Annoyed","Denial","Joking","Empathetic","Optimistic","Pessimistic","Sad","Surprise","Official report"]
        for i in tqdm(range(len(self.data))) :
            item = self.data[i]
            self.sentences.append(clean_tweets(item['Tweet']))
            self.targets.append(self.get_target([item[k] for k in self.emotions]))
            self.targets_one_hot.append(torch.tensor([item[k] for k in self.emotions], dtype=torch.float))
        self.encode()
        if EMPATH :
            self.lexicon_features = LexiconFeatures().parse_sentences(self.sentences)
        print("Dataset size: {}".format(len(self.sentences)))
        self.emobank = False

    def __len__(self) :
        return len(self.sentences)

    def __getitem__(self, idx) :
        one_hot = self.targets[idx]
        if not self.emobank :
            one_hot = self.targets_one_hot[idx]
        if EMPATH :
            return self.input_ids[idx], self.attention_masks[idx], self.token_type_ids[idx], self.targets[idx], self.source_lengths[idx], one_hot, self.lexicon_features[idx]
        else :
            return self.input_ids[idx], self.attention_masks[idx], self.token_type_ids[idx], self.targets[idx], self.source_lengths[idx], one_hot

    def encode(self) :
        self.input_ids = []
        self.attention_masks = []
        self.token_type_ids = []
        self.max_len, self.source_lengths = self.max_length()
        for sent in self.sentences :
            encoded_dict = self.tokenizer.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=self.max_len,
                                                    pad_to_max_length="True",
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt',
                                                    return_token_type_ids = True,
                                                    truncation = True)
            self.input_ids.append(encoded_dict['input_ids'])
            self.attention_masks.append(encoded_dict['attention_mask'])
            self.token_type_ids.append(encoded_dict['token_type_ids'])

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.token_type_ids = torch.cat(self.token_type_ids, dim=0)
        self.source_lengths = torch.LongTensor(self.source_lengths)
        print("input ids: {} attention_masks: {} token_type_ids: {} source_lengths: {}".format(
            self.input_ids.shape, self.attention_masks.shape, self.token_type_ids.shape, self.source_lengths.shape))

    def max_length(self) :
        max_len = 0
        lengths = []
        for sent in self.sentences:
            input_ids = self.tokenizer.encode(sent, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
            lengths.append(min(512, len(input_ids)))
        max_len = min(512, max_len)
        print(f"Max Length:{max_len}")
        return max_len, lengths

    def get_target(self, x) :
        temp = []
        for i,v in enumerate(x) :
            if v==1:
                temp.append(i)
        temp += [-1]*(11-len(temp))
        return torch.tensor(temp)

class Net(nn.Module) :
    def __init__(self, EMBED_SIZE=768) :
        super(Net, self).__init__()

        if ENCODER == 'bert' :
            self.bert = AutoModel.from_pretrained("bert-base-cased")
            self.embed_size = 768
        else :
            self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
            self.embed_size = 768

        if EMPATH :
            self.embed_size += 194
        self.num_classes_1 = 3
        self.num_classes_2 = 11
        print(f"Embeddings length: {self.embed_size}")

        self.fc_1 = nn.Linear(self.embed_size, self.num_classes_1)
        if HIERARCHY :
            self.fc_2 = nn.Linear(self.num_classes_1+self.embed_size, self.num_classes_2)
        else :
            self.fc_2 = nn.Linear(self.embed_size, self.num_classes_2)
        self.tanh = nn.Tanh()
        if USE_DROPOUT :
            self.dropout = nn.Dropout(p=DROPOUT_RATE)

    def forward(self,input_ids, attn_masks, token_type_ids, source_lengths, lexicon_features, category) :
        sentences = self.bert(input_ids, attn_masks, token_type_ids)[0]
        sentences = sentences[:,0,:]
        if USE_DROPOUT :
            sentences = self.dropout(sentences)
        if EMPATH :
            sentences = torch.cat((sentences, lexicon_features), dim=-1)
        sentences = self.forward_VAD(sentences) if category == "VAD" else self.forward_emotions(sentences)
        return sentences

    def forward_VAD(self,sentences) :
        sentences = self.fc_1(sentences)
        sentences = OUTPUT_FN(sentences)
        if ACTIVATION == 'bce' :
            sentences = sentences*4.0+1.0
        else :
            sentences = sentences*2.0+3.0
        return sentences

    def forward_emotions(self,sentences) :
        if HIERARCHY :
            VAD_predictions = self.forward_VAD(sentences)
            sentences = torch.cat((sentences,VAD_predictions), dim=0)
            if not USE_CONNECTION :
                sentences = sentences.detach()
        sentences = self.fc_2(sentences)
        sentences = OUTPUT_FN(sentences)
        return sentences

    def init_weights(self, dist='normal', bias_val=0.01):
        # refence used: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
        for m in [self.fc_1, self.fc_2]:
            if dist == 'uniform':
                torch.nn.init.xavier_uniform(m.weight)
            elif dist == 'normal':
                torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(bias_val)

def accuracy_emotions(output, target) :
#    print(target)
 #   print(output)
    lrap = label_ranking_average_precision_score(target.cpu(), output.cpu())
    output = (output >= THRESHOLD).long().cpu()
    target = target.cpu().long()
    micro_f1 = f1_score(target, output, average='micro')
    macro_f1 = f1_score(target, output, average='macro')
    acc = (output==target).float().sum()/(torch.ones_like(output).sum())
    hamming = hamming_loss(target, output)
    jacc = jaccard_score(target, output, average='samples')
    return torch.tensor([acc, micro_f1, macro_f1, jacc, lrap, hamming])

def accuracy_VAD(output, target) :
    output = output.cpu()
    target = target.cpu()
    output = [pearsonr(target[:,i], output[:,i])[0] for i in range(output.shape[1])]
    return torch.tensor(output)

def run_model(model, batch, category) :
    input_ids = batch[0].to(DEVICE)
    attn_masks = batch[1].to(DEVICE)
    token_type_ids = batch[2].to(DEVICE)
    source_lengths = batch[4].to(DEVICE)
    lexicon_features = None
    if EMPATH :
        lexicon_features = batch[6].to(DEVICE)
    return model(input_ids, attn_masks, token_type_ids, source_lengths, lexicon_features, category)

if __name__ == "__main__":
    
    senwave_train = None
    senwave_val = None
    senwave_test = None
    emobank_train = None
    emobank_val = None
    emobank_test = None

    if LOAD_PICKLE is not None :
        with open(LOAD_PICKLE,"rb") as fin :
            temp_dict = pickle.load(fin)
            senwave_train = temp_dict["senwave_train"]
            senwave_val = temp_dict["senwave_val"]
            senwave_test = temp_dict["senwave_test"]
            emobank_train = temp_dict["emobank_train"]
            emobank_val = temp_dict["emobank_val"]
            emobank_test = temp_dict["emobank_test"]
    else :
        senwave_train = DatasetModule(PATH=f"{DATA_DIR}/train.csv",category="senwave")
        senwave_val = DatasetModule(PATH=f"{DATA_DIR}/val.csv",category="senwave")

        emobank_train = DatasetModule(PATH=f"{DATA_DIR}/Emobank/train.csv",category="emobank")
        emobank_val = DatasetModule(PATH=f"{DATA_DIR}/Emobank/val.csv",category="emobank")

        senwave_test = DatasetModule(PATH=f"{DATA_DIR}/test.csv",category="senwave")
        emobank_test = DatasetModule(PATH=f"{DATA_DIR}/Emobank/test.csv",category="emobank")

        if SAVE_PICKLE :
            print("Saving data.....")
            temp_dict = {
                "senwave_train" : senwave_train,
                "senwave_val" : senwave_val,
                "senwave_test" : senwave_test,
                "emobank_train" : emobank_train,
                "emobank_val" : emobank_val,
                "emobank_test" : emobank_test,
            }
            with open("dataset_bt.pkl","wb") as fin :
                pickle.dump(temp_dict,fin)

    senwave_train = DataLoader(senwave_train, shuffle=True, batch_size=BATCH_SIZE)
    senwave_val = DataLoader(senwave_val, shuffle=False, batch_size=BATCH_SIZE)

    emobank_train = DataLoader(emobank_train, shuffle=True, batch_size=BATCH_SIZE)
    emobank_val = DataLoader(emobank_val,shuffle=False, batch_size=BATCH_SIZE)

    senwave_test = DataLoader(senwave_test, shuffle=False, batch_size=BATCH_SIZE)
    emobank_test = DataLoader(emobank_test, shuffle=False, batch_size=BATCH_SIZE)

    model = Net().to(DEVICE)
    model.init_weights(dist='normal')
    loss_fn = nn.BCELoss() if ACTIVATION == 'bce' else nn.MultiLabelMarginLoss()
    VAD_loss_fn = nn.MSELoss()

    optimizer = None
    if OPTIM == 'adamw' :
        if L2 :
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
        else :
            optimizer = AdamW(model.parameters(), lr=LR)
    else :
        if L2 :
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        else :
            optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = None
    if USE_SCHEDULER :
        total_steps = max(len(senwave_train), len(emobank_train)) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                    num_warmup_steps = int(total_steps*0.06),
                                    num_training_steps = total_steps)

    training_stats = []
    best_save = 1e8
    best_model_path = None
    best_epoch = -1

    for epoch_i in trange(EPOCHS) :
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        num_batches = max(len(senwave_train), len(emobank_train))
        senwave_ = list(enumerate(senwave_train))
        emobank_ = list(enumerate(emobank_train))
        model.train()
        train_loss = {"VAD":[],"Emotion":[]}
        VAD_loss, emotion_loss = None, None
        history = None
        
        for i in trange(num_batches) :
            model.zero_grad()
            if HIERARCHY :
                
                emobank_batch = emobank_[i%len(emobank_)][1]
                target = emobank_batch[3].to(DEVICE)
                emobank_output = run_model(model, emobank_batch, "VAD")
                VAD_loss = VAD_loss_fn(emobank_output, target.float())
                VAD_loss.backward()
                optimizer.step()
                
                model.zero_grad()
                senwave_batch = senwave_[i%len(senwave_)][1]
                target = senwave_batch[5].to(DEVICE) if ACTIVATION == "bce" else senwave_batch[3].to(DEVICE)
                senwave_output = run_model(model, senwave_batch, "Emotion")
                emotion_loss = loss_fn(senwave_output, target)
                
                if USE_CONNECTION and SUCCESSIVE_REG and history is not None :
                    loss = torch.norm(model.fc_1.weight-history.weight) + torch.norm(model.fc_1.bias-history.bias)
                    emotion_loss += SUCCESSIVE_REG_DELTA*loss
                emotion_loss.backward()
                optimizer.step()
                history = (model.fc_1).clone()
            else :   
                senwave_batch = senwave_[i%len(senwave_)][1]
                target = senwave_batch[5].to(DEVICE) if ACTIVATION == "bce" else senwave_batch[3].to(DEVICE)
                senwave_output = run_model(model, senwave_batch, "Emotion")
                emotion_loss = loss_fn(senwave_output, target)
                
                emobank_batch = emobank_[i%len(emobank_)][1]
                target = emobank_batch[3].to(DEVICE)
                emobank_output = run_model(model, emobank_batch, "VAD")
                VAD_loss = VAD_loss_fn(emobank_output, target.float())

                loss = (VAD_wt*VAD_loss + (1-VAD_wt)*emotion_loss).float()
                loss.backward()

                optimizer.step()
                
            if USE_SCHEDULER :
                 scheduler.step()

            if i%50 == 0 :
                print("Batch: {} VAD_Loss: {} Emotion_loss: {} Total_Loss: {} ".format(i, VAD_loss, emotion_loss, VAD_loss+emotion_loss))

            train_loss["VAD"].append(VAD_loss.item())
            train_loss["Emotion"].append(emotion_loss.item())


        val_loss = {"VAD":[],"Emotion":[]}
        val_acc = {"VAD":[],"Emotion":[]}
        model.eval()
        for i, batch in enumerate(senwave_val) :
            with torch.no_grad() :
                target = batch[5].to(DEVICE) if ACTIVATION == "bce" else batch[3].to(DEVICE)
                output = run_model(model, batch, "Emotion")
                loss = loss_fn(output, target)
                target = batch[5].to(DEVICE)
                acc = accuracy_emotions(output, target)
                val_loss["Emotion"].append(loss.item())
                val_acc["Emotion"].append(acc)

        for i, batch in enumerate(emobank_val) :
            with torch.no_grad() :
                target = batch[3].to(DEVICE)
                output = run_model(model, batch, "VAD")
                loss = VAD_loss_fn(output, target)
                acc = accuracy_VAD(output, target)
                val_loss["VAD"].append(loss.item())
                val_acc["VAD"].append(acc)

        temp = torch.stack(val_acc["Emotion"], dim=0).mean(dim=0).tolist()
        training_stats.append({
                'Total Validation Loss' : sum(val_loss["VAD"])/len(val_loss["VAD"]) + sum(val_loss["Emotion"])/len(val_loss["Emotion"]),
                'VAD training_loss' : sum(train_loss["VAD"])/len(train_loss["VAD"]),
                'VAD validation_loss' : sum(val_loss["VAD"])/len(val_loss["VAD"]),
                'VAD validation_r2' : torch.stack(val_acc["VAD"], dim=0).mean(dim=0).tolist(),
                'Emotion training_loss' : sum(train_loss["Emotion"])/len(train_loss["Emotion"]),
                'Emotion validation_loss' : sum(val_loss["Emotion"])/len(val_loss["Emotion"]),
                'Emotion validation_acc': temp[0],
                'Emotion validation_micro_f1': temp[1],
                'Emotion validation_macro_f1': temp[2],
                'Emotion validation_jacc': temp[3],
                'Emotion validation_lrap': temp[4],
                'Emotion validation_hamming': temp[5]
        })
        print(json.dumps(training_stats[-1], indent=4))

        if best_save > training_stats[-1]["Total Validation Loss"] :
            best_save = training_stats[-1]["Total Validation Loss"]
            best_model_path = f"{SAVE_DIR}/{EXP_NAME}/best.ckpt"
            best_epoch = epoch_i
            torch.save(model.state_dict(), best_model_path)
            print(f"{SAVE_POLICY} : Saving the model : {epoch_i}")

    model = Net().to(DEVICE)
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(best_model_path))
    os.system(f"rm {best_model_path}")

    test_loss = {"VAD":[],"Emotion":[]}
    test_acc = {"VAD":[],"Emotion":[]}
    model.eval()
    for i, batch in enumerate(senwave_test) :
        with torch.no_grad() :
            target = batch[5].to(DEVICE) if ACTIVATION == "bce" else batch[3].to(DEVICE)
            output = run_model(model, batch, "Emotion")
            loss = loss_fn(output, target)
            target = batch[5].to(DEVICE)
            acc = accuracy_emotions(output, target)
            test_loss["Emotion"].append(loss.item())
            test_acc["Emotion"].append(acc)

    for i, batch in enumerate(emobank_test) :
        with torch.no_grad() :
            target = batch[3].to(DEVICE)
            output = run_model(model, batch, "VAD")
            loss = VAD_loss_fn(output, target)
            acc = accuracy_VAD(output, target)
            test_loss["VAD"].append(loss.item())
            test_acc["VAD"].append(acc)

    temp = torch.stack(test_acc["Emotion"], dim=0).mean(dim=0).tolist()
    test_stats = {
                'Total test Loss' : sum(test_loss["VAD"])/len(test_loss["VAD"]) + sum(test_loss["Emotion"])/len(test_loss["Emotion"]),
                'VAD test_loss' : sum(test_loss["VAD"])/len(test_loss["VAD"]),
                'VAD test_r2' : torch.stack(test_acc["VAD"], dim=0).mean(dim=0).tolist(),
                'Emotion test_loss' : sum(test_loss["Emotion"])/len(test_loss["Emotion"]),
                'Emotion test_acc': temp[0],
                'Emotion test_micro_f1': temp[1],
                'Emotion test_macro_f1': temp[2],
                'Emotion test_jacc': temp[3],
                'Emotion test_lrap': temp[4],
                'Emotion test_hamming': temp[5],
                'EPOCHS': best_epoch
        }
    print(json.dumps(test_stats, indent=4))
    with open(f"{SAVE_DIR}/{EXP_NAME}/test.json","w") as fin :
        json.dump(test_stats, fin, indent=4)
