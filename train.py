import os
import json
import math
import argparse
import codecs
import random
import torch
import tokenizer
import pickle
import numpy as np
import itertools
from scipy import stats
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, label_ranking_average_precision_score, hamming_loss, jaccard_score
from create_features_v2 import clean_tweets
from tqdm import tqdm
from empath import Empath

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="bert")
parser.add_argument('--use_gpu', action="store_true")
parser.add_argument('--encoder', type=str, default="roberta")
parser.add_argument('--data_dir',type=str, default="data/")
parser.add_argument('--save_dir',type=str, default="saved_models")
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
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--seed', type=int, default=40)

args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

USE_GPU = args.use_gpu
EXP_NAME = args.exp_name
SAVE_DIR = args.save_dir
if not os.path.exists(os.path.join(SAVE_DIR,EXP_NAME)) :
    os.makedirs(os.path.join(SAVE_DIR,EXP_NAME))
DATA_DIR = args.data_dir
if not os.path.exists(SAVE_DIR) :
    raise Exception("Incorrect path to dataset")

ENCODER = args.encoder
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
EMPATH = args.use_empath
ACTIVATION = args.activation
USE_SCHEDULER = args.use_scheduler
THRESHOLD = 0.33 if ACTIVATION == 'tanh' else 0.5
OUTPUT_FN = nn.Tanh() if ACTIVATION == 'tanh' else nn.Sigmoid()
DEVICE = "cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu"
SAVE_POLICY = args.save_policy
OPTIM = args.optim
L2 = args.l2
WD = args.wd
USE_DROPOUT = args.use_dropout
DROPOUT_RATE = args.dropout_rate

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
	"DEVICE": DEVICE
}
print(json.dumps(params))
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

class CovidData(Dataset) :
    def __init__(self, PATH) :
        self.data = pd.read_csv(PATH).to_dict(orient="records")
        if ENCODER == 'bert' :
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        else :
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
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
          print(self.lexicon_features.shape)
        print("Dataset size: {}".format(len(self.sentences)))
    
    def __len__(self) :
        return len(self.sentences)
    
    def __getitem__(self, idx) :
        if EMPATH :
            return self.input_ids[idx], self.attention_masks[idx], self.token_type_ids[idx], self.targets[idx], self.source_lengths[idx], self.targets_one_hot[idx], self.lexicon_features[idx]
        else :
            return self.input_ids[idx], self.attention_masks[idx], self.token_type_ids[idx], self.targets[idx], self.source_lengths[idx], self.targets_one_hot[idx]

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
            self.bert = BertModel.from_pretrained("bert-base-cased")
            self.embed_size = 768
        else :
            self.bert = RobertaModel.from_pretrained("roberta-base")
            self.embed_size = 768
            
        if EMPATH :
            self.embed_size += 194
        self.num_classes = 11
        print(f"Embeddings length: {self.embed_size}")
        
        self.fc = nn.Linear(self.embed_size, self.num_classes)
        self.tanh = nn.Tanh()
        if USE_DROPOUT :
        	self.dropout = nn.Dropout(p=DROPOUT_RATE)
    
    def forward(self,input_ids, attn_masks, token_type_ids, source_lengths, lexicon_features) :
        sentences = self.bert(input_ids, attn_masks, token_type_ids)[0]
        sentences = sentences[:,0,:]
        if USE_DROPOUT :
        	sentences = self.dropout(sentences)
        if EMPATH :
            sentences = torch.cat((sentences, lexicon_features), dim=-1)
        sentences = self.fc(sentences)
        return sentences

def accuracy(output, target) :
    output = OUTPUT_FN(output) if ACTIVATION == 'bce' else output
    lrap = label_ranking_average_precision_score(target.cpu(), output.cpu())
    output = (output >= THRESHOLD).long().cpu()
    target = target.cpu().long()
    micro_f1 = f1_score(target, output, average='micro')
    macro_f1 = f1_score(target, output, average='macro')
    acc = (output==target).float().sum()/(torch.ones_like(output).sum())
    hamming = hamming_loss(target, output)
    jacc = jaccard_score(target, output, average='samples')
    return torch.tensor([acc, micro_f1, macro_f1, jacc, lrap, hamming])

def run_model(model, batch) :
    input_ids = batch[0].to(DEVICE)
    attn_masks = batch[1].to(DEVICE)
    token_type_ids = batch[2].to(DEVICE)
    source_lengths = batch[4].to(DEVICE)
    lexicon_features = None
    if EMPATH :
      lexicon_features = batch[6].to(DEVICE)
    return model(input_ids, attn_masks, token_type_ids, source_lengths, lexicon_features)

if __name__ == "__main__":
	
	train_dataloader = DataLoader(CovidData(PATH=f"{DATA_DIR}/train.csv"), shuffle=True, batch_size=BATCH_SIZE)
	val_dataloader = DataLoader(CovidData(PATH=f"{DATA_DIR}/val.csv"), shuffle=False, batch_size=BATCH_SIZE)
	model = Net().to(DEVICE)
	loss_fn = nn.BCEWithLogitsLoss() if ACTIVATION == 'bce' else nn.MultiLabelMarginLoss()
	
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
		total_steps = len(train_dataloader) * EPOCHS
		scheduler = get_linear_schedule_with_warmup(optimizer, 
                                    num_warmup_steps = int(total_steps*0.06),
                                    num_training_steps = total_steps)
	
	training_stats = []
	best_save = 1e8 if SAVE_POLICY == 'loss' else -1e8
	best_model_path = None
	
	for epoch_i in range(EPOCHS) :
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
		model.train()
		train_loss = []
		reg_loss = None
		for i, batch in enumerate(train_dataloader) :
			model.zero_grad()

			target = batch[3].to(DEVICE)
			one_hot = batch[5].to(DEVICE)
			output = run_model(model, batch)
			if ACTIVATION == "bce" :
				reg_loss = loss_fn(output, one_hot)
			else :
				output = OUTPUT_FN(output)
				reg_loss = loss_fn(output, target)
			train_loss.append(reg_loss.item())
			reg_loss.backward()
			
			optimizer.step()
			if USE_SCHEDULER :
				scheduler.step()

			if i%50 == 0 :
				print("Batch: {} train Loss: {} ".format(i, reg_loss))

		val_loss = []
		val_acc = []
		model.eval()
		for i, batch in enumerate(val_dataloader) :
			with torch.no_grad() :
				target = batch[3].to(DEVICE)
				one_hot = batch[5].to(DEVICE)
				output = run_model(model, batch)

				if ACTIVATION == "bce" :
					reg_loss = loss_fn(output, one_hot)
				else :
					output = OUTPUT_FN(output)
					reg_loss = loss_fn(output, target)
				val_loss.append(reg_loss.item())
				val_acc.append(accuracy(output, one_hot))
#				if i%100 == 0 :
#					print("Batch: {} val Loss: {} val_r: {}".format(i, reg_loss, val_acc[-1]))

		training_stats.append({
			'training loss' : sum(train_loss)/len(train_loss),
			'validation loss' : sum(val_loss)/len(val_loss),
			'val acc' : torch.stack(val_acc, dim=0).mean(dim=0).tolist(),
		})
		print(json.dumps(training_stats[-1]))
		print("acc, micro_f1, macro_f1, jacc, lrap, hamming")
		if SAVE_POLICY == 'loss' :
			if best_save > training_stats[-1]["validation loss"] :
				best_save = training_stats[-1]["validation loss"]
				best_model_path = f"{SAVE_DIR}/{EXP_NAME}/{epoch_i}.ckpt"
				torch.save(model.state_dict(), best_model_path)
				print(f"{SAVE_POLICY} : Saving the model : {epoch_i}")
		else :
			if best_save < training_stats[-1]["val acc"][0] :
				best_save = training_stats[-1]["val acc"][0]
				best_model_path = f"{SAVE_DIR}/{EXP_NAME}/{EPOCHS}.ckpt"
				torch.save(model.state_dict(), best_model_path)
				print(f"{SAVE_POLICY} : Saving the model: {epoch_i}")
    
    
	model = Net().to(DEVICE)
	torch.cuda.empty_cache()
	model.load_state_dict(torch.load(best_model_path))
	val_loss = []
	val_acc = []
	test_dataset = CovidData(PATH=f"{DATA_DIR}/test.csv")
	test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))
	model.eval()
	for i, batch in enumerate(test_dataloader) :
		with torch.no_grad() :
			target = batch[3].to(DEVICE)
			one_hot = batch[5].to(DEVICE)
			output = run_model(model, batch)

			reg_loss = None
			if ACTIVATION == "bce" :
				reg_loss = loss_fn(output, one_hot)
			else :
				output = OUTPUT_FN(output)
				reg_loss = loss_fn(output, target)
			val_loss.append(reg_loss.item())
			val_acc.append(accuracy(output, one_hot))
			print("Batch: {} val Loss: {} val_r: {}".format(i, reg_loss, val_acc[-1]))

	training_stats = ({
		'test loss' : sum(val_loss)/len(val_loss),
		'test acc' : torch.stack(val_acc, dim=0).mean(dim=0).tolist(),
	})
	print(training_stats)
	print("acc, micro_f1, macro_f1, jacc, lrap, hamming")
	with open(f"{SAVE_DIR}/{EXP_NAME}/test.json","w") as fin :
		json.dump(training_stats, fin, indent=4)
