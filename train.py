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
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
# from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, label_ranking_average_precision_score, hamming_loss, jaccard_score
from scipy.stats import pearsonr
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

class DatasetModule(Dataset) :
    def __init__(self, PATH, category) :
        self.data = pd.read_csv(PATH).to_dict(orient="records")
        if ENCODER == 'bert' :
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        else :
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        if self.category == 'emobank' :
            self.get_emobank()
        else :
            self.get_senwave()
    
    def get_emobank(self) :
        self.sentences = []
        self.targets = []
        self.emotions = ["Valence","Arousal","Dominance"]
        for i in tqdm(range(len(self.data))) :
            item = self.data[i]
            self.sentences.append(clean_tweets(item['Tweet']))
            self.targets.append(self.get_target([item[k] for k in self.emotions]))
        self.encode()
        if EMPATH :
          self.lexicon_features = LexiconFeatures().parse_sentences(self.sentences)
          print(self.lexicon_features.shape)
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
          print(self.lexicon_features.shape)
        print("Dataset size: {}".format(len(self.sentences)))
        self.emobank = False
    
    def __len__(self) :
        return len(self.sentences)
    
    def __getitem__(self, idx) :
        one_hot = None
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
            self.bert = AutoModel.from_pretrained("roberta-base")
            self.embed_size = 768
            
        if EMPATH :
            self.embed_size += 194
        self.num_classes_1 = 11
        self.num_classes_2 = 3
        print(f"Embeddings length: {self.embed_size}")
        
        self.fc_1 = nn.Linear(self.embed_size, self.num_classes_1)
        self.fc_2 = nn.Linear(self.embed_size, self.num_classes_2)
        self.tanh = nn.Tanh()
        if USE_DROPOUT :
        	self.dropout = nn.Dropout(p=DROPOUT_RATE)
    
    def forward(self,input_ids, attn_masks, token_type_ids, source_lengths, lexicon_features, category="VAD") :
        sentences = self.bert(input_ids, attn_masks, token_type_ids)[0]
        sentences = sentences[:,0,:]
        if USE_DROPOUT :
        	sentences = self.dropout(sentences)
        if EMPATH :
            sentences = torch.cat((sentences, lexicon_features), dim=-1)
        sentences = self.forward_VAD(sentences) if category == "VAD" else self.forward_emotions(sentences)
        return sentences
    
    def forward_VAD(self,sentences) :
        sentences = self.fc_2(sentences)
        sentences = OUTPUT_FN(sentences)
        if ACTIVATION == 'bce' :
            sentences = sentences*4+1
        else :
            sentences = sentences*2+3
		return sentences

	def forward_emotions(self,sentences) :
		sentences = self.fc_1(sentences)
		sentences = OUTPUT_FN(sentences)
		return sentences

def accuracy_emotions(output, target) :
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
    output = pearsonr(target, output)
    return output

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
	
	senwave_train = DataLoader(DatasetModule(PATH=f"{DATA_DIR}/train.csv","senwave"), shuffle=True, batch_size=BATCH_SIZE)
    senwave_val = DataLoader(DatasetModule(PATH=f"{DATA_DIR}/val.csv","senwave"), shuffle=False, batch_size=BATCH_SIZE)
    
    emobank_train = DataLoader(DatasetModule(PATH=f"{DATA_DIR}/Emobank/train.csv","emobank"), shuffle=True, batch_size=BATCH_SIZE)
    emobank_val = DataLoader(DatasetModule(PATH=f"{DATA_DIR}/Emobank/val.csv","emobank"), shuffle=False, batch_size=BATCH_SIZE)
    
	model = Net().to(DEVICE)
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
		total_steps = len(train_dataloader) * EPOCHS
		scheduler = get_linear_schedule_with_warmup(optimizer, 
                                    num_warmup_steps = int(total_steps*0.06),
                                    num_training_steps = total_steps)
	
	training_stats = []
	best_save = 1e8
	best_model_path = None
	
	for epoch_i in range(EPOCHS) :
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
		num_batches = max(len(senwave_train), len(emobank_train))
		senwave_ = list(enumerate(senwave_train))
		emobank_ = list(enumerate(emobank_train))
		model.train()
		train_loss = {"VAD":[],"Emotions":[]}
		VAD_loss, emotion_loss = None, None
		for i in tqdm(range(num_batches)) :
			model.zero_grad()
			
   			senwave_batch = senwave_[i%len(senwave_)][1]
			target = batch[5].to(DEVICE) if ACTIVATION == "bce" else batch[3].to(DEVICE)
			output = run_model(model, senwave_batch, "senwave")
			emotion_loss = loss_fn(output, target)
   
			emobank_batch = emobank_[i%len(emobank_)][1]
			target = batch[3].to(DEVICE)
			output = run_model(model, emobank_batch, "emobank")
			VAD_loss = VAD_loss_fn(output, target)
   
			loss = VAD_loss + emotion_loss
			loss.backward()
			
			optimizer.step()
			if USE_SCHEDULER :
				scheduler.step()

			if i%50 == 0 :
				print("Batch: {} VAD_Loss: {} Emotion_loss Total_Loss: {} ".format(i, VAD_loss, emotion_loss, loss))

			train_loss["VAD"].append(VAD_loss.item())
			train_loss["Emotion"].append(emotion_loss.item())


		val_loss = {"VAD":[],"Emotions":[]}
		val_acc = {"VAD":[],"Emotions":[]}
		model.eval()
		for i, batch in enumerate(senwave_val) :
			with torch.no_grad() :
				target = batch[5].to(DEVICE) if ACTIVATION == "bce" else batch[3].to(DEVICE)
				output = run_model(model, batch)
				loss = loss_fn(output, target)
				acc = accuracy_emotions(output, target)
				val_loss["Emotions"].append(loss.item())
				val_acc["Emotions"].append(acc)
    
		for i, batch in enumerate(emobank_val) :
			with torch.no_grad() :
				target = batch[3].to(DEVICE)
				output = run_model(model, batch)
				loss = VAD_loss_fn(output, target)
				acc = accuracy_VAD(output, target)
				val_loss["VAD"].append(loss.item())
				val_acc["VAD"].append(accuracy(acc))

		temp = torch.stack(val_acc["Emotion"], dim=0).mean(dim=0).tolist()
		training_stats.append({
			'Total Validation Loss' : sum(val_loss["VAD"])/len(train_loss["VAD"]) + sum(val_loss["Emotion"])/len(train_loss["Emotion"]),
			'VAD training_loss' : sum(train_loss["VAD"])/len(train_loss["VAD"]),
			'VAD validation_loss' : sum(val_loss["VAD"])/len(train_loss["VAD"]),
			'VAD validation_r2' : torch.stack(val_acc["VAD"], dim=0).mean(dim=0).tolist(),
			'Emotion training_loss' : sum(train_loss["Emotion"])/len(train_loss["Emotion"]),
			'Emotion validation_loss' : sum(val_loss["Emotion"])/len(train_loss["Emotion"]),
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
			best_model_path = f"{SAVE_DIR}/{EXP_NAME}/{epoch_i}.ckpt"
			torch.save(model.state_dict(), best_model_path)
			print(f"{SAVE_POLICY} : Saving the model : {epoch_i}")
    
    
	model = Net().to(DEVICE)
	torch.cuda.empty_cache()
	model.load_state_dict(torch.load(best_model_path))
	
	senwave_test = DataLoader(DatasetModule(PATH=f"{DATA_DIR}/test.csv","senwave"), shuffle=False, batch_size=BATCH_SIZE)
	emobank_test = DataLoader(DatasetModule(PATH=f"{DATA_DIR}/Emobank/test.csv","emobank"), shuffle=False, batch_size=BATCH_SIZE)

	test_loss = {"VAD":[],"Emotions":[]}
	test_acc = {"VAD":[],"Emotions":[]}
	model.eval()
	for i, batch in enumerate(senwave_test) :
		with torch.no_grad() :
			target = batch[5].to(DEVICE) if ACTIVATION == "bce" else batch[3].to(DEVICE)
			output = run_model(model, batch)
			loss = loss_fn(output, target)
			acc = accuracy_emotions(output, target)
			test_loss["Emotions"].append(loss.item())
			test_acc["Emotions"].append(acc)

	for i, batch in enumerate(emobank_test) :
		with torch.no_grad() :
			target = batch[3].to(DEVICE)
			output = run_model(model, batch)
			loss = VAD_loss_fn(output, target)
			acc = accuracy_VAD(output, target)
			test_loss["VAD"].append(loss.item())
			test_acc["VAD"].append(accuracy(acc))

	temp = torch.stack(test_acc["Emotion"], dim=0).mean(dim=0).tolist()
	training_stats.append({
		'Total test Loss' : sum(test_loss["VAD"])/len(train_loss["VAD"]) + sum(test_loss["Emotion"])/len(train_loss["Emotion"]),
		'VAD test_loss' : sum(test_loss["VAD"])/len(train_loss["VAD"]),
		'VAD test_r2' : torch.stack(test_acc["VAD"], dim=0).mean(dim=0).tolist(),
		'Emotion test_loss' : sum(test_loss["Emotion"])/len(train_loss["Emotion"]),
		'Emotion test_acc': temp[0],
		'Emotion test_micro_f1': temp[1],
		'Emotion test_macro_f1': temp[2],
		'Emotion test_jacc': temp[3],
		'Emotion test_lrap': temp[4],
		'Emotion test_hamming': temp[5]
	})
	print(json.dumps(training_stats[-1], indent=4))
	with open(f"{SAVE_DIR}/{EXP_NAME}/test.json","w") as fin :
		json.dump(training_stats, fin, indent=4)