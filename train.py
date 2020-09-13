import os
import json
import math
import argparse
import codecs
import random
import torch
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
from empath import Empath

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--model_name', type=str, default="BERT")
	parser.add_argument('--save_dir',type=str, default="saved_models")
  	parser.add_argument('--dataset', type=str, default=None)
	parser.add_argument('--use_empath', type=str, default="y")
  	parser.add_argument('--lr', type=float, default=2e-5)
  	parser.add_argument('--batch_size', type=float, default=32)
	parser.add_argument('--save_policy', type=str, default="loss")
	parser.add_argument('--activation', type=str, default="tanh")
	parser.add_argument('--optim', type=str, default="adam")
	parser.add_argument('--l2', type=str, default="y")
	parser.add_argument('--wd', type=float, default=0.01)
	parser.add_argument('--use_scheduler', type=str, default="n")
	parser.add_argument('--use_dropout', type=str, default="n")
	parser.add_argument('--bert_dropout', type=float, default=0.2)
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--seed', type=int, default=40)

	args = parser.parse_args()

	gpu_id = args.gpu_id
	print(f'GPU_ID = {gpu_id}\n')   
	MODEL_NAME = args.model_name
	print(f'MODEL_NAME = {MODEL_NAME}')
	DATA_PATH = args.dataset
	if DATA_PATH == None :
		raise Exception("Incorrect path to dataset")
	print(f'DATASET = {DATA_PATH}')
	SAVE_DIR = args.save_dir
	if not os.path.exists(SAVE_DIR) :
		os.mkdir(SAVE_DIR)
	print(f"SAVE_DIR = {SAVE_DIR}")
	USE_EMPATH = args.use_empath
	print(f'USE_EMPATH = {USE_EMPATH}')
  	IN_FEATURES = 768
	print(f'IN_FEATURES = {IN_FEATURES}')
	MODEL_SAVING_POLICY = args.save_policy
	print(f'MODEL_SAVING_POLICY = {MODEL_SAVING_POLICY}')
	ACTIVATION_FN = args.activation
	print(f'ACTIVATION_FN = {ACTIVATION_FN}')
	THRESHOLD = 0.33 if ACTIVATION_FN == "tanh" else 0.5
  	OUTPUT_FN = nn.Tanh() if ACTIVATION_FN == "tanh" else nn.Sigmoid()
	print(f'THRESHOLD = {THRESHOLD}')
	LOSS_FN = args.loss_fn
	print(f'LOSS_FN = {LOSS_FN}')
	OPTIM = args.optim
	print(f'OPTIM = {OPTIM}')
	L2_REGULARIZER = args.l2
	print(f'L2_REGULARIZER = {L2_REGULARIZER}') 
	WEIGHT_DECAY = args.wd
	if L2_REGULARIZER == 'y':
		print(f'WEIGHT_DECAY = {WEIGHT_DECAY}')
	USE_SCHEDULER = args.use_scheduler
	print(f'USE_SCHEDULER = {USE_SCHEDULER}')
	USE_DROPOUT = args.use_dropout
	print(f'USE_DROPOUT = {USE_DROPOUT}')
	BERT_DROPOUT = args.bert_dropout
	if USE_DROPOUT == 'y':
		print(f'BERT_DROPOUT = {BERT_DROPOUT}')
	EPOCHS = args.epochs
	print(f'EPOCHS = {EPOCHS}')
	BATCH_SIZE = args.batch_size
	print(f'BATCH_SIZE = {BATCH_SIZE}')
  	LR = args.lr
	print(f'LEARNING_RATE = {LR}')
	
	seed_val = args.seed
	print(f'\nSEED = {str(seed_val)}\n\n')

	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed(seed_val)

  # # If there's a GPU available...
	if torch.cuda.is_available():
		# Tell PyTorch to use the GPU.
		if gpu_id == 0:
		device = torch.device("cuda:0")
		else:
		device = torch.device("cuda:1")
		print('There are %d GPU(s) available.' % torch.cuda.device_count())
		print('We will use the GPU:', torch.cuda.get_device_name(0))
	# If not...
	else:
		print('No GPU available, using the CPU instead.')
		device = torch.device("cpu")

	class LexiconFeatures() :
		def __init__(self) :
		self.lexicon = Empath()

		def tokenize(self, text):
		text = [str(w) for w in tokenizer(text)]
		return text

		def get_features(self, text):
		features = list(self.lexicon.analyze(text, normalize=True).values())
		features = torch.as_tensor([features])
		return(features)

		def parse_sentences(self, sentences) :
		sent_features = []
		for sent in sentences:
			sent_features.append(self.get_features(sent))
		sent_features = torch.cat(sent_features, dim=0)
		print("Empath features: {}".format(sent_features.shape))
		return sent_features


	class CovidData(Dataset) :
		def __init__(self, PATH) :
		self.data = pd.read_csv(PATH).to_dict(orient="records")		
		if MODEL_NAME == 'BERT' :
			self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
		else :
			self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
		self.sentences = []
		self.targets = []
		self.targets_one_hot = []
		self.emotions = ["Thankful", "Anxious", "Annoyed", "Denial", "Joking", "Empathetic", "Optimistic", "Pessimistic", "Sad", "Surprise", "Official report"]
		for i in range(len(self.data)) :
			item = self.data[i]
			self.sentences.append(clean_tweets(item['Tweet']))
			self.targets.append(self.get_target([item[k] for k in self.emotions]))
			self.targets_one_hot.append(torch.tensor([item[k] for k in self.emotions], dtype=torch.float))
		self.encode()
		if USE_EMPATH == 'y' :
			self.lexicon_features = LexiconFeatures().parse_sentences(self.sentences)
			print(self.lexicon_features.shape)
		print("Dataset size: {}".format(len(self.sentences)))
		
		def __len__(self) :
		return len(self.sentences)
		
		def __getitem__(self, idx) :
		if USE_EMPATH == 'y' :
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
			if v == 1:
			temp.append(i)
		temp += [-1]*(11-len(temp))
		return torch.tensor(temp)


	class Net(nn.Module) :
		def __init__(self, embed_size=IN_FEATURES) :
		super(Net, self).__init__()
		self.embed_size = embed_size
		if USE_EMPATH :
			self.embed_size += 194
		self.num_classes = 11
		print(f"Embeddings length: {self.embed_size}")
		
		if MODEL_NAME == 'BERT' :
			self.bert = BertModel.from_pretrained("bert-base-cased")
		else :
			self.bert = RobertaModel.from_pretrained("roberta-base")
		self.fc = nn.Linear(self.embed_size, self.num_classes)
		self.dropout = nn.Dropout(BERT_DROPOUT)
		
		def forward(self,input_ids, attn_masks, token_type_ids, source_lengths, lexicon_features) :
		hidden_states = self.bert(input_ids, attn_masks, token_type_ids)[0]
		output_vectors = hidden_states[:,0,:]
		if USE_DROPOUT == 'y' :
			output_vectors = self.dropout(output_vectors)
		if USE_EMPATH == 'y' :
			output_vectors = torch.cat((output_vectors, lexicon_features), dim=-1)
		logits_out = self.fc(output_vectors)
		return logits_out


	def accuracy(output, target, threshold) :
		output = OUTPUT_FN(output)
		lrap = label_ranking_average_precision_score(target.cpu(), output.cpu())
		output = (output >= threshold).long().cpu()
		target = target.long().cpu()
		acc = (output==target).float().sum()/(torch.ones_like(output).sum())
		jacc = jaccard_score(target, output, average='samples')
		macro_f1 = f1_score(target, output, average='macro')
		micro_f1 = f1_score(target, output, average='micro')	
		hamming = hamming_loss(target, output)
		
		return torch.tensor([acc, jacc, macro_f1, micro_f1, lrap, hamming])


	def run_model(model, batch) :
		input_ids = batch[0].to(device)
		attn_masks = batch[1].to(device)
		token_type_ids = batch[2].to(device)
		source_lengths = batch[4].to(device)
		lexicon_features = None
		if USE_EMPATH == 'y' :
		lexicon_features = batch[6].to(device)
		return model(input_ids, attn_masks, token_type_ids, source_lengths, lexicon_features)

	train_dataloader = DataLoader(CovidData(PATH=f"{DATA_PATH}/train.csv"), shuffle=True, batch_size=batch_size)
	val_dataloader = DataLoader(CovidData(PATH=f"{DATA_PATH}/val.csv"), shuffle=False, batch_size=batch_size)
	loss_function, optimizer = None, None

	if ACTIVATION_FN == "tanh":
		loss_function = nn.MultiLabelMarginLoss()
	else:
		loss_function = nn.BCEWithLogitsLoss()

	if OPTIM == "adam" :
		if L2_REGULARIZER == 'n':
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
		else:
		print(f"L2_REGULARIZER = y and WEIGHT_DECAY = {WEIGHT_DECAY}")
		optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
	else:
		if L2_REGULARIZER == 'n':
		optimizer = AdamW(model.parameters(), lr=lr)
		else:
		print(f"L2_REGULARIZER = y and WEIGHT_DECAY = {WEIGHT_DECAY}")
		optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
	
	model = Net().to(device)
	training_stats = []
	total_steps = len(train_dataloader) * EPOCHS
	scheduler = get_linear_schedule_with_warmup(optimizer, 
						num_warmup_steps = int(total_steps*0.06),
						num_training_steps = total_steps)

	best_val_loss = math.inf
	best_val_acc = 0
	for epoch_i in range(EPOCHS) :
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
		model.train()
		train_loss = []
		reg_loss = None
		for i, batch in enumerate(train_dataloader) :
		model.zero_grad()

		target = batch[3].to(device)
		one_hot = batch[5].to(device)
		output = run_model(model, batch)
		if ACTIVATION_FN == "tanh" :
			output = OUTPUT_FN(output)
			reg_loss = loss_function(output, target)
		else :
			reg_loss = loss_function(output, one_hot)
		train_loss.append(reg_loss.item())
		
		reg_loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)				
		optimizer.step()
		
		if USE_SCHEDULER == 'y' :
			scheduler.step()

		model.eval()
		val_loss = []
		val_acc = []			
		for i, batch in enumerate(val_dataloader) :
		with torch.no_grad() :
			target = batch[3].to(device)
			one_hot = batch[5].to(device)
			output = run_model(model, batch)
			
			if ACTIVATION_FN == "tanh" :
			output = OUTPUT_FN(output)
			reg_loss = loss_function(output, target)
			else :
			reg_loss = loss_function(output, one_hot)
			val_loss.append(reg_loss.item())
			val_acc.append(accuracy(output, one_hot, THRESHOLD))
			

		avg_train_loss = sum(train_loss)/len(train_loss)
		avg_valid_loss = sum(val_loss)/len(val_loss)
		avg_val_acc = torch.stack(val_acc, dim=0).mean(dim=0).tolist()
		
		training_stats.append({
		'training loss' : avg_train_loss,
		'validation loss' : avg_valid_loss,
		'val acc' : avg_val_acc,
		})
		print(json.dumps(training_stats[-1]))

		if MODEL_SAVING_POLICY == "acc":
		if(best_val_acc <= avg_val_acc[0]):
			torch.save(model.state_dict(), f"{SAVE_DIR}/{MODEL_NAME}_EMPATH_{USE_EMPATH}.pt")
			best_val_acc = avg_val_acc[0]
		else:			
		if(best_val_loss >= avg_valid_loss):
			torch.save(model.state_dict(), f"{SAVE_DIR}/{MODEL_NAME}_EMPATH_{USE_EMPATH}.pt")
			best_val_loss = avg_valid_loss

	model = Net().cuda(gpu_id)
	model_path = SAVE_DIR + "/" + MODEL_NAME + "_EMPATH_" + USE_EMPATH + ".pt"
	model.load_state_dict(torch.load(model_path))
	val_loss = []
	val_acc = []
	test_dataset = CovidData(PATH=f"{DATA_PATH}/test.csv")
	test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))
	model.eval()
	for i, batch in enumerate(test_dataloader) :
		with torch.no_grad() :
		target = batch[3].to(device)
		one_hot = batch[5].to(device)
		output = run_model(model, batch)

		reg_loss = None
		if ACTIVATION_FN == "tanh" :
			output = OUTPUT_FN(output)
			reg_loss = loss_function(output, target)
		else :
			reg_loss = loss_function(output, one_hot)
		val_loss.append(reg_loss.item())
		val_acc.append(accuracy(output, one_hot, THRESHOLD))
		print("\n\nBatch: {} val Loss: {} val_r: {}".format(i, reg_loss, val_acc[-1]))

	test_stats = ({
		'test loss' : sum(val_loss)/len(val_loss),
		'test acc' : torch.stack(val_acc, dim=0).mean(dim=0),
	})
	print("\n\n", test_stats)
	print("\n\n\n\n")
