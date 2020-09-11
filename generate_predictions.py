import os
import csv
import nltk
nltk.download('punkt')
import json
import pandas as pd
from transformers import RobertaModel, RobertaTokenizer
from create_features_v2 import clean_tweets
from empath import Empath
import torch
import torch.nn as nn

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
  parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--model_name', type=str, default="BERT")
  parser.add_argument('--model_path',type=str)
  parser.add_argument('--output_path',type=str)
  parser.add_argument('--data', type=str, default=None)
	parser.add_argument('--use_empath', type=str, default="y")
	parser.add_argument('--activation', type=str, default="tanh")
  gpu_id = args.gpu_id
	print(f'GPU_ID = {gpu_id}\n')   
  MODEL_NAME = args.model_name
  print(f'MODEL_NAME = {MODEL_NAME}')
  DATA_PATH = args.dataset
  if DATA_PATH == None :
    raise Exception("Incorrect path to dataset")
  print(f'DATASET = {DATA_PATH}')
  MODEL_PATH = args.model_path
  print(f'MODEL PATH = {MODEL_PATH}')
  OUTPUT_PATH = args.output_path
  print(f'OUTPUT PATH = {OUTPUT_PATH}')
	USE_EMPATH = args.use_empath
	print(f'USE_EMPATH = {USE_EMPATH}')
  IN_FEATURES = 768
	print(f'IN_FEATURES = {IN_FEATURES}')
  print(f'ACTIVATION_FN = {ACTIVATION_FN}')
	THRESHOLD = 0.33 if ACTIVATION_FN == "tanh" else 0.5
  OUTPUT_FN = nn.Tanh() if ACTIVATION_FN == "tanh" else nn.Sigmoid()
	print(f'THRESHOLD = {THRESHOLD}')
 
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

  tokenizer = None
  if MODEL_NAME == 'BERT' :
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
  else :
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
  def encode(sentence) : 
    encoded_dict = tokenizer_.encode_plus(sentence,
                                            add_special_tokens=True,
                                            return_attention_mask = True,
                                            return_tensors = 'pt', 
                                            return_token_type_ids = True)
    return encoded_dict['input_ids'].to(device), encoded_dict['attention_mask'].to(device), encoded_dict['token_type_ids'].to(device)

  files = os.listdir(DATA_PATH)
  data = []
  for i in files :
    if "csv" not in i :
      continue
    temp = pd.read_csv(main_dir+"/"+i).to_dict(orient='records')
    data.extend(temp)
  print(f'NO OF TWEETS :{len(data)}')
  
  clean_data = []
  sentences = []
  for d in data :
    if d['lang'] != 'en' :
      continue
    temp_dict = {
        'id' : d['Unnamed: 0'],
        'uid' : d['id'],
        'date' : d['created_at'],
        'text' : clean_tweets(d['full_text'])
    }
  clean_data.append(temp_dict)
  sentences.append(temp_dict['text'])
  lexicons = LexiconFeatures().parse_sentences(sentences)

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
      if USE_EMPATH == 'y' :
        output_vectors = torch.cat((output_vectors, lexicon_features), dim=-1)
      logits_out = self.fc(output_vectors)
      logits_out = OUTPUT_FN(logits_out)
      return logits_out

  model = Net().to(device)
  model.load_state_dict(torch.load(MODEL_PATH))
  emotions = ["Thankful","Anxious","Annoyed","Denial","Joking","Empathetic","Optimistic","Pessimistic","Sad","Surprise","Official report"]
  model.eval()

  for data_point in clean_data :
    a,b,c = encode(data_point['text'])
    d = lexicons[i].to(device).unsqueeze(0)
    temp = model(a,b,c,d).squeeze()
    temp = (temp>=THRESHOLD).long().tolist()
    clean_data[i]['categories'] = [emotions[j] for j in range(len(emotions)) if temp[j] == 1]

  keys = clean_data[0].keys()
  with open(OUTPUT_PATH, 'w') as output_file:
      dict_writer = csv.DictWriter(output_file, keys, delimiter="\t")
      dict_writer.writeheader()
      dict_writer.writerows(clean_data)
