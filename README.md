# Emotion_Analysis_Twitter
**Code for**: 

Analyzing Changing Emotions in Indian Twitter Data :  "How Have We Reacted To The COVID- 19 Pandemic? Analyzing Changing Indian Emotions Through The Lens of Twitter" 8th ACM IKDD CODS and 26th COMAD, 2021 (Under Review)

### Setting up:

```bash
pip install -r requirements.txt
```

### Preparing dataset:

**add the train, val and test csv files to data/ folder**

### How to run:

#### For scraping/hydrating:
```bash
python scrape.py -s True -q [queries] -l [limit on tweets]  
python scrape.py -H True -f [files containing tweets ids]

Note : The -H stands for hydration, and -s for scraping. Restrictions related to coordinates, time intervals, can be modified directly in the script.
```

#### For training :
```bash
python train.py --gpu_id (gpu_id_to_use) --model_name (model_name) --save_dir (path to save dir) -- dataset (path to dataset) --use_empath (y/n) --lr (learning rate) --batch_size (batch_size) --save_policy (criterion_for_saving_policy) --activation (activation fn) --optim (optimizer) --l2 (y/n) --wd (weight_decay) --use_scheduler (use) --use_dropout (y/n) --bert_dropout (dropout value) --epochs (num_epochs) --seed (seed)
```

#### For generating predictions :
```bash
python generate_predictions.py --gpu_id (gpu_id) --model_name (BERT/ROBERTA) --model_path (path to saved model) --output_path (path to save dir) --data (path to dir containing hydrated csv) --use_empath (y/n) --activation (tanh/bce)
```
#### For plotting graphs :
```bash
python plot.py -f (csv_or_tsv_file_with_predictions) -i(unique_id_field) -d(date_field) -e(field_with_emotion_predictions) -b(text_field) -l(boolean_flag_for_leap_year) -t(timestep_for_type_2) -c(chunk_size_for_type_1) -a(address_of_aspect_file)
Note-1: Plot types: (1)Fixed number of tweets, (2)Fixed time interval, (3)Aspect mentions (for fixed number of tweets out of the total)  
Note-2: By default the aspect term searching is case sensitive
Note-3: Aspects labels can be supplied in the code
```
