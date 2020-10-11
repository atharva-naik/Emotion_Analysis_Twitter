import os
import re
import json
import string
import codecs
import numpy as np
import contractions
import emoji
import itertools
from pathlib import Path
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from tokenizer import tokenizer
#import matplotlib.pyplot as plt

# from bs4 import BeautifulSoup
# from geotext import GeoText
# from wordsegment import load, segment 
# load()
# from autocorrect import spell, Speller
# import twikenizer as twk

'''
emoticons_happy = set([
	':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
	':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
	'=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
	'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
	'<3'
	])

emoticons_sad = set([
	':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
	':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
	':c', ':{', '>:\\', ';('
	])

#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)

# https://en.wikipedia.org/wiki/Unicode_block
EMOJI_PATTERN = re.compile(
	"["
	"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	"\U0001F300-\U0001F5FF"  # symbols & pictographs
	"\U0001F600-\U0001F64F"  # emoticons
	"\U0001F680-\U0001F6FF"  # transport & map symbols
	"\U0001F700-\U0001F77F"  # alchemical symbols
	"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
	"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
	"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
	"\U0001FA00-\U0001FA6F"  # Chess Symbols
	"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
	"\U00002702-\U000027B0"  # Dingbats
	"\U000024C2-\U0001F251" 
	"]+", flags=re.UNICODE)

#Emoji patterns
emoji_pattern = re.compile("["
	 u"\U0001F600-\U0001F64F"  # emoticons
	 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
	 u"\U0001F680-\U0001F6FF"  # transport & map symbols
	 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	 u"\U00002702-\U000027B0"
	 u"\U000024C2-\U0001F251"
	 "]+", flags=re.UNICODE)


def clean_tweets_ver1(tweet):
	# twk = twk.Twikenizer()
	# twk.tokenize(tweet)
	print("\nOriginal tweet:", tweet)
	places = GeoText(tweet)
	# to keep names of places if they start with a #
	l = places.cities + places.countries
	for w in l:
		i = tweet.find(w)
		if(tweet[i-1]=='#'):
			tweet = tweet[:i-1] + tweet[i:]
	# to remove links that start with HTTP/HTTPS in the tweet 
	tweet = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)','', tweet, flags=re.MULTILINE)
	# to remove other url links
	tweet = re.sub(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,61}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)','', tweet, flags=re.MULTILINE)
	# to remove time
	tweet = re.sub(r'(\d|\d\d):\d\d(am|pm)','', tweet, flags=re.MULTILINE)
	tweet = re.sub(r'(\s|.)#\w+', r'\1', tweet)
	tweet = ' '.join(word for word in tweet.split(' ') if not word.startswith('#'))
	tweet = ' '.join(word for word in tweet.split(' ') if not word.startswith('&'))
	# to remove #, @ and # 
	tweet = ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",tweet).split())
	
	# tweet = ' '.join(segment(tweet)) 
	# # to check and correct spelling 
	# tweet = ' '.join([spell(w) for w in tweet.split()])
	# # to remove digits 
	# tweet = re.sub(r"\d", "", tweet)
	
	# # to lower the tweets 
	tweet = tweet.lower()
	return tweet

def clean_tweets_ver2(tweet):
	# stop_words = set(stopwords.words('english'))
	
	tweet = ' '.join(word for word in tweet.split(' ') if not (word.startswith('#') or word.startswith('@') or word.startswith('https') or word.startswith('http')) )
	word_tokens = word_tokenize(tweet)
	tweet = re.sub(r':', '', tweet)	
	tweet = re.sub(r'‚Ä¶', '', tweet)
	#replace consecutive non-ASCII characters with a space
	tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
	tweet = emoji_pattern.sub(r'', tweet)
	# filtered_tweet = [w for w in word_tokens if not w in stop_words]
	filtered_tweet = []
	for w in word_tokens:
		if w not in emoticons and w not in string.punctuation:
			filtered_tweet.append(w)
	# ' '.join(word for word in txt.split(' ') if not word.startswith('#'))
	return ' '.join(filtered_tweet)
'''

def translator(text):	
	with open("slang.txt", "r") as myCSVfile:
		# Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
		dataFromFile = csv.reader(myCSVfile, delimiter="=")
	text = text.strip().split(" ")
	j = 0
	for _str in text:
		# Removing Special Characters.
		_str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
		for row in dataFromFile:
			# Check if selected word matches short forms[LHS] in text file.
			if _str.upper() == row[0]:
				# If match found replace it with its Abbreviation in text file.
				text[j] = row[1]			
		j = j + 1
	# Replacing commas with spaces for final output.
	return ' '.join(text)


def clean_tweets(tweet):
	# print("\nOriginal tweet:", tweet)
	# T = tokenizer.TweetTokenizer(preserve_handles=False, preserve_hashes=False, preserve_case=True, preserve_url=False, regularize=True)
	# tokens = tokenizer.tokenize(tweet)
	text = tweet.strip()
	# HTML decoding
	# text = BeautifulSoup(text, 'lxml').get_text()
	# Remove emojis and non-ascii characters
	text = text.encode('ascii', 'ignore').decode('ascii')	
	# Remove mentions and hyperlinks
	text = ' '.join(word for word in text.split(' ') if not (word.startswith('@') or word.startswith('https') or word.startswith('http') or word.startswith('www.')))
	# Expand the contractions
	text = contractions.fix(text)
	# Remove hashtag while keeping hashtag text
	text = re.sub(r'#','', text)
	# Remove HTML special entities (e.g. &amp;)
	text = re.sub(r'\&\w*;', '', text)
	# Remove tickers
	text = re.sub(r'\$\w*', '', text)
	# Remove hyperlinks and urls
	text = re.sub(r'https?:\/\/.*\/\w*', '', text)
	text = re.sub(r'https?://[A-Za-z0-9./]+','', text)
	# Remove whitespace (including new line characters)
	text = re.sub(r'\s\s+','', text)
	text = re.sub(r'[ ]{2, }',' ', text)
	# Remove URL, RT, mention(@)
	text=  re.sub(r'http(\S)+', '', text)
	text=  re.sub(r'http ...', '', text)
	text = re.sub(r'http', '', text)
	text=  re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+','', text)
	text=  re.sub(r'RT[ ]?@','', text)
	text = re.sub(r'@[\S]+','', text)
	text = re.sub(r'@[A-Za-z0-9]+','',text)
	# Remove words with 4 or fewer letters
	# text = re.sub(r'\b\w{1,4}\b', '', text)
	# Remove &, < and >
	text = re.sub(r'&amp;?', 'and', text)
	text = re.sub(r'&lt;','<', text)
	text = re.sub(r'&gt;','>', text)
	# Cleaning UTF-8 BOM (Byte Order Mark)
	try:
		text = text.decode("utf-8-sig").replace(u"\ufffd", "?")
	except:
		text = text
	# Remove non-ascii words and characters
	text = re.sub(r'_[\S]?', r'', text)
	# Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
	text= ''.join(c for c in text if c <= '\uFFFF') 
	text = text.strip()
	# Remove misspelling words
	text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
	# Remove emoji
	text = emoji.demojize(text)
	text = text.replace(":"," ")
	text = ' '.join(text.split()) 
	text = re.sub("([^\x00-\x7F])+"," ",text)
	# Remove Mojibake (also extra spaces)
	text = ' '.join(re.sub("[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
	# Insert space between punctuation marks
	text = text.replace(r'([\w\d]+)([^\w\d ]+)', r'\1 \2')
	text = text.replace(r'([^\w\d ]+)([\w\d]+)', r'\1 \2')
	# Letters only
	# text = re.sub("[^a-zA-Z]", " ", text)
	# Remove extra spaces
	text = re.sub('\s+', ' ', text)	
	# In case filtered text is blank
	if text == "":
		text = tweet.strip()
		text = re.sub(r'https?:\/\/.*\/\w*', '', text)
		text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
		text=  re.sub(r'http(\S)+', '', text)
		text=  re.sub(r'http ...', '', text)
		text = re.sub(r'http', '', text)
		word_tokens = word_tokenize(text.strip())
		text = ' '.join(word for word in word_tokens if word not in string.punctuation).strip()
	
	# print("Cleaned Tweet: ", text)
	return text
