import os
import json
import time
import twarc
from twarc import Twarc
import pandas as pd 
import GetOldTweets3 as got 

''' This code contains a scraper and a hydrater. The scraper uses an 
external API GetOldTweets3 to get tweets from queries and other 
constraints, however some limited aspects of the tweets can only be accessed. 
The hydrater uses a twitter library called twarc, that takes in tweet ids
of tweets to retrieve all details that can be taken from twitter, if the tweet 
still exists. Comment the part that you don't want to use before running '''

#### NOTE: scrape data with getoldtweets3 ####

# max number of tweets to be scraped for each query
LIM = 10000
# specific coordinates to be scraped
coords = [(19.75, 75.71),(22.97, 78.66)]
# query terms to be searched for
queries = ['coronavirus','covid','outbreak','sars-cov-2','koronavirus',
'corona','wuhancoronavirus','lockdown','lock down','wuhanlockdown',
'kungflu','covid-19','covid19','coronials','coronapocalypse','panicbuy',
'panicbuying','panic buy','panicbuy','panic shop','panicshopping',
'panicshop','coronakindness','stayhomechallenge','DontBeASpreader',
'sheltering in place','shelteringinplace','chinesevirus','chinese virus',
'quarantinelife','staysafestayhome','stay safe stay home','flattenthecurve',
'flatten the curve','china virus','chinavirus','quarentinelife','covidiot',
'epitwitter','saferathome','SocialDistancingNow','Social Distancing',
'SocialDistancing']
# "from" and "to" dates for search NOTE ("yyyy-mm-dd" format)
time_intervals = [("2020-01-01","2020-03-05"),("2020-04-06","2020-06-30")]

# print params during run
print("max number of tweets per query {}".format(LIM))
print("query terms {}".format(queries))
print("time intervals {}".format(time_intervals))
print("locations specified {}".format(coords))

for _from, to in time_intervals:
    for query in queries:
        for x,y in coords:
            tweets_list = []
            # search for tweets with specified conditions
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setMaxTweets(LIM).setSince(_from).setUntil(to).setNear(x+', '+y).setWithin('300km')
            results = got.manager.TweetManager.getTweets(tweetCriteria)
            # convert to list of lists
            for tweet in results:
                tweets_list.append([tweet.username, tweet.date, tweet.retweets, tweet.favorites, tweet.text, tweet.hashtags, tweet.geo, tweet.id, tweet.permalink])

            # store results as csv
            df = pd.DataFrame(tweets_list, columns=['username', 'date', 'retweets', 'favorites', 'text', 'hashtags', 'geo', 'id', 'permalink']) 
            # drop duplicated text (retweets)
            df = df.drop_duplicates(subset='text')
            # sort by username
            df.sort_values("username",inplace=True)
            try:
                os.mkdir('data')
            except FileExistsError:
                pass
            name = "./data/" + query + ".lim="+ str(LIM) + "_since=" + _from + "_until=" + to + "_near=(" + str(x) +','+ str(y) +").csv"
            
            # print result stats for each search criterion 
            print('{} unique tweets scraped from {}'.format(len(df), query))
            print("saving to {}".format(len(df), name))
            df.to_csv(name)

#### scrape data with getoldtweets3 ####

#### NOTE: hydrate tweet ids with twarc ####

# create the twarc object
twarc = Twarc()

# read ids from a csv file of ids, with column name "ids"
ids = list(pd.read_csv('clean_country_code_IN.csv')['ids'])
tweets = []

# check input integrity
print("{} ids were read".format(len(ids)))
print("hydrating from {} to {} ...".format(ids[0], ids[-1]))

# create generator of hydrated tweets from list of ids
results = twarc.hydrate(ids)
for tweet in results:
    tweets.append(tweet)

# convert to dataframe
df = pd.DataFrame(tweets)
df.drop_duplicates(subset='text')
df.sort_values('username', inplace=True)
df.to_csv('hydrated_tweets.csv')
print("total {} tweets".format(len(df)))

# write the ids of tweets not found to a text file
not_found = list(set(ids).difference(set(list(df['id']))))
print("{} tweets were not found".format(len(not_found)))
print("their ids are saved to not_found.txt ...")
with open('not_found.txt',"w") as f:
    for id in not_found:
        f.write(id+'\n')

#### hydrate tweet ids with twarc ####