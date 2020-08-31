# Emotion_Analysis_Twitter
**Code for**: Analyzing Changing Emotions in Indian Twitter Data Rajdeep Mukherjee, Sriyash Poddar*, Atharva Naik*, Soham Dasgupta "How Have We Reacted To The COVID- 19 Pandemic? Analyzing Changing Indian Emotions Through The Lens of Twitter" 8th ACM IKDD CODS and 26th COMAD, 2021 (Under Review)
**Running Instructions**:
1. **For Scraper**: 
    1. to run as scraper: python scrape.py -s True
    2. to scrape with a list of queries(at least one): python scrape.py -s True -q elonmusk tesla
        (This will be interpreted as ["elonmusk", "tesla"])
    3. to specify limit on number of tweets: python scrape.py -s True -q elonmusk tesla -l 10000
        (at most 10,000 tweets will be scraped for any query with a given set of conditions (coordinates, time intervals etc.))     
    4. to run as hydrater: python scrape.py -H True
    5. restrictions related to coordinates, time intervals, can be modified directly in the script. Queries and limits on tweets can also be manually added 