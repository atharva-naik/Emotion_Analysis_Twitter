import os
import ast
import math
import argparse
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-f","--fname", type=str, default=None, help="address of csv file with predictions")
parser.add_argument("-i","--id", type=str, default=None, help="name of the field that represents the unique id in the spreadsheet, to drop duplicates")
parser.add_argument("-d","--date", type=str, default='date', help="name of the field that represents date")
parser.add_argument("-e","--emotions", type=str, default='categories', help="name of the field that contains emotions present in the tweet according to the model")
parser.add_argument("-b","--body_text", type=str, default='text', help="name of the field that text body of the processed tweet")
parser.add_argument("-l","--leap", type=bool, default=True, help = "do all dates fall in a leap year")
parser.add_argument("-t","--timestep", type=int, default=7, help="number of days in a bucket")
parser.add_argument("-c","--chunk_size", type=int, default=5000, help="number of tweets in a chunk")
parser.add_argument("-a","--aspects", type=str, default=None, help="text file containing aspects")
args = parser.parse_args()

# GLOBAL_PARAMETERS
# NOTE: Corresponding to each emotion, you could supply the correct amount of aspect names here 
aspect_names = {'Annoyed':['name 1','name 2','name 3','name 4','name 5','name 6','name 7'],
                'Optimistic':['name 1','name 2','name 3','name 4','name 5','name 6','name 7']} 

# NOTE: update emotion list for different type of models
emotions = ['Annoyed', 'Anxious', 'Denial', 'Empathetic', 'Joking', 'Official report', 'Optimistic', 'Pessimistic', 'Sad', 'Surprise', 'Thankful']

if args.leap:
    months = {'jan':0, 'feb':31, 'mar':60, 'apr':91, 'may':121, 'jun':152, 'jul':182, 'aug':213, 'sep':244, 'oct':274, 'nov':304, 'dec':335} # made for leap year keeping 2020 in mind
else:
    months = {'jan':0, 'feb':31, 'mar':59, 'apr':90, 'may':120, 'jun':151, 'jul':181, 'aug':212, 'sep':243, 'oct':273, 'nov':303, 'dec':334}
# END OF GLOBAL PARAMETERS

# FILE CONVENTIONS
# NOTE: PLEASE STICK TO THE FOLLOWING DATE FORMAT
# Sat Jan 25 14:46:50 +0000 2020
# THE MONTH AS FIRST 3 LETTERS AND DATE AFTER IT (required portion for the date)
# NOTE: Combine the aspects for multiple emotions if you want to, but follow this format
'''
$Annoyed
Aspect 0:
['Super|0.80218446', 'Om|0.7849536', 'P|0.7835137', 'Birthday|0.7781286', 'Rampal|0.77538383', 'Singh|0.7741655', 'R|0.7719685', 'Bhai|0.77027375', 'Shah|0.75322556', 'K|0.752318', 'Maharaj|0.749282', 'Ram|0.7480929', 'Shanti|0.7449265', 'singh|0.74229205', 'Amit|0.7415925', 'Kumar|0.7348441', 'Dil|0.73226017', 'C|0.72995895', 'Krishna|0.72852004', 'Wishes|0.7222283', 'H|0.7221066', 'Senior|0.71982396', 'Amrinder|0.7176805', 'Mr|0.7128144', 'Sahab|0.7118733', 'M|0.71103793', 'Honourable|0.7109686', 'Gandhi|0.7074396', 'madam|0.70738757', 'HappyYadav|0.7073167', 'HRD|0.7062956', 'Dr|0.70573205', 'Reddy|0.70455945', 'ble|0.70311415', 'Gupta|0.7027844', 'ki|0.70155555', 'Maa|0.7012368', 'Anna|0.699873', 'Sh|0.6979978', 'Manish|0.6978023', 'Rahul|0.69549716', 'birthday|0.69526064', 'heartiest|0.6907556', 'Inspiration|0.687852', 'Khan|0.68784475', 'Captain|0.6872393', 'Brother|0.68691075', 'ko|0.68633807', 'Citizen|0.68524444', 'respected|0.6836127', 'Madam|0.68358076', 'ne|0.68180156', 'Naveen|0.6810742', 'yr|0.68105805', 'Address|0.6808317', 'Chairman|0.67933404', 'Saint|0.67808235', 'Madurai|0.6768426', 'Sri|0.67570436', 'Happy|0.67422336', 'Leader|0.6733048', 'Respected|0.6717738', 'Shri|0.67100346', 'Ex|0.67065036', 'Old|0.67064494', 'Heartiest|0.67033863', 'Sanjay|0.6677464', 'Rajiv|0.6670197', 'Son|0.66669905', 'Honble|0.66386855', 'evening|0.6638428', 'Insan|0.6634636', 'S|0.66316533', 'Rukh|0.662693', 'Sant|0.6624124', 'Off|0.66215324', 'Raja|0.6615407', 'Prof|0.6614795', 'Oh|0.6609454', 'garu|0.6606664', 'IAS|0.6605636', 'O|0.6603335', 'L|0.65993774', 'Samir|0.6598133', 'Board|0.6587205', 'Job|0.6583066', 'G|0.6582495', 'Guru|0.6582382', 'congratulations|0.6578028', 'Vice|0.6569755', 'shri|0.6569246', 'Sharma|0.65610254', 'Prize|0.6559379', 'RIP|0.6538565', 'Ajay|0.65318114', 'Heart|0.6530305', 'SIR|0.6526507', 'Ratan|0.65216726', 'bhai|0.651717', 'Sonu|0.6515153']
Aspect 1:
['efforts|0.7925285', 'frontline|0.73940027', 'front|0.69747937', 'commitment|0.69533265', 'selfless|0.69503', 'volunteers|0.68990576', 'service|0.680462', 'dedication|0.66594934', 'towards|0.6642061', 'workers|0.6567296', 'gratitude|0.6494928', 'citizens|0.6436775', 'nurses|0.64280856', 'warriors|0.63395476', 'team|0.62274873', 'Salute|0.6205227', 'solidarity|0.62042594', 'support|0.6166887', 'nation|0.6163865', 'dedicated|0.60852504', 'courage|0.60784256', 'line|0.6055065', 'staff|0.6050607', 'Nation|0.60422564', 'forces|0.6020051', 'effort|0.5958905', 'serving|0.5806243', 'healthcare|0.5713203', 'doctors|0.5690801', 'leading|0.56776655', 'died|0.56667227', 'services|0.5649434', 'salute|0.56440455', 'Doctors|0.5638608', 'serve|0.5629959', 'collective|0.5620695', 'Kudos|0.55917275', 'peoples|0.557987', 'society|0.5565254', 'tirelessly|0.556382', 'spirit|0.555899', 'globe|0.555894', 'professionals|0.54791754', 'round|0.54293144', 'police|0.542194', 'brave|0.5412629', 'contributing|0.5395143', 'personnel|0.5395074', 'contribution|0.53805417', 'Police|0.5370683', 'nations|0.53671515', 'selflessly|0.53597736', 'clock|0.53546965', 'respect|0.53284174', 'heroes|0.53281343', 'standing|0.53253657', 'Indians|0.5308071', 'helping|0.524148', 'Nurses|0.5224388', 'supporting|0.51978064', 'committed|0.5190981', 'women|0.5189385', 'leadership|0.51754373', 'medical|0.5170412', 'working|0.5163151', 'entire|0.51490635', 'role|0.5131549', 'fund|0.5123849', 'determination|0.5111005', 'protecting|0.50834805', 'Their|0.5074765', 'countrymen|0.5074067', 'Warriors|0.5065216', 'salutes|0.5026718', 'Our|0.5024043', 'fighting|0.5017526', 'forefront|0.5006607', 'farmers|0.49915338', 'strengthen|0.49650857', 'fellow|0.49516043', 'relief|0.49444282', 'Fund|0.4925459', 'thanks|0.4920921', 'Indian|0.49197358', 'migrant|0.49131224', 'Medical|0.4893129', 'battle|0.48400337', 'State|0.48188758', 'PMCARES|0.48139796', 'combat|0.48041224', 'security|0.4789153', 'commendable|0.47860038', 'employees|0.47813568', 'tireless|0.47597274', 'needy|0.47558522', 'across|0.47550803', 'appreciated|0.4737871', 'Frontline|0.471452', 'involved|0.46795434', 'express|0.46759528']
$Anxious
Aspect 0:
['Super|0.80218446', 'Om|0.7849536', 'P|0.7835137', 'Birthday|0.7781286', 'Rampal|0.77538383', 'Singh|0.7741655', 'R|0.7719685', 'Bhai|0.77027375', 'Shah|0.75322556', 'K|0.752318', 'Maharaj|0.749282', 'Ram|0.7480929', 'Shanti|0.7449265', 'singh|0.74229205', 'Amit|0.7415925', 'Kumar|0.7348441', 'Dil|0.73226017', 'C|0.72995895', 'Krishna|0.72852004', 'Wishes|0.7222283', 'H|0.7221066', 'Senior|0.71982396', 'Amrinder|0.7176805', 'Mr|0.7128144', 'Sahab|0.7118733', 'M|0.71103793', 'Honourable|0.7109686', 'Gandhi|0.7074396', 'madam|0.70738757', 'HappyYadav|0.7073167', 'HRD|0.7062956', 'Dr|0.70573205', 'Reddy|0.70455945', 'ble|0.70311415', 'Gupta|0.7027844', 'ki|0.70155555', 'Maa|0.7012368', 'Anna|0.699873', 'Sh|0.6979978', 'Manish|0.6978023', 'Rahul|0.69549716', 'birthday|0.69526064', 'heartiest|0.6907556', 'Inspiration|0.687852', 'Khan|0.68784475', 'Captain|0.6872393', 'Brother|0.68691075', 'ko|0.68633807', 'Citizen|0.68524444', 'respected|0.6836127', 'Madam|0.68358076', 'ne|0.68180156', 'Naveen|0.6810742', 'yr|0.68105805', 'Address|0.6808317', 'Chairman|0.67933404', 'Saint|0.67808235', 'Madurai|0.6768426', 'Sri|0.67570436', 'Happy|0.67422336', 'Leader|0.6733048', 'Respected|0.6717738', 'Shri|0.67100346', 'Ex|0.67065036', 'Old|0.67064494', 'Heartiest|0.67033863', 'Sanjay|0.6677464', 'Rajiv|0.6670197', 'Son|0.66669905', 'Honble|0.66386855', 'evening|0.6638428', 'Insan|0.6634636', 'S|0.66316533', 'Rukh|0.662693', 'Sant|0.6624124', 'Off|0.66215324', 'Raja|0.6615407', 'Prof|0.6614795', 'Oh|0.6609454', 'garu|0.6606664', 'IAS|0.6605636', 'O|0.6603335', 'L|0.65993774', 'Samir|0.6598133', 'Board|0.6587205', 'Job|0.6583066', 'G|0.6582495', 'Guru|0.6582382', 'congratulations|0.6578028', 'Vice|0.6569755', 'shri|0.6569246', 'Sharma|0.65610254', 'Prize|0.6559379', 'RIP|0.6538565', 'Ajay|0.65318114', 'Heart|0.6530305', 'SIR|0.6526507', 'Ratan|0.65216726', 'bhai|0.651717', 'Sonu|0.6515153']
Aspect 1:
['efforts|0.7925285', 'frontline|0.73940027', 'front|0.69747937', 'commitment|0.69533265', 'selfless|0.69503', 'volunteers|0.68990576', 'service|0.680462', 'dedication|0.66594934', 'towards|0.6642061', 'workers|0.6567296', 'gratitude|0.6494928', 'citizens|0.6436775', 'nurses|0.64280856', 'warriors|0.63395476', 'team|0.62274873', 'Salute|0.6205227', 'solidarity|0.62042594', 'support|0.6166887', 'nation|0.6163865', 'dedicated|0.60852504', 'courage|0.60784256', 'line|0.6055065', 'staff|0.6050607', 'Nation|0.60422564', 'forces|0.6020051', 'effort|0.5958905', 'serving|0.5806243', 'healthcare|0.5713203', 'doctors|0.5690801', 'leading|0.56776655', 'died|0.56667227', 'services|0.5649434', 'salute|0.56440455', 'Doctors|0.5638608', 'serve|0.5629959', 'collective|0.5620695', 'Kudos|0.55917275', 'peoples|0.557987', 'society|0.5565254', 'tirelessly|0.556382', 'spirit|0.555899', 'globe|0.555894', 'professionals|0.54791754', 'round|0.54293144', 'police|0.542194', 'brave|0.5412629', 'contributing|0.5395143', 'personnel|0.5395074', 'contribution|0.53805417', 'Police|0.5370683', 'nations|0.53671515', 'selflessly|0.53597736', 'clock|0.53546965', 'respect|0.53284174', 'heroes|0.53281343', 'standing|0.53253657', 'Indians|0.5308071', 'helping|0.524148', 'Nurses|0.5224388', 'supporting|0.51978064', 'committed|0.5190981', 'women|0.5189385', 'leadership|0.51754373', 'medical|0.5170412', 'working|0.5163151', 'entire|0.51490635', 'role|0.5131549', 'fund|0.5123849', 'determination|0.5111005', 'protecting|0.50834805', 'Their|0.5074765', 'countrymen|0.5074067', 'Warriors|0.5065216', 'salutes|0.5026718', 'Our|0.5024043', 'fighting|0.5017526', 'forefront|0.5006607', 'farmers|0.49915338', 'strengthen|0.49650857', 'fellow|0.49516043', 'relief|0.49444282', 'Fund|0.4925459', 'thanks|0.4920921', 'Indian|0.49197358', 'migrant|0.49131224', 'Medical|0.4893129', 'battle|0.48400337', 'State|0.48188758', 'PMCARES|0.48139796', 'combat|0.48041224', 'security|0.4789153', 'commendable|0.47860038', 'employees|0.47813568', 'tireless|0.47597274', 'needy|0.47558522', 'across|0.47550803', 'appreciated|0.4737871', 'Frontline|0.471452', 'involved|0.46795434', 'express|0.46759528']
'''
# it is not too much extra effort as ABAE generates aspects in this format only
# you just have to combien aspects for various emotions, and add $<emotion> before they begin 
# NOTE: Change file extension from .log to .txt 

# read tsv file having the predictions of classifier
# example:
# {'id': 1145,
#  'uid': 1221082002490712064,
#  'date': 'Sat Jan 25 14:46:50 +0000 2020',
#  'text': 'coronavirus threat',
#  'categories': "['Anxious', 'Surprise', 'Official report']"}

# END OF FILE CONVENTIONS

# HELPER FUNCTIONS
# convert dates to serial number of day in the year  
def day_to_date(n):
    i = 11
    while n-list(months.values())[i]<=0:
        i -= 1
    month = list(months.keys())[i]
    day = n-list(months.values())[i]
    return month + ' ' + str(day)

def date_str_to_num(date):
    table = {k:i+1 for i,k in enumerate(months.keys())}
    month, day = date.split()
    month = str(table[month])
    day = str(day)
    if len(day) == 1: day = '0'+day
    if len(month) == 1: month = '0'+month
    return day+'/'+month

def parse_aspect_file(fname): # code to read aspect files
    '''Returns a dictionary which has dictionary of word:score for each aspect for each key/emotion'''
    '''the structure is dictionary of list of dictionary'''
    # aspects[emotion][k][word] = <score of that word> NOTE: k is the aspect number, strating from 0
    aspects = {}
    if fname:
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if line[0] == '$':
                    curr = line[1:]  
                    aspects[curr] = []
                elif line.split()[0] == 'Aspect':
                    aspects[curr].append([])
                else:
                    temp = ast.literal_eval(line)
                    aspects[curr][-1] = {} 
                    for item in temp:
                        k,v = item.split('|')
                        aspects[curr][-1][k] = float(v)                    
            f.close()
    return aspects

# END OF HELPER FUNCTIONS

# ANALSYIS OF COUNTS
if args.fname:
    if args.fname.split('.')[-1] == 'csv':
        if args.id:
            data = pd.read_csv(args.fname).drop_duplicates(subset=args.id).to_dict('records')
        else:
            data = pd.read_csv(args.fname).to_dict('records')
    elif args.fname.split('.')[-1] == 'tsv':
        data = pd.read_csv(args.fname, sep='\t').to_dict('records')
    else:
        raise(Exception("Please enter a csv or tsv file"))

    for i, item in enumerate(data):
        for j, word in enumerate(item[args.date].lower().split()):
            if word in months.keys():
                try:
                    item["day number"] = months[word] + int(item[args.date].lower().split()[j+1])
                except:
                    raise(Exception("date format error"))

    sorted_data = sorted(data, key=lambda x: x['day number'])
    norm_counts, counts, day_dist, dates = [], [], [], [] 
    min_day, max_day = sorted_data[0]['day number'], sorted_data[-1]['day number']

    j = min_day
    while j <= max_day:
        day_dist.append(0)
        dates.append((day_to_date(j), day_to_date(j+args.timestep-1)))
        counts.append({k:0 for k in emotions})
        norm_counts.append({k:0 for k in emotions})
        j += args.timestep

    for i, item in enumerate(sorted_data): 
        ind = int((item['day number'] - min_day + 1)/args.timestep)
        if (item['day number'] - min_day + 1)%args.timestep == 0: ind -= 1 
        day_dist[ind] += 1
        if type(item[args.emotions]) is str:
            emo_pres = ast.literal_eval(item[args.emotions])
        else:
            emo_pres = item[args.emotions]
        for emotion in emo_pres:
            counts[ind][emotion] += 1
            norm_counts[ind][emotion] += 1

    for emotion in emotions:
        for i in range(len(norm_counts)):
            if day_dist[i] != 0: 
                norm_counts[i][emotion] /= day_dist[i]

    chunk_counts, chunk_dates, temp = [], [], []
    ctr, from_, to, j = 0, 0, 0, -1
    for i,item in enumerate(sorted_data):
        if (ctr+1)%args.chunk_size == 0:
            to = item['day number']
            chunk_dates.append((day_to_date(from_), day_to_date(to)))
        if ctr%args.chunk_size == 0:
            j += 1
            from_ = item['day number']
            chunk_counts.append({k:0 for k in emotions})
        ctr += 1
        if type(item[args.emotions]) is str:
            emo_pres = ast.literal_eval(item[args.emotions])
        else:
            emo_pres = item[args.emotions]
        for emotion in emo_pres:
            chunk_counts[j][emotion] += 1 
    if ctr%args.chunk_size != 0:
        to = item['day number']
        chunk_dates.append((day_to_date(from_), day_to_date(to)))
    print("Summary:")
    print("{} tweets analysed".format(len(sorted_data)))
    print("dates range from '{}' to '{}'".format(sorted_data[0][args.date][0:19], sorted_data[-1][args.date][0:19]))
    print("chunk size = {} tweets".format(args.chunk_size))
    print("number of chunks = {}".format(len(chunk_counts)))
    print("time step = {} days".format(args.timestep))
    print("number of buckets for given time step = {}".format(len(counts)))

# ASPECT TREND ANALYSIS
aspects = parse_aspect_file(args.aspects)
aspect_counts = {}
if aspects:
    for i, emotion in enumerate(aspects.keys()):
        aspect_counts[emotion] = []
        for j in range(len(aspects[emotion])):
            aspect_counts[emotion].append([])
            for _ in range(len(chunk_counts)):
                aspect_counts[emotion][j].append(0)
    curr_chunk = -1
    for i, item in enumerate(sorted_data):
        if i%args.chunk_size == 0: curr_chunk += 1 
        if type(item[args.emotions]) is str:
            emo_pres = ast.literal_eval(item[args.emotions])
        else:
            emo_pres = item[args.emotions]
        for emotion in emo_pres:
            if emotion in aspects.keys():
                for j in range(len(aspects[emotion])):
                    for word in item[args.body_text].split():
                        if aspects[emotion][j].__contains__(word):
                            aspect_counts[emotion][j][curr_chunk] += 1

try:
    os.mkdir('results')
except FileExistsError:
    pass

# GRAPH PLOTTING FUNCTIONS
def plot_graphs(title, dates, counts, xlbl, ylbl, emotions, output):
    x = [i for i in range(len(dates))]
    for emotion in emotions:
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (25,17)
        plt.xticks(x, labels=[date_str_to_num(itm[0])+'-'+date_str_to_num(itm[1]) for itm in dates], rotation=90)
        
        y = [counts[i][emotion] for i in range(len(dates))]
        plt.plot(x, y)
        plt.xlabel(xlbl + ' of ' + emotion)
        plt.ylabel(ylbl)
        plt.title(title + ' of ' + emotion)
        plt.savefig('results/'+ output + '_'+ emotion + '.png')
        plt.clf()

def plot_aspects(title, dates, counts, xlbl, ylbl, labels, output):
    x = [i for i in range(len(dates))]
    for emotion in counts.keys():
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (25,17)
        plt.xticks(x, labels=[date_str_to_num(itm[0])+'-'+date_str_to_num(itm[1]) for itm in dates], rotation=90)

        for j in range(len(counts[emotion])):
            y = [counts[emotion][j][i] for i in range(len(dates))]
            if emotion in labels.keys():
                plt.plot(x, y, label=labels[emotion][j])
            else:
                plt.plot(x, y, label=j+1)

        plt.xlabel(xlbl + ' of ' + emotion)
        plt.ylabel(ylbl)
        plt.legend(loc='upper right')
        plt.title(title + ' of ' + emotion)
        plt.savefig('results/'+ output + '_'+ emotion + '.png')
        plt.clf()

print("Analysis finished ...")
print("Plotting ...")
# PLOT IF REQUESTED 
if args.fname:
    plot_graphs("Absolute Emotion Levels", dates, counts, 'time ranges', 'counts', emotions, "absolute_counts")
    plot_graphs("Normailsed Emotion Levels", dates, norm_counts, 'time ranges', 'counts', emotions, "normalised_counts")
    plot_graphs("Chunkwise Emotion Levels", chunk_dates, chunk_counts, 'time ranges', 'counts', emotions, "chunk_counts")

if args.aspects:
    #currently chunking scheme is same as global scheme
    plot_aspects("Chunkwise Aspect Mentions", chunk_dates, aspect_counts, 'time ranges', 'counts', aspect_names,"aspect_chunk_counts")
