from nltk.tokenize import word_tokenize
#from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
#import random
import numpy as np
import pandas as pd
#import re

#all_songs = pd.read_csv('lyrics/billboard_lyrics_1964-2015.csv', encoding="ISO-8859-1").as_matrix()
all_songs = pd.read_csv('lyrics/songdata.csv').as_matrix()
sw = list(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in sw] # remove stopwords
    #tokens = [t for t in tokens if not re.search("^'",t)]
    return tokens

words=[]
song_words=[]
for song in all_songs[:1000]:
    s=song[3]
    s=my_tokenizer(s)
    song_words.append(s)
    for i in s:
        words.append(i)

"""words=FreqDist(words)
print(words.most_common(20))
print(words["'ll"])
"""
words=set(words)

dic={}
rev_dic={}
n=0
for i in words:
    dic[i]=n
    rev_dic[n]=i
    n+=1
#print(dic)

data=np.zeros((len(song_words),len(dic)))
n=0
for song in song_words:
    for i in song:
        data[n][dic[i]]+=1
    n+=1
#[print(d) for d in data]

kmeans = KMeans(n_clusters=6, random_state=0).fit(data)
#print(kmeans.labels_)

"""
for i in range(0,len(kmeans.labels_)):
    if(kmeans.labels_[i]==1):
        w=song_words[i][sorted(range(n), key=lambda t: data[i][t])[-5:]]
        #w=song_words[i][data.indexof(max(data[i]))]
        print(w,)
"""
for i in range(0,len(kmeans.labels_)):
    if(kmeans.labels_[i]==1):
        j=sorted(range(n), key=lambda t: data[i][t])[-5:]
        [print(data[i][k]) for k in j]
        [print(rev_dic[k],) for k in j]
