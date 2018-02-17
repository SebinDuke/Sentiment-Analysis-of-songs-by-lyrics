import DataReader as DR
from nltk.tokenize import word_tokenize
#from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#import random
import numpy as np
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB

#Read training data
Ang_Songs=DR.readData("Data-Set/Angry/Train/","angry")
Hap_Songs=DR.readData("Data-Set/Happy/Train/","happy")
Sad_Songs=DR.readData("Data-Set/Sad/Train/","sad")
Rel_Songs=DR.readData("Data-Set/Relaxed/Train/","relaxed")
SongsTrain=[Ang_Songs,Hap_Songs,Sad_Songs,Rel_Songs]

#    PROCESSING TRAINING DATA

#tokenizing training data
sw = list(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()

def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in sw] # remove stopwords
    tokens = [t for t in tokens if not re.search(r"^'",t)]
    tokens = [t for t in tokens if not re.search(r"\\.+",t)]
    tokens = [t for t in tokens if not re.search(r".*\\x\d\d.*",t)] #NOT WORKING
    return tokens

SongWordsTrain=[[],[],[],[]]
WordsByClass=[[],[],[],[]]
for i in range(4):
    for song in SongsTrain[i]:
        s=song[4]
        s=my_tokenizer(s)
        SongWordsTrain[i].append(s)
        for j in s:
            WordsByClass[i].append(j)

print(WordsByClass)
