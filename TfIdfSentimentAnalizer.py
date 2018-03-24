from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#import random
import numpy as np
import pandas as pd
import re
import csv
from sklearn.naive_bayes import MultinomialNB

#Read training data
SongsTrain=[[],[],[],[]]
emotionToNum={"angry":0,"happy":1,"sad":2,"relaxed":3}
with open("FullDataSet/Train.csv","r") as file:
    reader=csv.reader(file)
    for row in reader:
        i=emotionToNum[row[4]]
        SongsTrain[i].append(row)

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

WordsByClass=[[],[],[],[]]

for i in range(4):
    for song in SongsTrain[i]:
        s=song[5]
        s=my_tokenizer(s)
        for j in s:
            WordsByClass[i].append(j)
#print(WordsByClass)

NoOfWords=[]
for i in WordsByClass:
    NoOfWords.append(len(i))

#print(NoOfWords)

TF=[[],[],[],[]]
for i in range(4):
    TF[i]=FreqDist(WordsByClass[i])
    #print(len(TF[i]))
    """
    for j in TF[i].keys():
        print(j,TF[i][j])
        """


IDF=[{},{},{},{}]
for i in range(4):
    for word in TF[i].keys():
        ct=0
        for j in range(4):
            if word in TF[j].keys():
                ct+=1
        idf=16/(ct**2)
        IDF[i][word]=idf

WordImp=[[],[],[],[]]
for i in range(4):
    for word in TF[i].keys():
        WordImp[i].append((word,IDF[i][word]*TF[i][word]/NoOfWords[i]))


ClassNames=("angry","happy","sad","relaxed")
ClassifiactionWords=[{},{},{},{}]
for i in range(4):
    #print("Top hundred most Importent words in class "+ClassNames[i]+" and their TF-IDF scores are:")
    k=0
    for j in sorted(WordImp[i],key=lambda imp: imp[1],reverse=True):
        #print(j)
        ClassifiactionWords[i][j[0]]=j[1]
        k+=1
        if(k==5000):
            break

#print(ClassifiactionWords)

#Read Testinging Data
SongsTest=[]
with open("PartialDataSet/Test.csv","r") as file:
    reader=csv.reader(file)
    for row in reader:
        SongsTest.append(row)

def predictSong(song):
    score=[0,0,0,0]
    for i in song:
        for j in range(4):
            if i in ClassifiactionWords[j].keys():
                score[j]+=ClassifiactionWords[j][i]
    index=score.index(max(score))
    return index

accuracy=0
for song in SongsTest:
    s=song[5]
    s=my_tokenizer(s)
    prediction=predictSong(s)
    prediction=ClassNames[prediction]
    print(song[1],song[2],song[4],prediction)
    
    if(song[4]==prediction):
        accuracy+=1

print(accuracy)     #(accuracy/100)*100=accuracy
