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

#Ang_Songs=DR.readData("Data-Set/Angry/Train/","angry")
Hap_Songs=DR.readData("Data-Set/Happy/Train/","happy")
Sad_Songs=DR.readData("Data-Set/Sad/Train/","sad")
#Rel_Songs=DR.readData("Data-Set/Relaxed/Train/","relaxed")

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

words=[]
HappySongWords=[]
for song in Hap_Songs[:1000]:
    s=song[4]
    s=my_tokenizer(s)
    HappySongWords.append(s)
    for i in s:
        if i not in words:
            words.append(i)

SadSongWords=[]
for song in Sad_Songs[:1000]:
    s=song[4]
    s=my_tokenizer(s)
    SadSongWords.append(s)
    for i in s:
        if i not in words:
            words.append(i)

#print(HappySongWords)
#print(SadSongWords)

dic={}
rev_dic=[]
n=0
for i in words:
    dic[i]=n
    rev_dic.append(i)
    n+=1

#print(len(dic))
#print(len(rev_dic))


HapLen=len(HappySongWords)
l=len(rev_dic)
data=np.zeros((HapLen+len(SadSongWords),l+1))
n=0
for song in HappySongWords:
    for i in song:
        data[n][dic[i]]+=1
        data[n][-1]=0
    n+=1

for song in SadSongWords:
    for i in song:
        data[n][dic[i]]+=1
        data[n][-1]=1
    n+=1

for d in data:
    print(d)

np.random.shuffle(data)
X = data[:,:l]
Y=data[:,-1]

Xtrain = X[:-150,]
Ytrain = Y[:-150,]
Xtest = X[-50:,]
Ytext = Y[-50:,]

print(Xtrain)
print(Ytrain)


model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("Classification rate for NB: ",model.score(Xtest,Ytext))