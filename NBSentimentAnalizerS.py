import DataReader as DR
from sklearn import svm
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

words=[]
SongWordsTrain=[[],[],[],[]]
for i in range(4):
    for song in SongsTrain[i]:
        s=song[4]
        s=my_tokenizer(s)
        SongWordsTrain[i].append(s)
        for j in s:
            if j not in words:
                words.append(j)

#print(SongWordsTrain)

#Building dictionary
dic={}
rev_dic=[]
n=0
for i in words:
    dic[i]=n
    rev_dic.append(i)
    n+=1

#print(dic)
#print(len(rev_dic))

#converting tokenized words to word counts
ListLen=len(SongWordsTrain[0])+len(SongWordsTrain[1])+len(SongWordsTrain[2])+len(SongWordsTrain[3])
l=len(rev_dic)
data=np.zeros((ListLen,l+1))
n=0
for i in range(4):
    for song in SongWordsTrain[i]:
        for j in song:
            data[n][dic[j]]+=1
       	
        n+=1

#for d in data:
#    print(d)
np.random.shuffle(data)
X = data[:,:l]
Y=data[:,-1]

#Train MultinomialNB model with data
model = MultinomialNB()
model.fit(X,Y)

model1 = svm.SVC(kernel='linear', C=1, gamma=1) 
model1.fit(X,Y)

#Read test data
Ang_Songs=DR.readData("Data-Set/Angry/Test/","angry")
Hap_Songs=DR.readData("Data-Set/Happy/Test/","happy")
Sad_Songs=DR.readData("Data-Set/Sad/Test/","sad")
Rel_Songs=DR.readData("Data-Set/Relaxed/Test/","relaxed")
SongsTest=[Ang_Songs,Hap_Songs,Sad_Songs,Rel_Songs]

#   PROCESS TEST DATA

#tokenizing testing data
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

SongWordsTest=[[],[],[],[]]
for i in range(4):
    for song in SongsTest[i]:
        s=song[4]
        s=my_tokenizer(s)
        SongWordsTest[i].append(s)

#print(SongWordsTrain)
#No need to build dictionary here

#converting tokenized words to word counts using only word from training dictionary 
ListLen=len(SongWordsTest[0])+len(SongWordsTest[1])+len(SongWordsTest[2])+len(SongWordsTest[3])
TestData=np.zeros((ListLen,l+1))
n=0
for i in range(4):
    for song in SongWordsTest[i]:
        for j in song:
            if j in dic:
                TestData[n][dic[j]]+=1
        TestData[n][-1]=i
        n+=1

#for d in data:
#    print(d)

#np.random.shuffle(data)
x =TestData[:,:l]
y=TestData[:,-1]



#Test the model using testing data
prediction=model.predict(x)
print("Classification rate for NB: ",model.score(X,Y))

print("Classification rate for NB: ",model.score(x,y))

prediction=model1.predict(x)
print("Classification rate for SVM: ",model1.score(X,Y))

print("Classification rate for SVM: ",model1.score(X,Y))

"""
k=0
for i in range(4):
    for j in range(len(SongsTest[i])):
        if(y[k]==prediction[k]):
            print("Correct: ",SongsTest[i][j][0],SongsTest[i][j][1])
        else:
            print("Inorrect:",SongsTest[i][j][0],SongsTest[i][j][1])
        k+=1
"""
