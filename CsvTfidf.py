import csv
import re
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn import svm
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem import PorterStemmer,SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression as LR
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier

TrainData=[]
SongsWordsTrain=[]
with open("PartialDataSet/Train.csv","r") as file:
    reader=csv.reader(file)
    for row in reader:
        #print(row)
        TrainData.append(row)
#print(TrainData)

TestData=[]
with open("PartialDataSet/Test.csv","r") as file:
    reader=csv.reader(file)
    for row in reader:
        #print(row)
        TestData.append(row)
#print(TestData)

sw = list(stopwords.words("english"))
#stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer("[\w’]+", flags=re.UNICODE)

def tokenize(s):
    s = s.lower() # downcase
    #tokens = word_tokenize(s) # split string into words (tokens)
    tokens = tokenizer.tokenize(s)
    tokens = [stemmer.stem(t) for t in tokens] # put words into base form
    """
    tokens = [t for t in tokens if t not in sw] # remove stopwords
    tokens = [t for t in tokens if len(t) > 1] # remove short words, they're probably not useful
    tokens = [t for t in tokens if not re.search(r"^'",t)]
    tokens = [t for t in tokens if not re.search(r"\.+",t)]
    tokens = [t for t in tokens if not re.search(r".*\\x\d\d.*",t)] #NOT WORKING
    """
    return tokens


emotionToNum={"angry":0,"happy":1,"sad":2,"relaxed":3}

TrainSongs=[[],[]]
for song in TrainData:
    TrainSongs[0].append(song[5])
    TrainSongs[1].append(emotionToNum[song[4]])
#print(TrainSongs)

TestSongs=[[],[]]
for song in TestData:
    TestSongs[0].append(song[5])
    TestSongs[1].append(emotionToNum[song[4]])
#print(TestSongs)


vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=3, ngram_range = (1,2), sublinear_tf =True, stop_words = "english")
train_x = vectorizer.fit_transform(TrainSongs[0])
test_x=vectorizer.transform(TestSongs[0])
WordList=vectorizer.get_feature_names()
#print(WordList)


print(train_x.shape)
#print(train_x)
#print(train_x.sorted_indices())
#print(train_x.max(axis=1))
#print(train_x[0,0])

modelA = MultinomialNB()
modelA.fit(train_x,TrainSongs[1])
print("Accuracy of MultinomialNB:")
print(modelA.score(test_x,TestSongs[1]))

modelB = svm.SVC(kernel='linear', C=1, gamma=1) 
modelB.fit( train_x,TrainSongs[1])
print("Accuracy of SVM:")
print(modelB.score(test_x,TestSongs[1]))

modelC = LR(multi_class='multinomial',solver='newton-cg')
modelC.fit( train_x,TrainSongs[1])
print("Accuracy of Logistic Regression:")
print(modelC.score(test_x,TestSongs[1]))

"""
modelD = GaussianNB()
Train_X=train_x.toarray()
Test_X=test_x.toarray()
modelD.fit(Train_X,TrainSongs[1])
print("Accuracy of GaussianNB:")
print(modelD.score(Test_X,TestSongs[1]))
"""