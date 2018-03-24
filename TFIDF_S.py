from sklearn.metrics import classification_report
#import eli5
import numpy as np
import DataReader_S as DR
from sklearn import svm
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem import *
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier as NN
from sklearn.metrics import confusion_matrix
#from pandas_ml import ConfusionMatrix
from sklearn.decomposition import  LatentDirichletAllocation

#import seaborn as sns

song_type=['Angry','Happy','Sad','Relaxed']

#Read training data
Ang_Songs=DR.readData("Data-Set/Angry/Train/","angry")
Hap_Songs=DR.readData("Data-Set/Happy/Train/","happy")
Sad_Songs=DR.readData("Data-Set/Sad/Train/","sad")
Rel_Songs=DR.readData("Data-Set/Relaxed/Train/","relaxed")
SongsTrain=[Ang_Songs,Hap_Songs,Sad_Songs,Rel_Songs]

#ReadTestingData
AngT_Songs=DR.readData("Data-Set/Angry/Test/","angry")
HapT_Songs=DR.readData("Data-Set/Happy/Test/","happy")
SadT_Songs=DR.readData("Data-Set/Sad/Test/","sad")
RelT_Songs=DR.readData("Data-Set/Relaxed/Test/","relaxed")
SongsTTrain=[AngT_Songs,HapT_Songs,SadT_Songs,RelT_Songs]


SongsWordsTrain=[[],[]]
for i in range(4):
	for song in SongsTrain[i]:
		s=song[4]
		SongsWordsTrain[0].append(s)
		SongsWordsTrain[1].append(i)

SongsWordsTTrain=[[],[]]
for i in range(4):
	for song in SongsTTrain[i]:
		s=song[4]
		SongsWordsTTrain[0].append(s)
		SongsWordsTTrain[1].append(i)
     

#stemmer = PorterStemmer()
stemmer = SnowballStemmer("english",ignore_stopwords=True)
tokenizer = RegexpTokenizer("[\w’]+", flags=re.UNICODE)



def tokenize(text):
	#tokens = word_tokenize(text)
	tokens = tokenizer.tokenize(text)
	tokens = [t for t in tokens if len(t) > 2]  # remove short words, they're probably not useful
	tokens = [t for t in tokens if not re.search(r"^'", t)]
	tokens = [t for t in tokens if not re.search(r"\\.+", t)]
	stems = []
	for item in tokens:
		stems.append(stemmer.stem(item))
	return stems



def print_top50(vectorizer, clf, class_labels,n=10):
	#Prints features with the highest coefficient values, per class
	feature_names = vectorizer.get_feature_names()
	for i, class_label in enumerate(class_labels):
		top50 = np.argsort(clf.coef_[i])[-n:]
		print(song_type[int(class_label)])
		print(".............")
		for j in top50:
				print(feature_names[j],clf.coef_[0][j])
		print(".............")


#print(SongWordsTrain)
#print(SongsTrain[:][3])

vectorizer = TfidfVectorizer(tokenizer=tokenize,min_df=2, ngram_range = (1,3), sublinear_tf = True, stop_words = "english")

print ("Vectorizing training...")

train_x = vectorizer.fit_transform(SongsWordsTrain[0])

"""
print(vectorizer.vocabulary_)
print(train_x)
print(train_x.shape)
"""
print(train_x.getnnz())
print ("Vectorizing test...")

#for i in vectorizer.get_feature_names():
#	print(i)
print(vectorizer.get_feature_names())

test_x = vectorizer.transform(SongsWordsTTrain[0])
#print(test_x)

print ("Training...")

print ("\nNB...")
modelA = MultinomialNB(alpha=1.0)
modelA.fit(train_x, SongsWordsTrain[1])

#print top50 features
print_top50(vectorizer,modelA,modelA.classes_,50)

#score on test set
print(modelA.score(test_x,SongsWordsTTrain[1]))
predict=modelA.predict(test_x)

print("\n\n")
"""
confusion_matrix = ConfusionMatrix(SongsWordsTTrain[1], predict)
print(confusion_matrix)
confusion_matrix.plot()
confusion_matrix.print_stats()
plt.show()
"""
#print(classification_report(SongsWordsTTrain[1],predict,SongsWordsTrain[1]))


print ("\nSVM...")
modelB = svm.SVC(kernel='linear', C=1, gamma=1) 
modelB.fit( train_x, SongsWordsTrain[1])

#print(modelB.predict(test_x))\

#score on test set
print(modelB.score(test_x,SongsWordsTTrain[1]))




print ("LR...")
modelC = LR(multi_class='multinomial',solver='newton-cg')
modelC.fit( train_x, SongsWordsTrain[1])
#print(modelC.predict(test_x))

#score on test set
print(modelC.score(test_x,SongsWordsTTrain[1]))




print ("KNN...")
knn = KNeighborsClassifier()
knn.fit(train_x,SongsWordsTrain[1])

#score on test set
print(knn.score(test_x,SongsWordsTTrain[1]))










