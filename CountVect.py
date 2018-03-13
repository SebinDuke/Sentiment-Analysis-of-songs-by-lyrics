import DataReader as DR
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier


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


vectorizer = CountVectorizer(min_df=2,ngram_range = (1,3))
print ("Vectorizing training...")

train_x = vectorizer.fit_transform(SongsWordsTrain[0])
#print(train_x)
print(train_x.shape)
print(train_x.getnnz())
print ("Vectorizing test...")
#print(vectorizer.get_feature_names())

test_x = vectorizer.transform(SongsWordsTTrain[0])
#print(test_x)
print ("Training...")
print ("NB...")
modelA = MultinomialNB()
modelA.fit(train_x, SongsWordsTrain[1])
#score on training set
print(modelA.score(train_x, SongsWordsTrain[1])) 
#print(modelA.predict(test_x))

#score on testing set
print(modelA.score(test_x,SongsWordsTTrain[1]))


print ("SVM...")
modelB = svm.SVC(kernel='linear', C=1, gamma=1) 
modelB.fit( train_x, SongsWordsTrain[1])
#score on training set
print(modelB.score(train_x, SongsWordsTrain[1]))
#print(modelB.predict(test_x))


#score on testing set
print(modelB.score(test_x,SongsWordsTTrain[1]))
"""
print ("LR...")
modelC = LR(multi_class='multinomial',solver='newton-cg')
modelC.fit( train_x, SongsWordsTrain[1])
modelC.fit( train_x, SongsWordsTrain[1])
#score on training set
print(modelC.score(train_x, SongsWordsTrain[1]))
#print(modelC.predict(test_x))

#score on testing set
print(modelC.score(test_x,SongsWordsTTrain[1]))
"""
print ("LR...")
modelC = LR(multi_class='multinomial',solver='newton-cg')
modelC.fit( train_x, SongsWordsTrain[1])
#score on training set
print(modelC.score(train_x, SongsWordsTrain[1]))
#print(modelC.predict(test_x))

#score on test set
print(modelC.score(test_x,SongsWordsTTrain[1]))

print ("KNN...")
knn = KNeighborsClassifier()
knn.fit(train_x,SongsWordsTrain[1])
#score on training set
print(knn.score(train_x, SongsWordsTrain[1]))

#score on test set
print(knn.score(test_x,SongsWordsTTrain[1]))

"""

lda = LatentDirichletAllocation(n_components=10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(train_x,SongsWordsTrain[0])
print(lda.score(train_x, SongsWordsTrain[1]))
print(lda.score(test_x,SongsWordsTTrain[1]))
"""