import DataReader as DR
from sklearn import svm
from nltk.tokenize import word_tokenize, RegexpTokenizer
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
from pandas_ml import ConfusionMatrix
from sklearn.decomposition import LatentDirichletAllocation
import DataReaderHindi as DRH

# read hindi dataset
AngH_Songs = DRH.readData("Hindi_Dataset/Angry/", 'A')
HapH_Songs = DRH.readData("Hindi_Dataset/Happy/", 'H')
SadH_Songs = DRH.readData("Hindi_Dataset/Sad/", 'S')
RelH_Songs = DRH.readData("Hindi_Dataset/Relaxed/", 'R')
SongsHindi = [AngH_Songs, HapH_Songs, SadH_Songs, RelH_Songs]

SongsHindiTest = [[], []]
for i in range(4):
    for song in SongsHindi[i]:
        s = song[5]
        SongsHindiTest[0].append(s)
        SongsHindiTest[1].append(i)

# import seaborn as sns

# Read training data
Ang_Songs = DR.readData("Data-Set/Angry/Train/", "angry")
Hap_Songs = DR.readData("Data-Set/Happy/Train/", "happy")
Sad_Songs = DR.readData("Data-Set/Sad/Train/", "sad")
Rel_Songs = DR.readData("Data-Set/Relaxed/Train/", "relaxed")
SongsTrain = [Ang_Songs, Hap_Songs, Sad_Songs, Rel_Songs]


SongsWordsTrain = [[], []]
for i in range(4):
    for song in SongsTrain[i]:
        s = song[4]
        SongsWordsTrain[0].append(s)
        SongsWordsTrain[1].append(i)


# stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer("[\w’]+", flags=re.UNICODE)


def tokenize(text):
    # tokens = word_tokenize(text)
    tokens = tokenizer.tokenize(text)
    stems = []
    for item in tokens:
        stems.append(stemmer.stem(item))
    return stems


# print(SongWordsTrain)
# print(SongsTrain[:][3])

vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=2, ngram_range=(1, 3), sublinear_tf=True, stop_words="english")

print("Vectorizing training...")

train_x = vectorizer.fit_transform(SongsWordsTrain[0])
# print(train_x)
print(train_x.shape)

print(train_x.getnnz())
print("Vectorizing test...")
# print(vectorizer.get_feature_names())


test_hindi = vectorizer.transform(SongsHindiTest[0])

print("Training...")

print("NB...")
modelA = MultinomialNB()
modelA.fit(train_x, SongsWordsTrain[1])
print("Hindi...")

print(modelA.score(test_hindi, SongsHindiTest[1]))

print("...")
predict = modelA.predict(test_hindi)

confusion_matrix = ConfusionMatrix(SongsHindiTest[1], predict)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,xticklabels=SongsWordsTrain[1], yticklabels=SongsWordsTTrain[1])
print(confusion_matrix)
"""confusion_matrix.plot()
confusion_matrix.print_stats()
plt.show()
"""

print("SVM...")
modelB = svm.SVC(kernel='linear', C=1, gamma=1)
modelB.fit(train_x, SongsWordsTrain[1])
# print(modelB.predict(test_x))
print("Hindi...")

print(modelB.score(test_hindi, SongsHindiTest[1]))

print("...")

print("LR...")
modelC = LR(multi_class='multinomial', solver='newton-cg')
modelC.fit(train_x, SongsWordsTrain[1])
# print(modelC.predict(test_x))

print("Hindi...")

print(modelC.score(test_hindi, SongsHindiTest[1]))

print("...")


print("KNN...")
knn = KNeighborsClassifier()
knn.fit(train_x, SongsWordsTrain[1])

print("Hindi...")

print(knn.score(test_hindi, SongsHindiTest[1]))

print("...")
