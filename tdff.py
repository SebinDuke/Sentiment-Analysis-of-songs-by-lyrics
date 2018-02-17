
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
from nltk.stem import *
import math
import DataReader as DR

Rel_Songs=DR.readData("Data-Set/Relaxed/Train/","relaxed")
#Hap_Songs=DR.readData("Data-Set/Happy/Train/","happy")
#Sad_Songs=DR.readData("Data-Set/Sad/Train/","sad")

stopwords = nltk.corpus.stopwords.words('english')
tokenizer = RegexpTokenizer("[\w’]+", flags=re.UNICODE)
#stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
#lemmatizer=WordNetLemmatizer()



def freq(word, doc):
    return doc.count(word)


def word_count(doc):
    return len(doc)


def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))


def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    return math.log(len(list_of_docs) / float(num_docs_containing(word, list_of_docs)))


def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))

#Compute the frequency for each term.
vocabulary = []
docs = {}
all_tips = []
for tip in (Rel_Songs):
    tokens = tokenizer.tokenize(tip[4])
    #bi_tokens = bigrams(tokens)
    #tri_tokens = trigrams(tokens)
    tokens = [token.lower() for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in stopwords]
    #tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    
    """    	
    bi_tokens = [' '.join(token).lower() for token in bi_tokens]
    bi_tokens = [token for token in bi_tokens if token not in stopwords]

    tri_tokens = [' '.join(token).lower() for token in tri_tokens]
    tri_tokens = [token for token in tri_tokens if token not in stopwords]
    """
    final_tokens = []
    final_tokens.extend(tokens)
    #final_tokens.extend(bi_tokens)
    #final_tokens.extend(tri_tokens)
    docs[tip[0]] = {'freq': {}, 'tf': {}, 'idf': {},'tf-idf': {}, 'tokens': []}	
    for token in final_tokens:
        #The frequency computed for each tip

        docs[tip[0]]['freq'][token] = freq(token, final_tokens)
        #The term-frequency (Normalized Frequency)
        docs[tip[0]]['tf'][token] = tf(token, final_tokens)
        docs[tip[0]]['tokens'] = final_tokens

    vocabulary.append(final_tokens)

for doc in docs:
    for token in docs[doc]['tf']:
        #The Inverse-Document-Frequency
        docs[doc]['idf'][token] = idf(token, vocabulary)
        #The tf-idf
        docs[doc]['tf-idf'][token] = tf_idf(token, docs[doc]['tokens'], vocabulary)

#Now let's find out the most relevant words by tf-idf.
words = {}
for doc in docs:
    for token in docs[doc]['tf-idf']:
        if token not in words:
            words[token] = docs[doc]['tf-idf'][token]
        else:
            if docs[doc]['tf-idf'][token] > words[token]:
                words[token] = docs[doc]['tf-idf'][token]
    """	
    print (doc)
    for token in docs[doc]['tf-idf']:
        print (token, docs[doc]['tf-idf'][token])
    """	

#Now let's find out the most relevant words by tf-idf.
freq_words = {}
for doc in docs:
    for token in docs[doc]['freq']:
        if token not in freq_words:
            freq_words[token] = docs[doc]['freq'][token]
        else:
            if docs[doc]['freq'][token] > freq_words[token]:
                freq_words[token] = docs[doc]['freq'][token]


# top 5 words acc to tf-idf
i=0
for item in sorted(words.items(), key=lambda x: x[1], reverse=True):
	#if i == 5:
	#	break    
	print ("%f <= %s" % (item[1], item[0]))
	#i+=1

"""
# top 5 words acc to freq	
i=0
for item in sorted(freq_words.items(), key=lambda x: x[1], reverse=True):
	if i == 5:
		break    
	print ("%f <= %s" % (item[1], item[0]))
	i+=1	
"""
