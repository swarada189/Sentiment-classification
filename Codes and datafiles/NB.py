

import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import *
import numpy as np
from sklearn.model_selection import StratifiedKFold	#cross-validation
 

def train_model(classifier, feature_vector_train, train_label, feature_vector_dev, dev_label,is_neural_net=False):
	# fit the training dataset on the classifier
	classifier.fit(feature_vector_train, train_label)
	
	#print "parameters for NB classifier: "
	#print classifier.get_params()

	# predict the labels on validation dataset
	predictions = classifier.predict(feature_vector_dev)	

	if is_neural_net:
		predictions = predictions.argmax(axis=-1)

	return metrics.accuracy_score(predictions, dev_label)



#----------------------------------------------------------------------------------------------------------
df=pd.read_csv("stanford-sentiment-treebank.csv")
data=df.values
X=data[:,0]						#np array => 1st column of data dataframe
y=data[:,3]

encoder = preprocessing.LabelEncoder()			#Encode labels with value between 0 and n_classes-1.	
y = encoder.fit_transform(y)				#Fit label encoder and return encoded labels


# FOR TF-IDF INPUT VECTOR
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df['sentence'])
X_tfidf =  tfidf_vect.transform(X)

#FOR N-GRAM VECTORS
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(df['sentence'])
X_tfidf_ngram =  tfidf_vect_ngram.transform(X)


cv = StratifiedKFold(n_splits=10)			#perform 10 folds cross-validation

print "-----------OUTPUT OF CV (USING TF-IDF)--------"
i = 1
for train, test in cv.split(X_tfidf, y):
	print "For fold = "+ str(i)		
	accuracy = train_model(naive_bayes.MultinomialNB(), X_tfidf[train], y[train], X_tfidf[test] ,y[test])
	print "WordLevel TF-IDF: ", accuracy	
	i+=1	

print "-----------OUTPUT OF CV (USING N-GRAM)--------"
i = 1
for train, test in cv.split(X_tfidf_ngram, y):
	print "For fold = "+ str(i)		
	accuracy = train_model(naive_bayes.MultinomialNB(), X_tfidf_ngram[train], y[train], X_tfidf_ngram[test] ,y[test])
	print "N-Gram vectors: ", accuracy	
	i+=1	


