

import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import *
import numpy as np
from sklearn.model_selection import StratifiedKFold	#cross-validation
from sklearn.decomposition import LatentDirichletAllocation as LDA

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


'''
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')		#scikit-learn's bag of words tool.
count_vect.fit(df['sentence'])

# transform the training data using count vectorizer object
X_count =  count_vect.transform(X)

print "Type of feature vector : " + str(type(X_count))
print "Size of feature vector : "+ str(X_count.shape)
'''
n_features = 21701
n_topics = 10
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(df['sentence'])


print("Topic modelling with LDA...")
lda = LDA(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

X_lda = lda.fit_transform(tf)
print lda.components_
# so lda_x is your doc-topic distribution that you can use for feature vector to your SVM model.
# lda.components_ is your topic-word distribution.

#combine using numpyhstack (BOW and lda features)

cv = StratifiedKFold(n_splits=2)			#perform 10 folds cross-validation

print "-----------OUTPUT OF CV (USING BagOfWords)--------"
i = 1
for train, test in cv.split(X_lda, y):
	print "For fold = "+ str(i)		
	accuracy = train_model(svm.SVC(kernel='linear'), X_lda[train], y[train], X_lda[test] ,y[test])
	print "Bag of words accuracy: ", accuracy	
	i+=1	


