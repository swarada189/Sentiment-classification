#This is a data analysis code that prints results of analysis(For part I)
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


#Average sentence length
df=pd.read_csv("stanford-sentiment-treebank.csv")
allsentList=df['sentence'].tolist()
avg_len=sum(len(sent.split()) for sent in allsentList) /len(allsentList)
allsentString=''.join(allsentList)			#this is my wordlist
print"average sentence length = "+ str(avg_len)


#data pre-processing code (Removal of stopwords)
'''
allsentString=allsentString.split()
print type(allsentString)
#filtered_words=[word for word in allsentString if word not in stopwords.words('english')]

filtered_word_list = allsentString[:] #make a copy of the word_list
print type(filtered_word_list)
for word in allsentString: # iterate over word_list
  if word.lower() in stopwords.words('english'): 
    filtered_word_list.remove(word)
print type(filtered_words)
vocab=set(word_tokenize(filtered_words))
'''

#Overall vocabulary size
vocab=set(word_tokenize(allsentString))
print "Overall vocabulary size ="+ str(len(vocab))


#Average pos/neg sentence length
pos_rows=df.loc[df['coarse']=='positive']
neg_rows=df.loc[df['coarse']=='negative']
pos_list=pos_rows['sentence'].tolist()
neg_list=neg_rows['sentence'].tolist()
avg_poslen=sum(len(sent.split()) for sent in pos_list) /len(pos_list)
avg_neglen=sum(len(sent.split()) for sent in neg_list) /len(neg_list)
print"average positive sentence length = "+ str(avg_poslen)
print"average negative sentence length = "+ str(avg_neglen)


#Overall pos/neg sentence vocab
wordlist= [word_tokenize(sent) for sent in pos_list]		#to be changed according abve logic
vocab=set(map(tuple,wordlist))
print "Positive sentence vocab = "+ str(len(vocab))
wordlist= [word_tokenize(sent) for sent in neg_list]
vocab=set(map(tuple,wordlist))
print "Negative sentence vocab " + str(len(vocab))

