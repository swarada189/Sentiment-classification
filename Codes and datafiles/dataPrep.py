import os
import sys

import pandas


def get_phrase_sentiments(base_directory):
    def group_labels(label):
        if label in ["very negative", "negative"]:
            return "negative"
        elif label in ["positive", "very positive"]:
            return "positive"
        else:
            return "neutral"

    dictionary = pandas.read_csv(os.path.join(base_directory, "dictionary.txt"), sep="|")
    dictionary.columns = ["sentence", "id"]
    dictionary = dictionary.set_index("id")

    sentiment_labels = pandas.read_csv(os.path.join(base_directory, "sentiment_labels.txt"), sep="|")
    sentiment_labels.columns = ["id", "sentiment"]
    sentiment_labels = sentiment_labels.set_index("id")

    phrase_sentiments = dictionary.join(sentiment_labels)

    phrase_sentiments["fine"] = pandas.cut(phrase_sentiments.sentiment, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                           include_lowest=True,
                                          labels=["very negative", "negative", "neutral", "positive", "very positive"])
    phrase_sentiments["coarse"] = phrase_sentiments.fine.apply(group_labels)
    #phrase_sentiments.drop("fine", axis=1)
    return phrase_sentiments.set_index("sentence")


def get_sentence_partitions(base_directory):
    sentences = pandas.read_csv(os.path.join(base_directory, "datasetSentences.txt"), index_col="sentence_index",
                                sep="\t")
    #splits = pandas.read_csv(os.path.join(base_directory, "datasetSplit.txt"), index_col="sentence_index")
    #sentences=sentences.join(splits)
    sentences["sentence"]=sentences["sentence"].str.replace("-LRB-","(")
    sentences["sentence"]=sentences["sentence"].str.replace("-RRB-",")")
    return sentences.set_index("sentence")


def combine_data(base_directory):
    phrase_sentiments = get_phrase_sentiments(base_directory)
    #print phrase_sentiments.sample(n=2)
    sentence_partitions = get_sentence_partitions(base_directory)
    #print sentence_partitions.sample(n=2)
    data = sentence_partitions.join(phrase_sentiments)
    data['coarse'] = data['coarse'].fillna('neutral')	
    print data.sample(n=2)
    return data


base_directory = sys.argv[1]			#specify path of Current working directory
print base_directory
data = combine_data(base_directory)
filename = os.path.join(base_directory, "stanford-sentiment-treebank.csv")
data.to_csv(filename)

