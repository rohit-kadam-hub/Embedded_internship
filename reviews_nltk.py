import nltk

from nltk.corpus import stopwords
import pandas as pd
stopwords_english = stopwords.words('english')

def format_sentence(sent):
    return ({word: True for word in nltk.word_tokenize(sent)})
posdataset= pd.read_csv("Positive_words.csv")
negdataset=["worst","poor","bad","pathetic","poor"]
flag=False
review=input("Enter the product review")
if("not" in review):
    flag=True
print(review)
data=nltk.word_tokenize(review)
print(data)

all_words_without_stopwords = [word for word in data if word not in stopwords_english]
print(all_words_without_stopwords)
processedreview=""
for word in all_words_without_stopwords:
    processedreview=processedreview+word+" "
print(processedreview)

pos=[]
for word in posdataset:
    pos.append([format_sentence(word), 'positive'])
print (pos)

neg = []
for word in negdataset:
    neg.append([format_sentence(word), 'negative'])
print (neg)

training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

from nltk.classify import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(training)
classifier.show_most_informative_features()

result=classifier.classify(format_sentence(processedreview))

if(flag):
    if(result=="positive"):
        result="negative"
    elif(result=="negative"):
        result="positive"

print("The analysis of",review," is ",result)

# is of the on am was not for to of

# good bad nice worst beautiful
