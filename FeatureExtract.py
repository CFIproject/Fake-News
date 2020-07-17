import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download()
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('fake_or_real_news.csv')

df = df.drop(columns = ['Unnamed: 0'],axis=1)

df['text'] = df['text'].str.lower()

#Tokenization
def identify_tokens(row):
    text = row['text']
    tokens = nltk.word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]#numbers are not taken; for loop, if(no num,punc=true)-so only words are storde in token_words
    return token_words

df['words'] = df.apply(identify_tokens, axis=1)
df

#Stemming
stemming = PorterStemmer()

def stem_list(row):
    my_list = row['words']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

df['stemmed_words'] = df.apply(stem_list, axis=1)
df

# Removing Stop Words
stops = set(stopwords.words("english"))   

def remove_stops(row):
    my_list = row['stemmed_words']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

df['stem_meaningful'] = df.apply(remove_stops, axis=1)
df


def rejoin_words(row):
    my_list = row['stem_meaningful']
    joined_words = ( " ".join(my_list))
    return joined_words

df['processed'] = df.apply(rejoin_words, axis=1)

cols_to_drop = ['text', 'words', 'stemmed_words', 'stem_meaningful']
df.drop(cols_to_drop, inplace=True,axis=1)

x = df['processed']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size = 0.2)
x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size = 0.2)

print("In train = ",x_train.shape[0])
print("In test = ",x_test.shape[0])
print("In cv = ",x_cv.shape[0])

df

#***************************************************************************************************************************************************************

#Since we cannot pass the text data directly to the machine learning classifiers as they take vectors of numbers as input, 
#I converted the text into numbers by building a Bag-of-Words model first with CounterVectorizer and then with TfidfVectorizer.

#Now, to convert the text data into word-count-vectors I used CountVectorizer method which provides a simple way to both tokenize
#a collection of text documents and to construct a vocabulary of known words, also it is used to encode new documents using that vocabulary.

#Created an instance of the CountVectorizer class,

#Called the fit_transform() function in order to tokenize, build the vocabulary and encoded the training dataset which returned an 
#encoded vector with a length of entire vocabulary and integer count for the number of times each word appeared in the document.

#Term frequency (or word count) has already been calculated by CountVectorizer method, now to calculate the inverse document frequency (idf):

#Created an instance of TfidfTransformer,

#Used the word-count-vectors generated by CountVectorizer method with fit_transform function.

#The word-count-vectors generated by this method were then used to generate tfidf-ngram features.

#******************************************************************************************************************************************************************


# Convert text to word count vectors with CountVectorizer
# create the transform
cvec = CountVectorizer()

# tokenize, build vocab and encode training data
traindata_cvec = cvec.fit_transform(x_train['processed'].values)

# summarize
print(cvec.vocabulary_)
# print(cvec.get_feature_names())

print(traindata_cvec.shape)

# tfidf + ngrams
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), use_idf=True, smooth_idf=True)
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(x_train['processed'].values)

first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]


# place tf-idf values in a pandas data frame
daf = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
daf.sort_values(by=["tfidf"],ascending=False)
