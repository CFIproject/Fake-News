import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download()
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
stemming = PorterStemmer()
from nltk.corpus import stopwords

df = pd.read_csv('fake_or_real_news.csv')
x = df['text']
y = df['label']

df = df.drop(columns = ['Unnamed: 0'],axis=1)
df = df.isnull().sum()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
train_x, cv_x, y_train, cv_y = train_test_split(x_train, y_train, test_size = 0.2)

print("In train = ",train_x.shape[0])
print("In test = ",x_test.shape[0])
print("In cv = ",cv_x.shape[0])

x = x.str.lower()

#Tokenization
def identify_tokens(row):
    text = row['text']
    tokens = nltk.word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]#numbers are not taken; for loop, if(no num,punc=true)-so only words are storde in token_words
    return token_words

df['words'] = df.apply(identify_tokens, axis=1)
df['words']

#Stemming
def stem_list(row):
    my_list = row['words']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

df['text'] = df.apply(stem_list, axis=1)

df = df.drop("words", axis=1)      
df = df.drop("stemmed_words", axis=1)      

# Removing Stop Words
stops = set(stopwords.words("english"))   

def remove_stops(row):
    my_list = row['text']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

df['text'] = df.apply(remove_stops, axis=1)
