import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download()
from nltk.tokenize import word_tokenize

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

def identify_tokens(row):
    text = row['text']
    tokens = nltk.word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]#numbers are not taken; for loop, if(no num,punc=true)-so only words are storde in token_words
    return token_words

df['words'] = df.apply(identify_tokens, axis=1)
df['words']
