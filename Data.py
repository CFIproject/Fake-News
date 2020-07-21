import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download()
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

df = pd.read_csv('fake_or_real_news.csv')
x = df['text']
y = df['label']

df = df.drop(columns = ['Unnamed: 0'],axis=1)

print(df.isnull().sum())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size = 0.2)

print("In train = ",x_train.shape[0])
print("In test = ",x_test.shape[0])
print("In cv = ",x_cv.shape[0])

df['text'] = df['text'].str.lower()

#Tokenization
def identify_tokens(row):
    text = row['text']
    tokens = nltk.word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]#numbers are not taken; for loop, if(no num,punc=true)-so only words are storde in token_words
    return token_words

df['text'] = df.apply(identify_tokens, axis=1)
df

#Stemming
stemming = PorterStemmer()

def stem_list(row):
    my_list = row['text']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

df['text'] = df.apply(stem_list, axis=1)
df

# Removing Stop Words
stops = set(stopwords.words("english"))   

def remove_stops(row):
    my_list = row['text']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

df['text'] = df.apply(remove_stops, axis=1)
df


 #DataFlair - Initialize a PassiveAggressiveClassifier
    #from sklearn.linear_model import PassiveAggressiveClassifier
    
    #pac=PassiveAggressiveClassifier(max_iter=50)
    #pac.fit(tfidf_vectorizer,y_train)
    #DataFlair - Predict on the test set and calculate accuracy
    #y_pred=pac.predict(tfidf_test)
    #score=accuracy_score(y_test,y_pred)
    #print(f'Accuracy: {round(score*100,2)}%')

