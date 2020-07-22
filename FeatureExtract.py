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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import svm
import seaborn as sb
from sklearn.metrics import confusion_matrix,f1_score


df = pd.read_csv('fake_or_real_news.csv')

df = df.drop(columns = ['Unnamed: 0'],axis=1)

print(df.isnull().sum())

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
#x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size = 0.2)

print("In train = ",x_train.shape[0])
print("In test = ",x_test.shape[0])
#print("In cv = ",x_cv.shape[0])

#In train =  5068
#In test =  1267

df

#Visualising data

#distribution of classes for prediction
def create_distribution(dataFile):
    
    return sb.countplot(x='label', data=dataFile, palette='hls')

asd = pd.DataFrame(df['label'])

create_distribution(df)


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


#Using bag of words

# using LogisticRegression
logReg_pipeline_cv = Pipeline([
    ('LogRCV', cvec),
    ('LogR_model', LogisticRegression())
])

logReg_pipeline_cv.fit(x_train['processed'], x_train['label'])
predictions_logReg = logReg_pipeline_cv.predict(x_test['processed'])
logReg_cv = np.mean(predictions_logReg == x_test['label'])
score_log = f1_score(y_test,predictions_logReg, average = 'binary',pos_label='REAL')

# using SVM
svm_pipeline_cv = Pipeline([
    ('svmCV', cvec),
    ('svm_model', svm.LinearSVC())
])

svm_pipeline_cv.fit(x_train['processed'], x_train['label'])
predictions_svm = svm_pipeline_cv.predict(x_test['processed'])
svm_cv = np.mean(predictions_svm == x_test['label'])
score_svm = f1_score(y_test,predictions_svm, average = 'binary',pos_label='REAL')

confusion_matrix(y_test,predictions_logReg,labels=['FAKE','REAL'])

#array([[591,  51],
#       [ 66, 559]])

confusion_matrix(y_test,predictions_svm,labels=['FAKE','REAL'])

#array([[565,  77],
#      [ 80, 545]])

print(f'Logisitc Regression Accuracy : {logReg_cv}')
print(f'SVM Accuracy :                 {svm_cv}')
print(f'Logisitc Regression f1 score : {score_log}')
print(f'SVM f1 score :                 {score_svm}')

#Logisitc Regression Accuracy : 0.9131807419100236
#SVM Accuracy :                 0.8839779005524862
#Logisitc Regression f1 score : 0.9164133738601823
#SVM f1 score :                 0.8887206661619985

#Using tfidf vectors

# using LogisticRegression
logReg_pipeline_cv = Pipeline([
    ('LogRCV', tfidf_vectorizer),
    ('LogR_model', LogisticRegression())
])

logReg_pipeline_cv.fit(x_train['processed'], x_train['label'])
predictions_logReg_ = logReg_pipeline_cv.predict(x_test['processed'])
logReg_ngram = np.mean(predictions_logReg_ == x_test['label'])
score_log_ = f1_score(y_test,predictions_logReg_, average = 'binary',pos_label='REAL')
                                                   
# using SVM
svm_pipeline_cv = Pipeline([
    ('svmCV', tfidf_vectorizer),
    ('svm_model', svm.LinearSVC())
])

svm_pipeline_cv.fit(x_train['processed'], x_train['label'])
predictions_svm_ = svm_pipeline_cv.predict(x_test['processed'])
svm_ngram = np.mean(predictions_svm_ == x_test['label'])
score_svm_ = f1_score(y_test,predictions_svm_, average = 'binary',pos_label='REAL')
                      
confusion_matrix(y_test,predictions_logReg_,labels=['FAKE','REAL'])

#array([[594,  48],
#      [ 59, 566]])

confusion_matrix(y_test,predictions_svm_,labels=['FAKE','REAL'])

#array([[604,  38],
#       [ 41, 584]])


print(f'Logistic Regression Accuracy : {logReg_ngram}')
print(f'SVM Accuracy :                 {svm_ngram}')
print(f'Logistic Regression f1 score : {score_log_}')
print(f'SVM f1 score :                 {score_svm_}')

#Logisitc Regression Accuracy : 0.9013417521704814
#SVM Accuracy :                 0.9218626677190213
#Logisitc Regression f1 score : 0.904507257448434
#SVM f1 score :                 0.9248291571753986


