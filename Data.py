import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("fake_or_real_news.csv")

y = df.label 

df = df.drop("Unnamed: 0", axis=1)      

X_train, X_test, Y_train, Y_test = train_test_split(df,y,test_size = 0.2, train_size = 0.8)
X_train, X_cv, Y_train, Y_cv = train_test_split(X_train,Y_train,test_size = 0.2, train_size = 0.8)

