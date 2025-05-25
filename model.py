import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sb
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle
df = pd.read_csv("winequality-white.csv",delimiter=";")
df = df.drop('residual sugar', axis=1)
df['best_quality'] = [1 if x >6 else 0 for x in df['quality']]
X = df.drop(['quality','best_quality'],axis=1)
Y = df['best_quality']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=40)
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model = knn_model.fit(xtrain,ytrain)
pickle.dump(knn_model, open('kwine_model.pkl', 'wb'))