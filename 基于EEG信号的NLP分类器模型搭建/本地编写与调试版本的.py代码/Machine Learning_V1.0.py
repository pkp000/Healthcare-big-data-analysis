# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/4/10 21:51

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv('..\\EEG_Eye_State_Classification.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
'''
from sklearn.preprocessing import StandardScaler
standardScaler=StandardScaler()
X_train=standardScaler.fit_transform(X_train)
X_test=standardScaler.fit_transform(X_test)
'''

class1 = GaussianNB()
class1.fit(X_train, y_train)

y_pred = class1.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('naive_bayes :', accuracy_score(y_test, y_pred))


X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(X, y, test_size=0.2, random_state=0)

standardScaler = StandardScaler()
X_knn_train = standardScaler.fit_transform(X_knn_train)
X_knn_test = standardScaler.fit_transform(X_knn_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_knn_train, y_knn_train)

y_knn_pred = knn.predict(X_test)
cm1 = confusion_matrix(y_knn_test, y_knn_pred)
print(cm1)
print('neighbors :', accuracy_score(y_knn_test, y_knn_pred))
