# N a m e   :裴鲲鹏
# Student ID:202100172014
# Date&Time :2024/4/13 10:25


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.fft import fft

df = pd.read_csv('..\\EEG_Eye_State_Classification.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Apply FFT to convert time-domain signals to frequency-domain signals
X_fft = np.abs(fft(X))

X_train, X_test, y_train, y_test = train_test_split(X_fft, y, test_size=0.2, random_state=0)

class1 = GaussianNB()
class1.fit(X_train, y_train)

y_pred = class1.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('naive_bayes :', accuracy_score(y_test, y_pred))

X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(X_fft, y, test_size=0.2, random_state=0)

standardScaler = StandardScaler()
X_knn_train = standardScaler.fit_transform(X_knn_train)
X_knn_test = standardScaler.fit_transform(X_knn_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_knn_train, y_knn_train)

y_knn_pred = knn.predict(X_knn_test)
cm1 = confusion_matrix(y_knn_test, y_knn_pred)
print(cm1)
print('neighbors :', accuracy_score(y_knn_test, y_knn_pred))
