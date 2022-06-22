#Switching CPU operation instructions to AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Adding progression logging
import logging
logging.getLogger().setLevel(logging.INFO)
#Standard imports ^

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("Machine Learning Datasets/KNN_Project_Data.csv")
print(df.head())

myscaler = StandardScaler()
myscaler.fit(X=df.drop("TARGET CLASS", axis=1))
X = myscaler.transform(X=df.drop("TARGET CLASS", axis=1))

tdf = pd.DataFrame(X, columns=df.columns[:-1])
print(tdf.head())

y = df["TARGET CLASS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)

"""
First training of KNN algorithm
"""
"""
myKNN = KNeighborsClassifier(n_neighbors=1)
myKNN.fit(X_train, y_train)

y_predict = myKNN.predict(X_test)
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))
"""
"""
Retraining KNN algorithm
"""
myKNN = KNeighborsClassifier(n_neighbors=31)
myKNN.fit(X_train, y_train)
y_predict = myKNN.predict(X_test)

print('WITH K=31')
print('')
print(confusion_matrix(y_test,y_predict))
print('')
print(classification_report(y_test,y_predict))

err_rates = []
for idx in range(1,40):
  knn = KNeighborsClassifier(n_neighbors = idx)
  knn.fit(X_train, y_train)
  pred_idx = knn.predict(X_test)
  err_rates.append(np.mean(y_test != pred_idx))

plt.style.use('ggplot')
plt.subplots(figsize = (10,6))
plt.plot(range(1,40), err_rates, linestyle='dashed', color='blue', marker='o', markerfacecolor='red')
plt.xlabel('K-value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs K-value')
plt.show()