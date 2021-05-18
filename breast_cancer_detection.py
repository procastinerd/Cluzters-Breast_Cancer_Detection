# -*- coding: utf-8 -*-
"""Breast_Cancer_Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dF9ihIcWSONh733ANn7qO7sTSyEd8Out

Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Importing the dataset"""

dataset = pd.read_csv('/content/data.csv')
X = dataset.iloc[:, 1:31].values
y = dataset.iloc[:, 31].values

dataset.head()

print(X)

print(y)

dataset.drop('id', axis = 1,inplace=True)

dataset.head()

print("Breast Cancer data set dimensions : {}".format(dataset.shape))

"""Encoding dependent variable (Benign:0, Malign:1)"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

dataset.isnull().sum()
dataset.isna().sum()

dataset.describe()

dataset.info()

"""Data Visualisation"""

import seaborn as sns

sns.countplot(dataset['diagnosis'])
plt.title('Count of cancer type')
plt.xlabel('Cancer lethality')
plt.ylabel('Count')
plt.show()

sns.relplot(x="radius_mean",y="texture_mean",hue="diagnosis",data=dataset);

dataset.groupby('diagnosis').size()
dataset.groupby('diagnosis').hist(figsize=(15, 15))

dataset["radius_mean"].plot(kind = 'hist',bins = 200,figsize = (4,4))
plt.title("radius_mean")
plt.xlabel("radius_mean")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize = (20, 12))

corr = dataset.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(corr, mask = mask, linewidths = 1, annot = True, fmt = ".2f")
plt.show()

plt.figure(figsize = (20, 15))
plotnumber = 1

for column in dataset:
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(dataset[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()

"""Removing highly correlated features"""

corr_matrix = dataset.corr().abs() 

mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
tri_dataset = corr_matrix.mask(mask)

to_drop = [x for x in tri_dataset.columns if any(tri_dataset[x] > 0.92)]

dataset = dataset.drop(to_drop, axis = 1)

print(f"The reduced dataframe has {dataset.shape[1]} columns.")

dataset.head()

"""Outlier resolving"""

def mod_outlier(dataset):
        dataset1 = dataset.copy()
        dataset = dataset._get_numeric_data()

        q1 = dataset.quantile(0.25)
        q3 = dataset.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 -(1.5 * iqr) 
        upper_bound = q3 +(1.5 * iqr)

        for col in dataset.columns:
            for i in range(0,len(dataset[col])):
                if dataset[col][i] < lower_bound[col]:            
                    dataset[col][i] = lower_bound[col]

                if dataset[col][i] > upper_bound[col]:            
                    dataset[col][i] = upper_bound[col]    

        for col in dataset.columns:
            dataset1[col] = dataset[col]

        return(dataset1)

dataset = mod_outlier(dataset)

dataset.head()

"""Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

"""Feature Scaling"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression

LRclassifier = LogisticRegression()
LRclassifier.fit(X_train, y_train)

y_pred = LRclassifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score 
print(accuracy_score(y_train, LRclassifier.predict(X_train)))

LRclassifier_acc = accuracy_score(y_test, LRclassifier.predict(X_test))
print(LRclassifier_acc)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

"""KNN"""

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(accuracy_score(y_train, knn.predict(X_train)))
knn_acc = accuracy_score(y_test, knn.predict(X_test))
print(knn_acc)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

"""Support Vector Classifier"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC()
parameters = {
    'gamma' : [0.0001, 0.001, 0.01, 0.1],
    'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_search = GridSearchCV(svc, parameters)
grid_search.fit(X_train, y_train)

grid_search.best_params_
grid_search.best_score_

svc = SVC(C = 10, gamma = 0.01)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score 
print(accuracy_score(y_train, svc.predict(X_train)))

svc_acc = accuracy_score(y_test, svc.predict(X_test))
print(svc_acc)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

"""Decision Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

parameters = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(2, 32, 1),
    'min_samples_leaf' : range(1, 10, 1),
    'min_samples_split' : range(2, 10, 1),
    'splitter' : ['best', 'random']
}

grid_search_dt = GridSearchCV(dtc, parameters, cv = 5, n_jobs = -1, verbose = 1)
grid_search_dt.fit(X_train, y_train)

grid_search_dt.best_params_

grid_search_dt.best_score_

dtc = DecisionTreeClassifier(criterion = 'entropy', max_depth = 28, min_samples_leaf = 1,
                             min_samples_split = 8, splitter = 'random')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print(accuracy_score(y_train, dtc.predict(X_train)))

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))
print(dtc_acc)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

"""
Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 11, max_features = 'auto',
                                  min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)
rand_clf.fit(X_train, y_train)

y_pred = rand_clf.predict(X_test)

print(accuracy_score(y_train, rand_clf.predict(X_train)))

ran_clf_acc = accuracy_score(y_test, y_pred)
print(ran_clf_acc)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

"""Model Score Comparison """

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'SVC', 'Decision Tree Classifier', 'Random Forest Classifier'],
    'Score': [LRclassifier_acc, knn_acc, svc_acc, dtc_acc, ran_clf_acc]
})

models.sort_values(by = 'Score', ascending = False)

import pickle

filename= "model.pkl"

model_pkl= open(filename, 'wb')
data = pickle.dump(svc, model_pkl)

model_pkl.close()