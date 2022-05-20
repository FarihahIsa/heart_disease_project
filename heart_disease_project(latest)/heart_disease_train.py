#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:10:58 2022

@author: farihahisa
"""

# libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import rainbow

# scaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split

# model building

from sklearn.neighbors import KNeighborsClassifier

import pickle
#%% Path


SS_SCALER_PATH = os.path.join(os.getcwd(),  'ss_scaler.pkl')
KNN_CLASS_PATH = os.path.join(os.getcwd(), 'knn_classifier.pkl')
#%% EDA

# Step 1) Loading of data

data = pd.read_csv('heart.csv')

# Step 2) Inspection of data

data.head()

data.info() #---> shows data doesnt have any missing/null values

data.describe().T

# data visualisation

# seaborn heatmap - check correlation between various features
plt.figure(figsize=(20,12))
sns.set_context('notebook', font_scale = 1.3)
sns.heatmap(data.corr(),annot=True, linewidths=2)
plt.tight_layout()

#histogram plot
data.hist()
plt.show() #--> shows each feature and label is distributed along different ranges

# bar plot for target class
sns.countplot( x = 'output', data = data) #--> show that classes are almost balanced

# Step 3) Data Cleaning --> no need to clean data

# Step 4) Data Interpretation/Feature selection

# Step 5) Data pre-processing
# with categorical variables, need to break each categorical column into dummy columns 
# with 1 and 0.

data = pd.get_dummies(data, columns = ['sex','cp','fbs','restecg','exng','slp','caa','thall'])
ss = StandardScaler()
columns_to_scale = ['age','trtbps','chol','thalachh','oldpeak']
data[columns_to_scale] = ss.fit_transform (data[columns_to_scale])

# save standardscaler
pickle.dump(ss, open('ss_scaler.pkl','wb'))

# train-test-split
X = data.drop(['output'], axis=1)
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=(0))

#%% ML
# K Neighbors Classifier 
knn_scores = []
for k in range(1,21):     #- 1-20 neighbors used
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train,y_train)
    knn_scores.append(knn_classifier.score(X_test,y_test))

# plot line graph for knn
plt.plot([k for k in range(1,21)], knn_scores, color = 'red') 
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i,knn_scores[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('KNeighbors Classifier scores for different K values')
#---> show from the line graph, the maximum scores 87% achieved when n_neighbor=8

# save the trained model as a pickle string
pickle.dump(knn_classifier, open('knn_classifier.pkl','wb'))







