## Importing required libraries ##
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set()

## Ignore Warnings ##
import warnings
warnings.filterwarnings("ignore")

## Setting working directory ##
os.chdir('C:/Users/your_file_path')

## Loading data. Example used here is heart-disease data included in repo. ##
heartdata = pd.read_csv('heart.disease.data.clean.csv')

## Correlation Plot ##
plt.figure(figsize=(12,8))
sns.heatmap(heartdata.corr(),cmap='Greens',annot=False)

## Dropping unused columns ##
heartdata.drop(['age', 'trestbps', 'chol', 'cp', 'cigs',
'years', 'fbs', 'famhist', 'thalach', 'thal'],axis=1, inplace=True)

## Replotting correlation plot ##
plt.figure(figsize=(12,8))
sns.heatmap(heartdata.corr(),cmap='Greens',annot=False)

## Categorizing "num" column ##
heartdata["num"] = np.where(heartdata["num"] >0,1,0)

## Defining target and feature variables ##
cols = heartdata.columns
target_col = 'num'
feat_cols = [c for c in cols if c != target_col]
X = heartdata[feat_cols].values
y = heartdata[target_col].values

## Train/Test split ##
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=42)

## Fitting model with 3 neighbors ##
model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
model.fit(X_train, y_train)

## Executing predictions ##
preds = model3.predict(X_test)

## Printing test and predictions ##
print('Actual for test data set')
print(y_test)
print('Prediction for test data set')
print(preds)

## Printing the differences ##
differs = y_test - preds3
print('Differences between the two sets')
print(differs)

## Printing the accuracy score ##
print(accuracy_score(y_test,preds))

