## Performance Evaluations. Data used is bank.csv, located in repo ##

## Importing Required Libraries ##
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

## Importing Data ##
data = pd.read_csv('bank.csv')

## Pre-processing, done individually ##
data["y"] = le.fit_transform(data["y"])
data["job_n"] = le.fit_transform(data["job"])
data["housing"] = le.fit_transform(data["housing"])
data["education_n"] = le.fit_transform(data["education"])
data["marital_n"] = le.fit_transform(data["marital"])
data["default"] = le.fit_transform(data["default"])
data["poutcome"] = le.fit_transform(data["poutcome"])
data["loan"] = le.fit_transform(data["loan"])
data["contact_n"] = le.fit_transform(data["contact"])
data["month_n"] = le.fit_transform(data["month"])

## Deleting unnecesary data ##
del data['job']
del data['marital']
del data['contact']
del data['education']
del data['month']

## Renaming columns ##
data.rename(columns = {'job_n':'job',
'education_n':'education', 'marital_n':'marital', 
'contact_n':'contact', 'month_n':'month'}, inplace = True)

## Model prep of data ##
cols = data.columns
pred_col = 'y'
feat_cols = [c for c in cols if c != pred_col]
x = data[feat_cols].values
y = data[pred_col].values

## Train/Test split ##
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=44)

## Defining and fitting Logistic Regression ##
model = LogisticRegression()
model.fit(x_train, y_train)

## Predictions ##
preds = model.predict(x_test)

## Accuracy Score ##
print(accuracy_score(y_test,preds))

## Confusion Matrix ##
cm = confusion_matrix(y_test, preds)
target_labels = np.unique(y_test)

sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True, cmap="RdPu",
            xticklabels=target_labels, yticklabels=target_labels)

plt.xlabel('predicted label')
plt.ylabel('actual label');

## Classification Report ##
print(classification_report(y_test, preds))

## Plotting of ROC Curve ##
lr_prob = cross_val_predict(model, x_train, y_train, cv=3, method='predict_proba')
lr_score = lr_prob[:, 1]

def ROC_Curve(title, y_train, scores, label=None):
    from sklearn.metrics import roc_curve
    
    

    fpr, tpr, thresholds = roc_curve(y_train, scores)
    print('AUC Score ({}): {:.2f} '.format(title, roc_auc_score(y_train, scores)))
    

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve: {}'.format(title), fontsize=16)
    plt.show()
ROC_Curve('Portuguese Banking Marketing Initiative',y_train,lr_score)









