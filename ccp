import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Datasets/Churn_Modelling.csv')
dataset.head()
dataset.info()
dataset.describe()
dataset= dataset.drop(columns = ['RowNumber','CustomerId','Surname'])
dataset.info()
dataset['Gender'].unique()
dataset= pd.get_dummies(data=dataset,drop_first=True)
dataset
dataset.Exited.plot.hist()
(dataset.Exited==1).sum()
dataset_2=dataset.drop(columns='Exited')
dataset_2.corrwith(dataset['Exited']).plot.bar(figsize=(16,9), title='Correlated with Exited Column', rot = 45,grid = True)
corr=dataset.corr()
plt.figure(figsize=(16,9))
sns.heatmap(corr,annot=True)
X= dataset.drop(columns='Exited')
y= dataset['Exited']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_test.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
X_train
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred= clf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
results=pd.DataFrame([['Logistic regression',acc,f1,prec,rec]],columns=['Model','Accuracy','F1','Precision','Recall'])
results
print(confusion_matrix(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
y_pred= clf.predict(X_test)
acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
RF_results=pd.DataFrame([['Random Forest Classifier',acc,f1,prec,rec]],columns=['Model','Accuracy','F1','Precision','Recall'])
results.append(RF_results,ignore_index=True)
print(confusion_matrix(y_test,y_pred))
dataset.head()
single_obs=[[647,40,3,85000.45,2,0,0,92012.45,0,1,1]]
clf.predict(scaler.fit_transform(single_obs))
