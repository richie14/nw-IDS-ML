import pandas as pd
import numpy as np
from sklearn import metrics 

from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('Jun12output1.csv')

#df.head()
y = df.Tag
X = df.drop(['SrNo','Tag'],axis = 1)


import csv
import sys

filename = sys.argv[1]

with open(filename, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
        
new_input = pd.read_csv(filename)


from sklearn.model_selection import train_test_split

print("\nTrain-Test split:")
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.01,random_state=4)
print(X_train.shape, X_test.shape, y_train.shape,  y_test.shape)


from sklearn.metrics import r2_score

from sklearn.metrics import roc_auc_score

knn =  KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train.values, y_train)
print("\nK-Nearest Neighbour:")
print(100 *knn.score(X_test, y_test))
print("K-Nearest Neighbour Prediction:")
print(knn.predict((np.array(new_input))))

knn_pred=knn.predict(X_test) 


from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:")
print(confusion_matrix(y_test,knn_pred))

print("Classification report:")
print(classification_report(y_test,knn_pred))

tp, fn, fp, tn = confusion_matrix(y_test,knn_pred,labels=[1,0]).reshape(-1)
print('\nStatisAnalysis')
print('True Positive Rate :', round(tp*100/(tp+fn+fp+tn),2))
print('False Negative Rate :', round(fn*100/(tp+fn+fp+tn),2))
print('False Positive Rate :', round(fp*100/(tp+fn+fp+tn),2))
print('True Negative Rate :', round(tn*100/(tp+fn+fp+tn),2))

print('\nAccuracy :', round((tp+tn)/(tp+fp+fn+fp)))
