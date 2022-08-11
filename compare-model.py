import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


from sklearn.datasets import make_classification
from sklearn import preprocessing

from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Jun12output1.csv')

new_input = pd.read_csv('test.csv')

#df.head()
y = df.Tag
X = df.drop(['SrNo','Tag'],axis = 1)


#divide database into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.01,random_state=4)
print(X_train.shape, X_test.shape, y_train.shape,  y_test.shape)


rf = RandomForestRegressor(n_estimators = 100)
rf.fit(X_train.values, y_train)
print("\nRandomForest Regression:")
print(100 * rf.score(X_test, y_test))
print("RandomForest Regression Prediction:")
print(rf.predict((np.array(new_input))))

rf_pred=rf.predict(X_test) 


tp, fn, fp, tn = confusion_matrix(y_test,rf_pred,labels=[1,0]).reshape(-1)
print('\nStatisAnalysis')
print('True Positive Rate :', round(tp*100/(tp+fn+fp+tn),2))
print('False Negative Rate :', round(fn*100/(tp+fn+fp+tn),2))
print('False Positive Rate :', round(fp*100/(tp+fn+fp+tn),2))
print('True Negative Rate :', round(tn*100/(tp+fn+fp+tn),2))
print('\nAccuracy :', round((tp+tn)/(tp+fp+fn+fp)))

print("Classification report:")
print(classification_report(y_test,rf_pred))


decisionTree = tree.DecisionTreeClassifier(max_depth=5)
decisionTree.fit(X_train, y_train)
y_pred = decisionTree.predict(X_test)
accuracy4 = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage4 = 100 * accuracy4
print("\nDecision tree:")
print(accuracy_percentage4)
print(100 *decisionTree.score(X_test, y_test))
print("Decision Tree Prediction:")
print(decisionTree.predict((np.array(new_input))))

dt_pred=decisionTree.predict(X_test) 

tp, fn, fp, tn = confusion_matrix(y_test,dt_pred,labels=[1,0]).reshape(-1)
print('\nStatisAnalysis')
print('True Positive Rate :', round(tp*100/(tp+fn+fp+tn),2))
print('False Negative Rate :', round(fn*100/(tp+fn+fp+tn),2))
print('False Positive Rate :', round(fp*100/(tp+fn+fp+tn),2))
print('True Negative Rate :', round(tn*100/(tp+fn+fp+tn),2))
print('\nAccuracy :', round((tp+tn)/(tp+fp+fn+fp)))

print("Confusion Matrix:")
print(confusion_matrix(y_test,dt_pred))

print("Classification report:")
print(classification_report(y_test,dt_pred))


knn =  KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train.values, y_train)
print("\nK-Nearest Neighbour:")
print(100 *knn.score(X_test, y_test))
print("K-Nearest Neighbour Prediction:")
print(knn.predict((np.array(new_input))))

knn_pred=knn.predict(X_test) 
tp, fn, fp, tn = confusion_matrix(y_test,knn_pred,labels=[1,0]).reshape(-1)
print('\nStatisAnalysis')
print('True Positive Rate :', round(tp*100/(tp+fn+fp+tn),2))
print('False Negative Rate :', round(fn*100/(tp+fn+fp+tn),2))
print('False Positive Rate :', round(fp*100/(tp+fn+fp+tn),2))
print('True Negative Rate :', round(tn*100/(tp+fn+fp+tn),2))
print('\nAccuracy :', round((tp+tn)/(tp+fp+fn+fp)))


print("Confusion Matrix:")
print(confusion_matrix(y_test,knn_pred))

print("Classification report:")
print(classification_report(y_test,knn_pred))


nb = GaussianNB()
nb.fit(X_train, y_train)
print("\nNaive Baise:")
print(100 *nb.score(X_test, y_test))
# get prediction for new input
print("Naive Baise Prediction:")
print(nb.predict((np.array(new_input))))

nb_pred=nb.predict(X_test) 
tp, fn, fp, tn = confusion_matrix(y_test,nb_pred,labels=[1,0]).reshape(-1)
print('\nStatisAnalysis')
print('True Positive Rate :', round(tp*100/(tp+fn+fp+tn),2))
print('False Negative Rate :', round(fn*100/(tp+fn+fp+tn),2))
print('False Positive Rate :', round(fp*100/(tp+fn+fp+tn),2))
print('True Negative Rate :', round(tn*100/(tp+fn+fp+tn),2))
print('\nAccuracy :', round((tp+tn)/(tp+fp+fn+fp)))


print("Confusion Matrix:")
print(confusion_matrix(y_test,nb_pred))

print("Classification report:")
print(classification_report(y_test,nb_pred))


"""
#num_of_classes = len(df.Tag.unique())
#Xgb = XGBClassifier(booster='gbtree', objective='multi:softprob', use_label_encoder=False, random_state=42, eval_metric="auc", num_class=num_of_classes)
#Xgb.fit(X_train.values,y_train)
#val = Xgb.predict(X_test)
#lb = preprocessing.LabelBinarizer()
#lb.fit(y_test)
#y_test_lb = lb.transform(y_test)
#val_lb = lb.transform(val)

#print("MultiClass Classifier Prediction:")
#print(Xgb.predict((np.array(new_input))))

#print("\nMultiClass Classifier:")
#print(roc_auc_score(y_test_lb, val_lb, average='macro'))

#preprocess data
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)

loR = linear_model.LogisticRegression(max_iter=10000000000)
loR.fit(X_train, y_train)
#print("\nLogistic Regression:")
#print(100 * loR.score(X_test, y_test))
#print("Logistic Regression Prediction:")
#print(loR.predict((np.array(new_input))))

liR = LinearRegression()
liR.fit(X_train, y_train)
print("\nLinear Regression:")
print(100 * liR.score(X_test, y_test))
print("Linear Regression Prediction:")
print(liR.predict((np.array(new_input))))


print("\nGradient Boosting:")
gb=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1,max_features=12, random_state=0).fit(X_train, y_train)
print(100 *gb.score(X_test, y_test))
print("Gradient Boosting Prediction:")
print(gb.predict((np.array(new_input))))


sv =  SVC(probability=True)
sv.fit(X_train.values, y_train)
print("\nSVM:")
print(100 *sv.score(X_test, y_test))
print("SVM Prediction:")
print(sv.predict((np.array(new_input))))


nb = GaussianNB()
nb.fit(X_train, y_train)
print("\nNaive Baise:")
print(100 *nb.score(X_test, y_test))
# get prediction for new input
print("Naive Baise Prediction:")
print(nb.predict((np.array(new_input))))
"""

