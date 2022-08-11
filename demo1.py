
#Importing libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
    
#Loading dataset

data = pd.read_csv('Iris.csv')

print('Actual dataset size :', data.shape)
#print(data.head())

#Splitting dataset

y=data.Species
x=data.drop(['Species','Id'], axis=1)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#print(x_train.head())

print('Training dataset size: ', x_train.shape)

#print(x_test.head())

print('Testing dataset size: ', x_test.shape)


# ********** Linear regression **********

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import linear_model

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

# Make predicitions 
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)


print(lr.coef_)


#Evaluation
#print(accuracy_score(y_train, y_lr_train_pred))
#print('Test data :', accuracy_score(y_test, y_lr_test_pred))


# R2 and MSE
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

#print(lr_test_mse)

lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()

lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']






#Data visualization

#import matplotlib.pyplot as plt
#import numpy as np
#plt.figure(figsize=(5,5))
#plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)
#z = np.polyfit(y_train, y_lr_train_pred, 1)
#p = np.poly1d(z)
#plt.plot(y_train,p(y_train),"#F8766D")
#plt.ylabel('Predicted LogS')
#plt.xlabel('Experimental LogS')



# ***** Random Forest*****
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=42)
rf.fit(x_train, y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)


rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random forest',rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']


from sklearn.tree import ExtraTreeRegressor
et = ExtraTreeRegressor(random_state=42)
et.fit(x_train, y_train)


predicted = et.predict([[6.2,2.2,4.5,1.5]])
print('It is type: ', predicted)



# ********** Linear regression **********



# ********** Linear regression **********




# ********** Linear regression **********


# ********** Linear regression **********



# ********** Linear regression **********




# ********** Linear regression **********




# ********** Linear regression **********




# ********** Linear regression **********




# ********** Linear regression **********



print(pd.concat([lr_results, rf_results]))

