import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

df = pd.read_csv('Jun12output1.csv')

new_input = pd.read_csv('test.csv')

#print(df)
print(df.shape)
dataset = df.values
#Splitting dataset

Y=df.Tag

X=df.drop(['SrNo','Tag'], axis=1)


min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
#print(X_scale)


X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

model = Sequential([    Dense(32, activation='relu', input_shape=(18,)),    Dense(32, activation='relu'),    Dense(1, activation='sigmoid'),])
model.compile(optimizer='sgd',              loss='binary_crossentropy',              metrics=['accuracy'])
hist = model.fit(X_train, Y_train,batch_size=32, epochs=10,validation_data=(X_val, Y_val))

print(100* model.evaluate(X_test, Y_test)[1])


print(model.predict((np.array(new_input))))
