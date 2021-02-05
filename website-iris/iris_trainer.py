#import os
#import mpld3
import joblib
#import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# load data
iris = pd.read_csv('data/iris.csv')
X = iris.drop('species', axis=1)
y = iris['species']
y.unique()

# prepare training
encoder = LabelBinarizer()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
scaler = MinMaxScaler()
scaler.fit(X_train)
scaler_X_train = scaler.transform(X_train)
scaler_X_test = scaler.transform(X_test)

# create the models (neural network; one row per layer)
model = Sequential()
model.add(Dense(units=4, activation='relu', input_shape=[4,]))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# train our model
early_stop = EarlyStopping(patience=10)
model.fit(x=scaler_X_train,y=y_train,epochs=300,validation_data=(scaler_X_test,y_test),callbacks=[early_stop])

# reprocess the Data once again
scaler_X = scaler.fit_transform(X)
model2 = Sequential()
model2.add(Dense(units=4, activation='relu', input_shape=[4,]))
model2.add(Dense(units=3, activation='softmax'))
model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model2.fit(x=scaler_X,y=y,epochs=300)

# check results
metrics = pd.DataFrame(model.history.history)
print("reliability : " + str(model.evaluate(scaler_X_test,y_test,verbose=0)))

model2.save("final_iris_model.h5")
joblib.dump(scaler,'iris_scaler.pkl')
