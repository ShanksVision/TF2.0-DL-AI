# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 21:55:09 2020

@author: Shankar J
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mintemp_series = pd.read_csv("daily-min-temperatures.csv")

train_data = np.array(mintemp_series['Temp'][:2000])
test_data = np.array(mintemp_series['Temp'][2000:])

T = 30 #number of days needed to predict the future temp
D = 1 #Only 1 dimension which is temp
X=[]
Y=[]

for t in range(len(train_data)-T):
    x_train = train_data[t:t+T]
    X.append(x_train)
    y_train = train_data[t+T]
    Y.append(y_train)
    
#Format the data to be in NxTxD format

X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y)
N = len(X)

print("X Shape", X.shape, "Y Shape", Y.shape)

#Generate test or validation data set in similar way

X_test=[]
Y_test=[]

for t in range(len(test_data)-T):
    x_test = test_data[t:t+T]
    X_test.append(x_test)
    y_test = test_data[t+T]
    Y_test.append(y_test)
    
#Format the data to be in NxTxD format

X_test = np.array(X_test).reshape(-1, T, 1)
Y_test = np.array(Y_test)

print("Test X Shape", X_test.shape, "Test Y Shape", Y_test.shape)

#build the model
i = tf.keras.Input(shape=(T, D))
x = tf.keras.layers.GRU(30)(i)
x = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(i, x)

print(model.summary())

#compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse')

#fit the model
model_history = model.fit(X, Y, epochs=100, validation_data=(X_test, Y_test))

#plot the loss
plt.plot(model_history.history['loss'], label='train_loss')
plt.plot(model_history.history['val_loss'], label='val_loss')
plt.title("Training loss curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#one step forecast
Y_pred = model.predict(X_test)
print(Y_pred.shape)

Y_pred = Y_pred.flatten() #since it was 2 dim 1620x1 instead of 1620,

plt.plot(Y_test, label='targets')
plt.plot(Y_pred, label='predictions')
plt.legend()
plt.show()

print(tf.keras.metrics.mean_absolute_error(Y_test, Y_pred))

#multi-step forecast
Y_pred_multi = []

# first validation input
last_x = X_test[0] # 1-D array of length T

while len(Y_pred_multi) < len(Y_test):
  p = model.predict(last_x.reshape(1, T, 1))[0,0] # 1x1 array -> scalar
  
  # update the predictions list
  Y_pred_multi.append(p)
  
  # make the new input
  last_x = np.roll(last_x, -1)
  last_x[-1] = p

Y_pred_multi = np.array(Y_pred_multi)
plt.plot(Y_test, label='forecast target')
plt.plot(Y_pred_multi, label='forecast prediction')
plt.legend()