# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 19:14:55 2020

@author: Shankar J
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

(x_train, y_train), (x_test, y_test) =  tf.keras.datasets.mnist.load_data()

#normalize the input data
x_train = x_train/255
x_test = x_test/255

#Create the model
(T, D) = x_train[0].shape #Shape of input data
M = 10 #Number of hidden units in RNN
K = np.unique(y_train).shape[0] #output units

i = tf.keras.Input(shape=(T,D))
x = tf.keras.layers.LSTM(M)(i)
x = tf.keras.layers.Dense(K, activation='softmax')(x)

model = tf.keras.models.Model(i, x)

print(model.summary())

#Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'],
              optimizer=tf.keras.optimizers.Adam(0.01))

#fit the model
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

#predict test set
y_pred = model.predict(x_test).argmax(axis=1)

#print the classification report
print(metrics.classification_report(y_test, y_pred))

#Now try the global max pooling to see if it improves anything
i = tf.keras.Input(shape=(T,D))
x = tf.keras.layers.LSTM(M, return_sequences=True)(i)
x = tf.keras.layers.GlobalMaxPool1D()(x)
x = tf.keras.layers.Dense(K, activation='softmax')(x)

model = tf.keras.models.Model(i, x)

print(model.summary())

#Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'],
              optimizer=tf.keras.optimizers.Adam())

#fit the model
model_history = model.fit(x_train, y_train, epochs=40, 
                          validation_data=(x_test, y_test))

#plot the data
plt.plot(model_history.history['loss'], label='train_loss')
plt.plot(model_history.history['val_loss'], label='val_loss')
plt.title("Training loss curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#predict test set
y_pred = model.predict(x_test).argmax(axis=1)

#print the classification report
print(metrics.classification_report(y_test, y_pred))