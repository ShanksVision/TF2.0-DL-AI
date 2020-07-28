# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:46:21 2020

@author: Shankar J
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

#import the spam ham data
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

#pre-process the data to get into trainable format
df = df.drop(df.columns[2:], axis=1)
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})

#Train test split
df_train, df_test, y_train, y_test = train_test_split(df['v2'], df['v1'], 
                                                    test_size=0.33)

#Processing text data has the following steps
#Text -> Get tokens -> sequence of integers -> vectors

#text to sequence of integers
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df_train)
sequence_train = tokenizer.texts_to_sequences(df_train)
sequence_test = tokenizer.texts_to_sequences(df_test)

#generate the NxT input matrix for train and test set
V = len(tokenizer.word_index) #number of tokens
X_train = tf.keras.preprocessing.sequence.pad_sequences(sequence_train, 
                                                        padding='pre')
max_length = X_train.shape[1]
X_test = tf.keras.preprocessing.sequence.pad_sequences(sequence_test, maxlen=
                                                       max_length, padding='pre')

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
T = max_length # Length of time series

#Create the model
D = 30 #Embedding layer depth, aka feature vector length

M = 20 #Hidden units

i = tf.keras.Input(shape=(T,))
x = tf.keras.layers.Embedding(V+1, D)(i) #Integers to vectors (last step in pre-processing)
x = tf.keras.layers.LSTM(M, return_sequences=True)(x)
x = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(i, x)

print(model.summary());

#Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model
model_history = model.fit(X_train, y_train, epochs=15, validation_data=
                          (X_test, y_test))

#plot the loss
plt.plot(model_history.history['loss'], label='train_loss')
plt.plot(model_history.history['val_loss'], label='val_loss')
plt.title("Training loss curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#predict the test set
y_pred = model.predict(X_test)

y_pred = y_pred.reshape(len(y_test),)
y_pred = np.round(y_pred)


#Show classfication report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
      
#print some incorrect samples
mis_idx = np.where(y_test != y_pred)[0]
wrong_preds = np.random.choice(mis_idx, size=10)
for i in range(10):
     print(df_test.iloc[wrong_preds[i]])
     print('Original Label : {0}'.format(y_test[wrong_preds[i]]))
     print('Predicted Label : {0}'.format(y_pred[wrong_preds[i]]))
     print('********************')
