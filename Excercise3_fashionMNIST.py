# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:53:19 2020

@author: Shankar J
"""
import tensorflow as tf
import numpy as np
import sklearn.metrics as met

#Check if training will use GPU - In python shell
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#Load the data
mnist_fashion = tf.keras.datasets.fashion_mnist

#split the data
(X_train, y_train), (X_test, y_test) = mnist_fashion.load_data()

#normalize the data
X_train = X_train/255 #8 bit data
X_test = X_test/255

#Create a dictionary for mapping class numbers to string labels
label_dict = {
 0: 'Top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Boot',
}

#Map target class numeric values to one hot encoded arrays
y_train = tf.keras.utils.to_categorical(y_train, dtype='uint8')
y_test = tf.keras.utils.to_categorical(y_test, dtype='uint8')

#Change the input shape to match with tensorflow required format 3 dim tensor
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

#Create a model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(32, (3,3), (1,1), 'same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (3,3), (1,1), 'same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(4,4), padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

#Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#fit the model
nn = model.fit(x=X_train, y=y_train, batch_size=1000, 
               epochs=20, validation_data=(X_test, y_test))

#plot the losses
import matplotlib.pyplot as plt
plt.plot(nn.history['loss'], label='loss')
plt.plot(nn.history['val_loss'], label='val_loss')
plt.legend()          

#evaluate the model
print(model.evaluate(X_test, y_test))

#predict the test set values
y_pred = model.predict(X_test)

#Getback to original labels format from one hot encoded array
y_test_class = y_test.argmax(axis=1)
y_pred_class = y_pred.argmax(axis=1)

#print the confusion matrix
cm = met.confusion_matrix(y_test_class, y_pred_class)
print("--------Confusion Matrix------\n")
print(cm)

#plot the corellation map
# import seaborn as sns
# sns.heatmap(cm)

#print the classification report
print("--------Classification report------\n")
cr = met.classification_report(y_test_class, y_pred_class, target_names=list(label_dict.values()))
print(cr)

#print model summary
print(model.summary())

#show some misclassified samples
# mis_idx = np.where(y_test != y_pred)[0]
# plt.imshow(X_test[mis_idx[0]], cmap='gray')
# plt.title("Y_test %s Y_Pred %s" % (y_test[mis_idx[0]], y_pred[mis_idx[0]]))
# plt.imshow(X_test[mis_idx[-1]], cmap='gray')
# plt.title("Y_test %s Y_Pred %s" % (y_test[mis_idx[-1]], y_pred[mis_idx[-1]]))
# index_from_middle = int(np.round(len(mis_idx)/2))
# plt.imshow(X_test[index_from_middle], cmap='gray')
# plt.title("Y_test %s Y_Pred %s" % (y_test[mis_idx[index_from_middle]], 
#                                     y_pred[mis_idx[index_from_middle]]))

