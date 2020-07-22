# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 07:00:28 2020

@author: Shankar J

CIFAR 10 with multi GPU and optimizer change
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import math

#Download the cifar10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

#Pre-process the data set
X_train = X_train/255
X_test = X_test/255

y_train = y_train.flatten()
y_test = y_test.flatten()

label_dict = {0: "airplane",
              1: "automobile",
              2: "bird",
              3: "cat",
              4: "deer",
              5: "dog",
              6: "frog",
              7: "horse",
              8: "ship",
              9: "truck"}

#Create multi gpu training strategy
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

batch_size_per_gpu = 64
global_batch_size = batch_size_per_gpu * strategy.num_replicas_in_sync

#Configure callback for early stopping and pick the best model
early_stopper = tf.keras.callbacks.EarlyStopping('val_accuracy', min_delta=0.01,
                                                 patience=22, verbose=1,
                                                 baseline=0.5, mode='max')

checkpoint = tf.keras.callbacks.ModelCheckpoint('models/Excercise6_bestmodel.h5', 
                                                monitor='val_accuracy', verbose=1, 
                                                save_best_only=True)

#Configure data augmenation to improve generalization
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=25,
                                                                  zoom_range=0.4,
                                                                  horizontal_flip=True                                                                  
                                                                  )
augmented_data = image_generator.flow(X_train, y_train, 
                                      batch_size=global_batch_size)

with strategy.scope():
    
    #Create the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(32,32,3)))
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="elu"))
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="elu"))
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="elu"))
    model.add(tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="elu"))
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="elu"))
    model.add(tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="elu"))
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="elu"))
    model.add(tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="elu"))
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation="elu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1024, activation="elu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation="softmax")) 
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

#Fit the model
before = time.perf_counter()
network_history = model.fit(augmented_data, epochs=100, 
                            steps_per_epoch=math.ceil(len(X_train)/global_batch_size), 
                            validation_data=(X_test, y_test), 
                            callbacks=[early_stopper, checkpoint])
after = time.perf_counter()
print("--------Time taken to train model-----")
print(str(after-before) + " seconds")

#Evaluate test set
print("--------Test set Loss and Accuracy----")
print(model.evaluate(X_test, y_test))

#Predict test set 
y_pred = model.predict(X_test)

#plot the loss 
plt.plot(network_history.history['loss'], label='train_loss')
plt.plot(network_history.history['val_loss'], label='val_loss')
plt.title("Training loss curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

y_pred =  y_pred.argmax(axis=1)

from sklearn import metrics
print("--------Confusion Matrix------")
print(metrics.confusion_matrix(y_test, y_pred))

#print the classification report
print("--------Classification report------")
print(metrics.classification_report(y_test, y_pred, 
                                    target_names=list(label_dict.values())))

#print model summary
print("----Model Summary----")
print(model.summary())

#Show some misclassified samples
mis_idx = np.where(y_test != y_pred)[0]
wrong_preds = np.random.choice(mis_idx, size=16)
w=32
h=32
ax = []
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 4
for i, j in enumerate(wrong_preds):
    img = X_test[j]
    ax.append(fig.add_subplot(rows, columns, i+1))
    ax[-1].set_title("True label: %s, Predicted: %s" % (label_dict[y_test[j]], 
                                                       label_dict[y_pred[j]]))
    plt.imshow(img)
plt.tight_layout(True)
plt.show()


          

