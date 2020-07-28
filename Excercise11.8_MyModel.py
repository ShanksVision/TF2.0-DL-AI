# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 08:36:44 2020

@author: cupertino_user

Implement OdBat classification using a custom model
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, glob
import tensorflow.keras.preprocessing.image as imageprocess 
import sklearn.metrics as met

def plot_loss_wrong_preds(history, title, x=None, y=None, yhat=None, labels=None):
    #Plot the train loss and val loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    #Show some misclassified samples    
    if(labels):
        #Check if y is one hot encoded
        if(len(y.shape) != 1):
            y = y.argmax(axis=1)
            yhat = yhat.argmax(axis=1)  
        #get some wrong predictions
        mis_idx = np.where(y != yhat)[0]
        size = min(len(mis_idx), 25)
        wrong_preds = np.random.choice(mis_idx, size=size) 
        ax = []
        fig=plt.figure(figsize=(18, 18))
        columns = 5
        rows = 5
        color_map = None
        for i, j in enumerate(wrong_preds):
            if(type(x) is np.ndarray):
                img = tf.keras.preprocessing.image.array_to_img(x[j])
                color_map = 'gray' if x.shape[3] == 1 else None
            else:
                img = tf.keras.preprocessing.image.load_img(x[j])
            ax.append(fig.add_subplot(rows, columns, i+1))
            ax[-1].set_title("True label: %s, Predicted: %s" % (labels[y[j]], 
                                                               labels[yhat[j]]))
            plt.imshow(img, cmap=color_map)
        plt.tight_layout(True)
        plt.show()
    
def print_model_metrics(y, yhat, labels, title, stream=sys.stdout, wrong_preds=False):
    
    #Check if y is one hot encoded
    if(len(y.shape) != 1):
        y = y.argmax(axis=1)
        yhat = yhat.argmax(axis=1)
        
    print('\n' + title + '\n------------\n', file=stream)   
    
    print("Classification Metrics\n----------\n", file=stream)  
    print(met.classification_report(y, yhat, target_names=labels,
                                    zero_division=1), file=stream) 
    
    print("Confusion Matrix\n----------\n", file=stream)
    print(met.confusion_matrix(y, yhat), file=stream)    
        
    if(wrong_preds):
        print("Wrong Predictions\n----------\n", file=stream)
        mis_idx = np.where(y != yhat)[0]
        size = min(len(mis_idx), 9)
        wrong_preds = np.random.choice(mis_idx, size=size)
        for i in range(size):
            #print(df_test.iloc[wrong_preds[i]], file=stream)
            print('Original Label : {0}'.format(y[wrong_preds[i]]), 
                  file=stream)
            print('Predicted Label : {0}'.format(yhat[wrong_preds[i]]),
                  file=stream)
            print('********************', file=stream)
               
def preprocess_input(x):
    return x/255            
                   
train_path = '../Data/x1288/train'
test_path = '../Data/x1288/test'

#Preprocessed image size
IMAGE_SIZE = (256, 256, 1)          

#Get train, test file path lists and class counts
train_imgs = glob.glob(train_path + '/*/*.bmp')
test_imgs = glob.glob(test_path + '/*/*.bmp')
class_path = glob.glob(train_path + '/*')

#Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE))
model.add(tf.keras.layers.Conv2D(64, (5,5), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(64, (5,5), padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(256, (3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2, padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(2048, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1024, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(3, activation="softmax"))


#Configure callback for early stopping and pick the best model
early_stopper = tf.keras.callbacks.EarlyStopping('val_accuracy', min_delta=0.01,
                                                 patience=25, verbose=1,
                                                 mode='max', baseline=0.85)

checkpoint = tf.keras.callbacks.ModelCheckpoint('models/Excercise11_bestmodel_relu.h5', 
                                                monitor='val_accuracy', verbose=1, 
                                                save_best_only=True)

#Configure the image generator
batch_size = 64

# train_gen = imageprocess.ImageDataGenerator(preprocessing_function=preprocess_input, 
#                                                   rotation_range=10, zoom_range=0.1) 
train_gen = imageprocess.ImageDataGenerator(preprocessing_function=preprocess_input)                                                   

test_gen = imageprocess.ImageDataGenerator(preprocessing_function=preprocess_input)                                                    

train_flow = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE[0:2],
                                    color_mode='grayscale', class_mode='categorical',
                                    batch_size=batch_size)
test_flow = test_gen.flow_from_directory(test_path, target_size=IMAGE_SIZE[0:2],
                                    color_mode='grayscale', class_mode='categorical',
                                    batch_size=batch_size)

#Create numpy arrays associated with train and test images and figure out their shapes
train_img_count = len(train_imgs)
test_img_count = len(test_imgs)
class_count = len(class_path)

#Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Fit the model
network_history = model.fit(train_flow, epochs=60, 
                            steps_per_epoch=int(np.ceil(train_img_count/batch_size)), 
                            validation_data=test_flow,
                            validation_steps=int(np.ceil(test_img_count/batch_size)),
                            callbacks=[early_stopper, checkpoint])

#predict the output and get the confusion matrix and classification report

X_test_gen_data = np.zeros((test_img_count, IMAGE_SIZE[0], IMAGE_SIZE[1],
                            IMAGE_SIZE[2]))
y_pred = np.zeros((test_img_count, class_count))
y_test = np.zeros((test_img_count, class_count))
y_train = np.zeros((train_img_count, class_count))
y_pred_train = np.zeros((train_img_count, class_count))

#Populate the train and test numpy arrays to calculate metrics
i = 0
for X, y in test_flow:
    batch_samples = len(y)
    X_test_gen_data[i:i+batch_samples] = X
    y_pred[i:i+batch_samples] = model.predict_on_batch(X)
    y_test[i:i+batch_samples] = y
    i += batch_samples
    if(i >= test_img_count):
        break    

i = 0
for X, y in train_flow:
    batch_samples = len(y)    
    y_pred_train[i:i+batch_samples] = model.predict_on_batch(X)
    y_train[i:i+batch_samples] = y
    i += batch_samples 
    if(i >= train_img_count):
        break       

y_pred = np.round(y_pred)
y_pred_train = np.round(y_pred_train)

#plot loss curves and missed predictions
class_labels = ['damage', 'tape', 'wrinkle']
plot_loss_wrong_preds(network_history, "Loss Curves - MySimpleModel",
                      X_test_gen_data, y_test, y_pred, class_labels)

#print metrics
#print_model_metrics(y_test, y_pred, class_labels, 'Test Metrics')

#save metrics
file_stream = open('results/Ex11.8_Mymodel_OdBat.txt', 'w')
print("Model Summary\n----------\n", file=file_stream)  
model.summary(print_fn=lambda x: print(x, file=file_stream)) 
print_model_metrics(y_train, y_pred_train, class_labels, 'Train Metrics', file_stream)
print_model_metrics(y_test, y_pred, class_labels, 'Test Metrics', file_stream)
file_stream.close()





          