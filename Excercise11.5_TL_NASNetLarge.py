# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:20:38 2020

@author: Shankar J
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met 
import sys, glob
import tensorflow.keras.preprocessing.image as imageprocess 

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
        for i, j in enumerate(wrong_preds):
            if(type(x) is np.ndarray):
                img = tf.keras.preprocessing.image.array_to_img(x[j])
            else:
                img = tf.keras.preprocessing.image.load_img(x[j])
            ax.append(fig.add_subplot(rows, columns, i+1))
            ax[-1].set_title("True label: %s, Predicted: %s" % (labels[y[j]], 
                                                               labels[yhat[j]]))
            plt.imshow(img)
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
        
                               
train_path = '../Data/x1288/train_3channel'
test_path = '../Data/x1288/test_3channel'

#acceptable size for the pretrained-model
#VGG16 needs something between 200-224 pixels and 3 channels
IMAGE_SIZE = (331, 331, 3)          

#Get train, test file path lists and class counts
train_imgs = glob.glob(train_path + '/*/*.bmp')
test_imgs = glob.glob(test_path + '/*/*.bmp')
class_path = glob.glob(train_path + '/*')

#Create the model
nasnet_body = tf.keras.applications.NASNetLarge(include_top=False, input_shape=
                                              IMAGE_SIZE, pooling='avg')

#output shape of 1 input image(1, 224, 224, 3) thru the vgg_body  
features = nasnet_body.predict(np.random.random([1] + list(IMAGE_SIZE)))
D = features.shape[1]

i = tf.keras.layers.Input(shape=(D,))
# x = tf.keras.layers.Dense(512, activation='relu')(i)
# x = tf.keras.layers.Dropout(0.3)(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(3, activation='softmax')(i)

model = tf.keras.models.Model(i, x)

model.summary()

#Create a image data generator to stream images in to network
gen = imageprocess.ImageDataGenerator(preprocessing_function=tf.keras.applications.
                                      nasnet.preprocess_input)

#Create a train and test instance for image data generator
batch_size = 128
train_gen = gen.flow_from_directory(train_path, target_size=IMAGE_SIZE[0:2],
                                    color_mode='rgb', class_mode='categorical',
                                    batch_size=batch_size)
test_gen = gen.flow_from_directory(test_path, target_size=IMAGE_SIZE[0:2],
                                    color_mode='rgb', class_mode='categorical',
                                    batch_size=batch_size)


#Create numpy arrays associated with train and test images and figure out their shapes
train_img_count = len(train_imgs)
test_img_count = len(test_imgs)
class_count = len(class_path)

X_train = np.zeros((train_img_count, D))
y_train = np.zeros((train_img_count, class_count))
X_test = np.zeros((test_img_count, D))
y_test = np.zeros((test_img_count, class_count))
X_test_gen_data = np.zeros((test_img_count, IMAGE_SIZE[0], IMAGE_SIZE[1],
                            IMAGE_SIZE[2]))

#After creating numpy arrays, populate them with predictions
#These predictions will serve as input to our network
i = 0
for X, y in train_gen:
    #get batch size as last batch might not be true batch size
    current_batch_size = len(y)    
    
    #get flattened features
    features = nasnet_body.predict(X)      
    
    #fill the numpy arrays
    X_train[i:i+current_batch_size] = features
    y_train[i:i+current_batch_size] = y
    
    #increment iterator var
    i += current_batch_size
    print('Current batch end : {0}'.format(i))
    
    #print out batch counts just to make sure it works
    if i >= train_img_count:
        print('exiting loop, reached end')
        break

#Repeat same for test set
i = 0
for X, y in test_gen:
    #get batch size as last batch might not be true batch size
    current_batch_size = len(y)
    X_test_gen_data[i:i+current_batch_size] = X
    
    #get flattened features
    features = nasnet_body.predict(X)       
    
    #fill the numpy arrays
    X_test[i:i+current_batch_size] = features
    y_test[i:i+current_batch_size] = y
    
    #increment iterator var
    i += current_batch_size
    print('Current batch end : {0}'.format(i))
    
    #print out batch counts just to make sure it works
    if i >= test_img_count:
        print('exiting loop, reached end')
        break
    
#Check feature vector values to see if they need normalization
print(X_train.max(), X_train.min())
    
#Scale the feature tensor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])

#fit the model for the unnormalized feature vectors
model_history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, 
                                                                        y_test))
#predict the output and get the confusion matrix and classification report
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)
y_pred_train = model.predict(X_train)
y_pred_train = np.round(y_pred_train)


#fit the model for the unnormalized feature vectors
# model_history = model.fit(X_train_norm, y_train, epochs=15, validation_data=(X_test_norm, 
#                                                                         y_test))
# #predict the output and get the confusion matrix and classification report
# y_pred = model.predict(X_test_norm)
# y_pred = np.round(y_pred)
# y_pred_train = model.predict(X_train_norm)
# y_pred_train = np.round(y_pred_train)

#plot loss curves and missed predictions
class_labels = ['damage', 'tape', 'wrinkle']
plot_loss_wrong_preds(model_history, "Loss Curves - NasnetLarge_FeatureTensor_unnormalized",
                      X_test_gen_data, y_test, y_pred, class_labels)

#print metrics
#print_model_metrics(y_test, y_pred, class_labels, 'Test Metrics')

#save metrics
file_stream = open('results/Ex11.5_NASNetLarge_OpBat_15ep.txt', 'w')
print("Xception Model Summary\n----------\n", file=file_stream)  
nasnet_body.summary(print_fn=lambda x: print(x, file=file_stream)) 
print("FC Model Summary\n----------\n", file=file_stream)  
model.summary(print_fn=lambda x: print(x, file=file_stream)) 
print_model_metrics(y_train, y_pred_train, class_labels, 'Train Metrics', file_stream)
print_model_metrics(y_test, y_pred, class_labels, 'Test Metrics', file_stream)
file_stream.close()

