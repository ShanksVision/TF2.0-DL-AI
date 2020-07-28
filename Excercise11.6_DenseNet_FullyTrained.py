# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 07:46:03 2020

@author: cupertino_user
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
IMAGE_SIZE = (224, 224, 3)          

#Get train, test file path lists and class counts
train_imgs = glob.glob(train_path + '/*/*.bmp')
test_imgs = glob.glob(test_path + '/*/*.bmp')
class_path = glob.glob(train_path + '/*')

#Create the model
dense121_body = tf.keras.applications.DenseNet121(include_top=False, input_shape=
                                              IMAGE_SIZE, pooling='avg', weights=None)

i = dense121_body.output
# x = tf.keras.layers.Dense(512, activation='relu')(i)
# x = tf.keras.layers.Dropout(0.3)(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(3, activation='softmax')(i)

model = tf.keras.models.Model(dense121_body.input, x)

model.summary()

#Create a image data generator to stream images in to network
gen = imageprocess.ImageDataGenerator(preprocessing_function=tf.keras.applications.
                                      densenet.preprocess_input)

#Create a train and test instance for image data generator
batch_size = 64
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

#compile the model
optimizer = tf.keras.optimizers.SGD(0.1, 0.9, nesterov=True)

def custom_lr_adjuster(epoch):    
    total_epochs = 150  
    initial_lr = 0.1
    interval = range(int(0.5 * total_epochs),int(.75 * total_epochs))
    if epoch < min(interval):
        return initial_lr
    elif epoch in interval:
        return float(initial_lr/10)
    else:
        return float(initial_lr/100)   

scheduler = tf.keras.callbacks.LearningRateScheduler(custom_lr_adjuster, 
                                                     verbose=1)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
              metrics=['accuracy'])

#fit the model for the unnormalized feature vectors

model_history = model.fit(train_gen, validation_data=test_gen, epochs=150,
                          steps_per_epoch=int(np.ceil(train_img_count/batch_size)),
                          validation_steps=int(np.ceil(test_img_count/batch_size)),
                          callbacks=[scheduler])

#predict the output and get the confusion matrix and classification report

X_test_gen_data = np.zeros((test_img_count, IMAGE_SIZE[0], IMAGE_SIZE[1],
                            IMAGE_SIZE[2]))
y_pred = np.zeros((test_img_count, class_count))
y_test = np.zeros((test_img_count, class_count))
y_train = np.zeros((train_img_count, class_count))
y_pred_train = np.zeros((train_img_count, class_count))

#Populate the train and test numpy arrays to calculate metrics
i = 0
for X, y in test_gen:
    batch_samples = len(y)
    X_test_gen_data[i:i+batch_samples] = X
    y_pred[i:i+batch_samples] = model.predict_on_batch(X)
    y_test[i:i+batch_samples] = y
    i += batch_samples
    if(i >= test_img_count):
        break    

i = 0
for X, y in train_gen:
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
plot_loss_wrong_preds(model_history, "Loss Curves - DenseNet_FullyTrained", 
                      X_test_gen_data, y_test, y_pred, class_labels)
 
#print metrics
#print_model_metrics(y_test, y_pred, class_labels, 'Test Metrics')

#save metrics
file_stream = open('results/Ex11.6_Dense121_OdBat_15ep.txt', 'w') 
print("Model Summary\n----------\n", file=file_stream)  
model.summary(print_fn=lambda x: print(x, file=file_stream)) 
print_model_metrics(y_train, y_pred_train, class_labels, 'Train Metrics', file_stream)
print_model_metrics(y_test, y_pred, class_labels, 'Test Metrics', file_stream)
file_stream.close()

