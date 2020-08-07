# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 19:38:13 2020

@author: Shankar J

Create a custom model using tf keras with all underlying bells and whistles
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ThreeParamRadialDistortion(tf.keras.Model):
    def __init__(self, input_dim):
        super(ThreeParamRadialDistortion, self).__init__(dtype='float64')
        self.k = tf.Variable(tf.random_normal_initializer()((3, 1), dtype='float64'))
        # self.k0 = tf.Variable(np.random.randn(1))
        # self.k1 = tf.Variable(np.random.randn(1))
        # self.k2 = tf.Variable(np.random.randn(1))
        self.params = [self.k]
        self.dims = input_dim
        if input_dim not in range(1,3):
            raise ValueError("Number of input dim need to be 1 or 2")     
        
    def call(self, inputs):   
        if inputs.shape[1] != self.dims:
            raise ValueError("Input data dimension mismatch")
            
        #r = np.linalg.norm(inputs, axis=1).reshape(-1, 1)
        #r_series = np.concatenate((r**2, r**4, r**6), axis=1)
        r = tf.norm(inputs, axis=1)
        r = tf.reshape(r, (-1, 1))
        r_series = tf.concat([r**2, r**4, r**6], 1)
        outputs = inputs + (inputs * (tf.matmul(r_series, self.k)))
        #outputs = inputs / (1 + (tf.matmul(r_series, self.k)))
        return outputs
        
#Import the dataset

data = pd.read_csv('distortion_data.csv', sep=';', header=0)
N = len(data)
D = 2
K = 2
X = data.iloc[:,0:2].to_numpy()
y = data.iloc[:,2:].to_numpy()
pixel_to_mm = 29.379

X_scaled = X/29.379
Xscaler = StandardScaler()
yscaler = StandardScaler()
X_norm = Xscaler.fit_transform(X)
y_norm = yscaler.fit_transform(y)

#Plot the data points
plt.plot(X_norm[:,0], X_norm[:,1], 'r.', label='Distorted')
plt.plot(y_norm[:,0], y_norm[:,1], 'b.', label='Undistorted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Undistored and distorted data points')
plt.legend()
plt.show()

def get_loss(model, inputs, targets):
    preds = model(inputs)
    err = targets - preds
    #print("Error : ", err)
    loss = tf.reduce_mean(tf.square(err))
    
    return loss

def get_grad(model, inputs, targets, epoch):
    with tf.GradientTape() as tape:
        loss_value = get_loss(model, inputs, targets)
        print("Epoch #:", epoch, ", Loss:", loss_value.numpy())
   
    grad = tape.gradient(loss_value, model.params)
    #print("Gradient #:", grad)
    
    return grad
    
#Create the model
model = ThreeParamRadialDistortion(2)

print("Initial Params")
print(model.k)

#Train the model
losses = []

optimizer = tf.keras.optimizers.SGD(0.1, momentum=0.9, nesterov=True)

# Run the training loop
for i in range(1000):
  # Get gradients
  grads = get_grad(model, X_norm, y_norm, i)
  
  # Do one step of gradient descent: param <- param - learning_rate * grad
  optimizer.apply_gradients(zip(grads, model.params))
  
  # Store the losstf.
  loss = get_loss(model, X_norm, y_norm)
  losses.append(loss)

plt.plot(losses)
plt.title('Loss curve')
plt.show()

#Predict the output
y_pred = model.predict(X_norm)
y_pred_actual = yscaler.inverse_transform(y_pred)

#plot the fitted points
# plt.plot(y_norm[:,0], y_norm[:,1], 'r.', label='Distorted')
# plt.plot(y_pred[:,0], y_pred[:,1], 'b.', label='Undistorted_predicted')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Undistored and distorted data points')
# plt.legend()
# plt.show()

#plot fitted points
plt.plot(y[:,0], y[:,1], 'r.', label='Distorted')
plt.plot(y_pred_actual[:,0], y_pred_actual[:,1], 'b.', label='Undistorted_predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Undistored and distorted data points')
plt.legend()
plt.show()

print("Final Params")
print(model.k)

print("Mean RMS Error")
print(tf.reduce_mean(tf.square(y - y_pred_actual)))
