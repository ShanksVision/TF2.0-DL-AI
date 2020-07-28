# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 07:59:51 2020

@author: Shankar J

Simple GAN using MNIST dataset
"""

import tensorflow as tf
import numpy as np
import sys, os
import matplotlib.pyplot as plt

#Import the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#Normalize the data set between -1 and 1, mean centered at 0
X_train = 2 * (X_train/255.0) - 1
X_test = 2 * (X_test/255.0) - 1

#Latent space dimension for generating image frm noise
latent_dim = 200

#Flatten the data. We will use not use conv networks just simple ANN
N, H, W = X_train.shape
D = H * W
X_train = X_train.reshape(-1, D)
X_test = X_test.reshape(-1, D)

#Now create the generator and discriminator model
def build_generator(noise_dim):
    i = tf.keras.Input(shape=(noise_dim,))
    x = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(0.2))(i)
    x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
    x = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(0.2))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(0.2))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.7)(x)
    x = tf.keras.layers.Dense(D, activation='tanh')(x)
    model = tf.keras.models.Model(i, x)
    return model

def build_discriminator(flattened_vec):
    i = tf.keras.Input(shape=(flattened_vec,))
    x = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(0.2))(i)
    x = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(0.2))(x)
    x = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(0.2))(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(i, x)
    return model

#Build the actual models and create the combined new model
discriminator = build_discriminator(D)
discriminator.compile(tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'],
                      loss='binary_crossentropy')

generator = build_generator(latent_dim)
z = tf.keras.Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False #freeze the discriminator
fake_pred = discriminator(img)
gan_model = tf.keras.models.Model(z, fake_pred) #combined model
gan_model.compile(tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

#setup parameters, variables for training the GAN
batch_size = 64
epochs = 25000
sample_data_gen_freq = 300 #Every 500 steps save some data

#Batch y labels to use during training
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

# loss history vars
d_loss_history = []
g_loss_history = []

#Create a folder to store created images
if not os.path.exists('../Data/gan_gen_img'):
    os.makedirs('../Data/gan_gen_img')
    
#A function to plot gen images and save it
def sample_images(epoch):
  rows, cols = 5, 5
  noise = np.random.randn(rows * cols, latent_dim)
  imgs = generator.predict(noise)

  # Rescale images 0 - 1
  imgs = 0.5 * imgs + 0.5

  fig, axs = plt.subplots(rows, cols)
  idx = 0
  for i in range(rows):
    for j in range(cols):
      axs[i,j].imshow(imgs[idx].reshape(H, W), cmap='gray')
      axs[i,j].axis('off')
      idx += 1
  fig.savefig("../Data/gan_gen_img/%d.png" % epoch)
  plt.close()

#Train the GAN
for epoch in range(epochs):
    #***********Train discrimintor *********
    #Generate real imgs
    rand_idx = np.random.randint(0, len(X_train), batch_size)
    real_img = X_train[rand_idx]
    
    #generate fake image
    rand_vec = np.random.randn(batch_size, latent_dim)
    fake_img = generator.predict(rand_vec)
    
    d_loss_real, d_acc_real = discriminator.train_on_batch(real_img, ones)
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_img, zeros)
    
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_acc = 0.5 * (d_acc_real + d_acc_fake)
    
    #*********Train Generator ***************
    rand_vec = np.random.randn(batch_size, latent_dim)
    g_loss = gan_model.train_on_batch(rand_vec, ones)
    
    rand_vec = np.random.randn(batch_size, latent_dim)
    g_loss = gan_model.train_on_batch(rand_vec, ones)
    
    #****Save loss data and gen images and log training 
    d_loss_history.append(d_loss)
    g_loss_history.append(g_loss)
  
    if epoch % 100 == 0:
        print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f}, \
             d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")
  
    if epoch % sample_data_gen_freq == 0:
        sample_images(epoch)
    
plt.plot(d_loss_history, label='Discriminator loss')
plt.plot(g_loss_history, label='Generator loss')
plt.legend()
plt.title('Loss in GAN Training')
plt.show()

print('*******Discriminator*******\n')
discriminator.summary()

print('*******Generator*******\n')
generator.summary()

print('*******Combined GAN Model*******\n')
gan_model.summary()