#!/usr/bin/env python
# coding: utf-8

# In[1]:


#CIFAR10 and CIFAR100


# In[ ]:


import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

class model:
    @staticmethod
    def build():
        # Load the pre-trained VGG16 model
        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
        # Freeze all layers in the base model
        base_model.trainable = False

        # Create the transfer learning model by adding custom classification layers on top of the base model
        model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')  # Adjust the number of output classes accordingly
        ])

        # Create an optimizer with the specified learning rate
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Compile the model with the custom optimizer
        model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

        return model
    
smlp_global = model()
global_model = model.build()
global_model.summary()


# In[3]:


# FMNIST


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
def build_lenet(input_shape=(28, 28, 1)):
  # Define Sequential Model
  model = tf.keras.Sequential()
  
  # C1 Convolution Layer
  model.add(tf.keras.layers.Conv2D(filters=6, strides=(1,1), kernel_size=(5,5), activation='tanh', input_shape=input_shape))
  
  # S2 SubSampling Layer
  model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))

  # C3 Convolution Layer
  model.add(tf.keras.layers.Conv2D(filters=6, strides=(1,1), kernel_size=(5,5), activation='tanh'))

  # S4 SubSampling Layer
  model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))

  # C5 Fully Connected Layer
  model.add(tf.keras.layers.Dense(units=120, activation='tanh'))

  # Flatten the output so that we can connect it with the fully connected layers by converting it into a 1D Array
  model.add(tf.keras.layers.Flatten())

  # FC6 Fully Connected Layers
  model.add(tf.keras.layers.Dense(units=84, activation='tanh'))

  # Output Layer
  model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

  # Compile the Model
  model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0), metrics=['accuracy'])

  return model


global_model=build_lenet()

