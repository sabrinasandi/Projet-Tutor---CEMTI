#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# In[2]:


from tensorflow import keras

train_datagen = keras.preprocessing.image.ImageDataGenerator()
import os
import tensorflow as tf
from tensorflow import keras

base_dir = '/Users/sandisabrina/Documents/CEMTI/flask-project/data/'

train_dir = os.path.join(base_dir, 'train')

image_size = 224


train_datagen = keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(image_size, image_size))

base_dir = '/Users/sandisabrina/Documents/CEMTI/flask-project/data/'

validation_dir = os.path.join(base_dir, 'test')

image_size = 224

validation_datagen = keras.preprocessing.image.ImageDataGenerator()

validation_generator = validation_datagen.flow_from_directory(directory=validation_dir, target_size=(image_size, image_size))

import numpy
batch_size = 32


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
base_model.trainable = True
base_model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(2, activation='sigmoid')
])
base_model.summary()
base_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
epochs = 15
steps_per_epoch = numpy.ceil(train_generator.n / batch_size)
validation_steps = numpy.ceil(validation_generator.n / batch_size)

history = base_model.fit_generator(generator=train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_steps)

base_model.save('MobileNet_TransferLearning_violence.h5')


# In[ ]:




