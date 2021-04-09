#!/usr/bin/env python
# coding: utf-8

# In[49]:


from tensorflow import keras

train_datagen = keras.preprocessing.image.ImageDataGenerator()


# In[50]:


import os
import tensorflow as tf
from tensorflow import keras

base_dir = '/Users/sandisabrina/Documents/CEMTI/flask-project/data/'

train_dir = os.path.join(base_dir, 'train')

image_size = 224


train_datagen = keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(image_size, image_size))


# In[51]:


base_dir = '/Users/sandisabrina/Documents/CEMTI/flask-project/data/'

validation_dir = os.path.join(base_dir, 'test')

image_size = 224

validation_datagen = keras.preprocessing.image.ImageDataGenerator()

validation_generator = validation_datagen.flow_from_directory(directory=validation_dir, target_size=(image_size, image_size))


# In[53]:


import tensorflow as tf

base_model = tf.keras.applications.MobileNet()
base_model.summary()


# In[64]:


import numpy
batch_size = 32
base_model = tf.keras.applications.MobileNet(input_shape=(224,224,3), include_top=False)
base_model.trainable = False
base_model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(2, activation='sigmoid')
])
base_model.summary()
base_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
epochs = 25
steps_per_epoch = numpy.ceil(train_generator.n / batch_size)
validation_steps = numpy.ceil(validation_generator.n / batch_size)

history = base_model.fit_generator(generator=train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_steps)

base_model.save('MobileNet_TransferLearning_violence.h5')


# In[ ]:




