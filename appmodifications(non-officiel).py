#Bibliotheques flask
from flask import Flask, render_template, request, redirect, url_for, session
import datetime
# Expressions regulière 
import re
import os
# Bibliothèque ResNet 
import tensorflow as tf
import keras
from keras import Model
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet import preprocess_input
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


# Generate GUID ID 
import uuid

# BDD SQLITE3
import sqlite3


# Connexion BDD
connection = sqlite3.connect("CEMTI.db", check_same_thread=False)
cursor = connection.cursor()

# Nous créons un objet app : cette ligne est systématique nécessaire.
app = Flask(__name__)
app.permanent_session_lifetime =datetime.timedelta(days=365)
app.secret_key = 'RayWass'
chemin_image_descripteurs = ""

dir ='/Users/sandisabrina/Documents/CEMTI/flask-project'

def ResultatViolenteOuPasViolente():
	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img


########################## Modèles #############################################
def MobileNet(nom_image,v_nv):

	MobileNet = tf.keras.applications.MobileNet(
    include_top=False,
    pooling="avg"
    )
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
    # Transfert Learning 
	model = tf.keras.Sequential()
	model.add(MobileNet)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
    
	model.load_weights(dir+"/static/best_mobilenet.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224

	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = dir+'/static',
    target_size = (img_size,img_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
    )
    
	test_generator.reset()

	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in MobileNet.layers[:88]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in MobileNet.layers[:88]]
	activation_model = tf.keras.models.Model(inputs=MobileNet.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/"  + str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 3 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
# Dans notre cas (3/1) et donc il affichera 3 images pour chaque ligne



	n_features = activations[86].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[86].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[86][0,
	        								:, :,
											col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, # Displays the grid
						row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
	plt.title(layer_names[87])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursMobileNet.png')	       
	prediction_img = "L'image est predite "+str(pred[0][0]*100)+"% non-violente et " + str(pred[0][1]*100)+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img



def  MobileNet_V2(nom_image,v_nv):
	MobileNetV2 = tf.keras.applications.MobileNetV2(
    include_top=False,
    pooling="avg"
    )
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
    # Transfert Learning 
	model = tf.keras.Sequential()
	model.add(MobileNetV2)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
    
	model.load_weights(dir+"/static/best_mobilenetv2.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224

	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
    directory = dir+'/static/test',
    target_size = (img_size,img_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
    )
    
	test_generator.reset()

	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in MobileNetV2.layers[:88]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in MobileNetV2.layers[:88]]
	activation_model = tf.keras.models.Model(inputs=MobileNetV2.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/"+ str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 3 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
# Dans notre cas (3/1) et donc il afficheras 3 images pour chaque ligne



	n_features = activations[86].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[86].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[86][0,:, :, col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
	plt.title(layer_names[87])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursMobileNetV2.png')	        
	prediction_img = "L'image est predite "+str(pred[0][0]*100)+"% non-violente et " + str(pred[0][1]*100)+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img
	



def Vgg16(nom_image,v_nv):
	VGG16 = tf.keras.applications.VGG16(
    include_top=False,
    pooling="avg"
    )
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
    # Transfert Learning 
	model = tf.keras.Sequential()
	model.add(VGG16)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
	model.load_weights(dir+"/static/best_Vgg16.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
    directory = dir+'/static/test',
    target_size = (img_size,img_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
    )
	test_generator.reset()

	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in VGG16.layers[:20]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in VGG16.layers[:20]]
	activation_model = tf.keras.models.Model(inputs=VGG16.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/"+ str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 3 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
# Dans notre cas (3/1) et donc il afficheras 3 images pour chaque ligne



	n_features = activations[18].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[18].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[18][0,:, :,col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
	plt.title(layer_names[19])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursVgg16.png')	       
	prediction_img = "L'image est predite "+str(pred[0][0]*100)+"% non-violente et " + str(pred[0][1]*100)+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img



def Vgg19(nom_image,v_nv):
	VGG19 = tf.keras.applications.VGG19(
    include_top=False,
    pooling="avg"
    )
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
    # Transfert Learning 
	model = tf.keras.Sequential()
	model.add(VGG19)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
    
	model.load_weights(dir+"/static/best_Vgg19.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
    directory = dir+'/static/test',
    target_size = (img_size,img_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
    )
    
	test_generator.reset()

	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in VGG19.layers[:23]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in VGG19.layers[:23]]
	activation_model = tf.keras.models.Model(inputs=VGG19.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/"+ str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 3 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
# Dans notre cas (3/1) et donc il afficheras 3 images pour chaque ligne



	n_features = activations[21].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[21].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[21][0,:, :, col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
	plt.title(layer_names[22])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursVgg19.png')	        
	prediction_img = "L'image est predite "+str(pred[0][0]*100)+"% non-violente et " + str(pred[0][1]*100)+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img



def Inception_V3(nom_image,v_nv):
	InceptionV3 = tf.keras.applications.InceptionV3(
    include_top=False,
    pooling="avg"
    )
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
    # Transfert Learning 
	model = tf.keras.Sequential()
	model.add(InceptionV3)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
    
	model.load_weights(dir+"/static/best_inceptionv3.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
    directory = dir+'/static/test',
    target_size = (img_size,img_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
    )
    
	test_generator.reset()

	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in InceptionV3.layers[:88]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in InceptionV3.layers[:88]]
	activation_model = tf.keras.models.Model(inputs=InceptionV3.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/"+ str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 3 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
# Dans notre cas (3/1) et donc il afficheras 3 images pour chaque ligne



	n_features = activations[86].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[86].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[86][0,:, :,col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
	plt.title(layer_names[87])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursInceptionV3.png')	        
	prediction_img = "L'image est predite "+str(pred[0][0]*100)+"% non-violente et " + str(pred[0][1]*100)+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img



def Inception_ResNet_V2(nom_image,v_nv):
	InceptionResNetV2 = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    pooling="avg"
    )
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
    # Transfert Learning 
	model = tf.keras.Sequential()
	model.add(InceptionResNetV2)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
    
	model.load_weights(dir+"/static/best_inception_resnetv2.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
    directory = dir+'/static/test',
    target_size = (img_size,img_size),
    batch_size = BATCH_SIZE_TESTING,
    class_mode = None,
    shuffle = False,
    seed = 123
    )
    
	test_generator.reset()

	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in InceptionResNetV2.layers[:88]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in InceptionResNetV2.layers[:88]]
	activation_model = tf.keras.models.Model(inputs=InceptionResNetV2.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/" + str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 3 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
# Dans notre cas (3/1) et donc il afficheras 3 images pour chaque ligne



	n_features = activations[86].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[86].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[86][0,:, :,col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
	plt.title(layer_names[87])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursInceptionResNetV2.png')	        
	prediction_img = "L'image est predite "+str(pred[0][0]*100)+"% non-violente et " + str(pred[0][1]*100)+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img


############################################# Modèles Rayene ###############################################################

 ##### IMAGES
 ## ResNetV2 je ne l'utilise pas 
def ResNet101_V2(nom_image,v_nv):	
	ResNet_101_V2 = tf.keras.applications.ResNet101V2(
	include_top=False,
	pooling="avg"
	)
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
	# Transfert Learning 
	model = tf.keras.Sequential()
	model.add(ResNet_101_V2)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
    
	model.load_weights(dir+"/static/best_ResNet_101_V2.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = dir+'/static/test',
	target_size = (img_size,img_size),
	batch_size = BATCH_SIZE_TESTING,
	class_mode = None,
	shuffle = False,
	seed = 123
	)
    
	test_generator.reset()
	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in ResNet_101_V2.layers[:176]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in ResNet_101_V2.layers[:176]]
	activation_model = tf.keras.models.Model(inputs=ResNet_101_V2.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/"+ str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 2 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
	# Dans notre cas (3/1) et donc il afficheras 3 images pour chaque ligne



	n_features = activations[175].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[175].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[175][0,:, :,col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
	plt.title(layer_names[175])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursResNet101_V2.png')	    
	prediction_img = "L'image est predite "+str(pred[0][0]*100)+"% non-violente et " + str(pred[0][1]*100)+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img


def ResNet101(nom_image,v_nv):
	
	ResNet_101 = tf.keras.applications.ResNet101(
	include_top=False,
	pooling="avg"
	)
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
	# Transfert Learning 
	model = tf.keras.Sequential()
	model.add(ResNet_101)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
    
	model.load_weights(dir+"/static/best_ResNet_101.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = dir+'/static/test',
	target_size = (img_size,img_size),
	batch_size = BATCH_SIZE_TESTING,
	class_mode = None,
	shuffle = False,
	seed = 123
	)
    
	test_generator.reset()
	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in ResNet_101.layers[:176]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in ResNet_101.layers[:176]]
	activation_model = tf.keras.models.Model(inputs=ResNet_101.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/"+ str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 2 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
# Dans notre cas (3/1) et donc il afficheras 3 images pour chaque ligne



	n_features = activations[175].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[175].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[175][0,:, :,col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
	plt.title(layer_names[175])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursResNet101.png')	    
	prediction_img = "L'image est predite "+str(round(pred[0][0]*100))+"% non-violente et " + str(round(pred[0][1]*100))+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img

	
def ResNet_152(nom_image,v_nv):
	
	ResNet_101 = tf.keras.applications.ResNet152(
	include_top=False,
	pooling="avg"
	)
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
	# Transfert Learning 
	model = tf.keras.Sequential()
	model.add(ResNet_101)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
    
	model.load_weights(dir+"/static/best_ResNet152.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = dir+'/static/test',
	target_size = (img_size,img_size),
	batch_size = BATCH_SIZE_TESTING,
	class_mode = None,
	shuffle = False,
	seed = 123
	)
    
	test_generator.reset()
	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in ResNet_101.layers[:176]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in ResNet_101.layers[:176]]
	activation_model = tf.keras.models.Model(inputs=ResNet_101.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/"+ str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 2 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
# Dans notre cas (3/1) et donc il afficheras 3 images pour chaque ligne



	n_features = activations[175].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[175].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[175][0,:, :,col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
	plt.title(layer_names[175])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursResNet_152.png')	    
	prediction_img = "L'image est predite "+str(round(pred[0][0]*100))+"% non-violente et " + str(round(pred[0][1]*100))+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img


def DenseNet_201(nom_image,v_nv):
	
	ResNet_101 = tf.keras.applications.DenseNet201(
	include_top=False,
	pooling="avg"
	)
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
	# Transfert Learning 
	model = tf.keras.Sequential()
	model.add(ResNet_101)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
    
	model.load_weights(dir+"/static/best_DenseNet201.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = dir+'/static/test',
	target_size = (img_size,img_size),
	batch_size = BATCH_SIZE_TESTING,
	class_mode = None,
	shuffle = False,
	seed = 123
	)
    
	test_generator.reset()
	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in ResNet_101.layers[:176]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in ResNet_101.layers[:176]]
	activation_model = tf.keras.models.Model(inputs=ResNet_101.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/"+ str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 2 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
# Dans notre cas (3/1) et donc il afficheras 3 images pour chaque ligne



	n_features = activations[175].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[175].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[175][0,:, :,col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
	plt.title(layer_names[175])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursDenseNet_201.png')	    
	prediction_img = "L'image est predite "+str(round(pred[0][0]*100,2))+"% non-violente et " + str(round(pred[0][1]*100))+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img



def NASNet_Large(nom_image,v_nv):
	
	NASNetLarge = tf.keras.applications.NASNetLarge(
	include_top=False,
	pooling="avg"
	)
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
	# Transfert Learning 
	model = tf.keras.Sequential()
	model.add(NASNetLarge)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
    
	model.load_weights(dir+"/static/best_NasNetLarge.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = dir+'/static/test',
	target_size = (img_size,img_size),
	batch_size = BATCH_SIZE_TESTING,
	class_mode = None,
	shuffle = False,
	seed = 123
	)
    
	test_generator.reset()
	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
	predicted_class_indices = np.argmax(pred, axis = 1)
	layer_names =[]
	for layer in NASNetLarge.layers[:1040]:
		layer_names.append(layer.name) 
		# Names of the layers, so you can have them as part of your plot
	layer_outputs = [layer.output for layer in NASNetLarge.layers[:1040]]
	activation_model = tf.keras.models.Model(inputs=NASNetLarge.input, outputs=layer_outputs)
	img_path = dir+"/static/test/"+ str(v_nv)+"/"+ str(nom_image)

	img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	img_tensor = keras.preprocessing.image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	# Returns a list of five Numpy arrays: one array per layer activation
	activations = activation_model.predict(img_tensor)
	images_per_row = 2 # nombre d'image per row : il depend du nombre de canneaux (nb_channels/Nb_img_per_row)
# Dans notre cas (3/1) et donc il afficheras 3 images pour chaque ligne



	n_features = activations[500].shape[-1] # Number of features in the feature map (notre cas 3)
	size = activations[500].shape[1] #The feature map has shape (1, size, size, n_features) (notre cas 224) .
	n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols): # Tiles each filter into a big horizontal grid
		for row in range(images_per_row):
			channel_image = activations[500][0,:, :,col * images_per_row + row]
			channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	   
	plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
	plt.title(layer_names[500])
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.savefig(dir+'/static/Descripteurs_image/DescripteursNASNetLarge.png')	    
	prediction_img = "L'image est predite "+str(round(pred[0][0]*100,2))+"% non-violente et " + str(round(pred[0][1]*100))+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img
	#---------------------Modèles vidéos---------------------#
def Segmentation_video(path_video,V_Nv):
    list_fichier = os.listdir(dir+"/static/test/violence")
    # Suppression des images figurant dans le repertoire test 
    for fichier in list_fichier:
        os.remove(dir+"/static/test/violence/"+fichier+"")
    list_fichier = os.listdir(dir+"/static/test/violence")
    for fichier in list_fichier:
        os.remove(dir+"/static/test/non-violence/"+fichier+"")
    # Segmentation de la vidéo 
    vidcap = cv2.VideoCapture(path_video)
    success,image = vidcap.read()
    count = 0
    while success:
        if(V_Nv == "non-violence"):     
            cv2.imwrite(dir+"/static/test/non-violence/frame%d.jpg" % count, image)     # save frame as JPEG file
        else:
            cv2.imwrite(dir+"/static/test/violence/frame%d.jpg" % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        count += 1
        
def ResNet101_video():
	ResNet101 = tf.keras.applications.ResNet101(
		include_top=False,
		pooling="avg"
	)
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
	# Transfert Learning 
	model = tf.keras.Sequential()
	model.add(ResNet101)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False
	model.load_weights(dir+"/static/best_ResNet_101.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224

	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = dir+'/static/test',
	target_size = (img_size,img_size),
	batch_size = BATCH_SIZE_TESTING,
	class_mode = None,
	shuffle = False,
	seed = 123
	)

	test_generator.reset()
	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
	predicted_class_indices = np.argmax(pred, axis = 1)

	prediction_img=""
	moy_violente = 0.0
	moy_Nviolente = 0.0
	for i in range(len(pred)) : 
		#prediction_img =prediction_img + " Frame "+str(i) +"est predite "+str(round(pred[i][1]*100,2))+"% non-violente et " + str(round(pred[i][0]*100))+"% violente"
		moy_violente = moy_violente + pred[i][1]
		moy_Nviolente = moy_Nviolente + pred[i][0]

	if(moy_violente>=moy_Nviolente):
		prediction_img = prediction_img + "La vidéo est prédite : "+ str((moy_violente*100)/len(pred)) +" % Violente et " + str((moy_Nviolente*100)/len(pred)) + "% Non-violente . La vidéo est Violente"
	else :
		prediction_img = prediction_img + "La vidéo est prédite : "+ str((moy_violente*100)/len(pred)) +" % Violente et " + str((moy_Nviolente*100)/len(pred))  + "% Non-violente . La vidéo est Non Violente"
	return prediction_img



def ResNet152_video():
	ResNet152 = tf.keras.applications.ResNet152(
		include_top=False,
		pooling="avg"
	)
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
	# Transfert Learning 
	model = tf.keras.Sequential()
	model.add(ResNet152)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False

	model.load_weights(dir+"/static/best_ResNet152.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224

	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = dir+'/static/test',
	target_size = (img_size,img_size),
	batch_size = BATCH_SIZE_TESTING,
	class_mode = None,
	shuffle = False,
	seed = 123
	)

	test_generator.reset()
	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
	predicted_class_indices = np.argmax(pred, axis = 1)
	prediction_img=""
	moy_violente = 0.0
	moy_Nviolente = 0.0
	for i in range(len(pred)) : 
		#prediction_img =prediction_img + " Frame "+str(i) +"est predite "+str(round(pred[i][1]*100,2))+"% non-violente et " + str(round(pred[i][0]*100))+"% violente"
		moy_violente = moy_violente + pred[i][1]
		moy_Nviolente = moy_Nviolente + pred[i][0]

	if(moy_violente>=moy_Nviolente):
		prediction_img = prediction_img + "La vidéo est prédite : "+ str((moy_violente*100)/len(pred)) +" % Violente et " + str((moy_Nviolente*100)/len(pred)) + "% Non-violente . La vidéo est Violente"
	else :
		prediction_img = prediction_img + "La vidéo est prédite : "+ str((moy_violente*100)/len(pred)) +" % Violente et " + str((moy_Nviolente*100)/len(pred))  + "% Non-violente . La vidéo est Non Violente"
	return prediction_img
def NASNetLarge_video():
	NASNetLarge = tf.keras.applications.NASNetLarge(
		include_top=False,
		pooling="avg"
	)
	prediction_img = ""
	NUMB_CLASSES = 2
	DENSE_LAYER_ACTIVATION = 'softmax'
	# Transfert Learning 
	model = tf.keras.Sequential()
	model.add(NASNetLarge)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False

	model.load_weights(dir+"/static/best_NasNetLarge.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224

	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = dir+'/static/test',
	target_size = (img_size,img_size),
	batch_size = BATCH_SIZE_TESTING,
	class_mode = None,
	shuffle = False,
	seed = 123
	)

	test_generator.reset()
	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
	predicted_class_indices = np.argmax(pred, axis = 1)  

	prediction_img=""
	moy_violente = 0.0
	moy_Nviolente = 0.0
	for i in range(len(pred)) : 
	#prediction_img =prediction_img + " Frame "+str(i) +"est predite "+str(round(pred[i][1]*100,2))+"% non-violente et " + str(round(pred[i][0]*100))+"% violente"
		moy_violente = moy_violente + pred[i][1]
		moy_Nviolente = moy_Nviolente + pred[i][0]
	if(moy_violente>=moy_Nviolente):
		prediction_img = prediction_img + "La vidéo est prédite : "+ str((moy_violente*100)/len(pred)) +" % Violente et " + str((moy_Nviolente*100)/len(pred)) + "% Non-violente . La vidéo est Violente"
	else :
		prediction_img = prediction_img + "La vidéo est prédite : "+ str((moy_violente*100)/len(pred)) +" % Violente et " + str((moy_Nviolente*100)/len(pred))  + "% Non-violente . La vidéo est Non Violente"
	return prediction_img



def DenseNet201_video():
	NASNetLarge = tf.keras.applications.NASNetLarge(
		include_top=False,
		pooling="avg"
	)
	prediction_img = ""
	NUMB_CLASSES = 2	
	DENSE_LAYER_ACTIVATION = 'softmax'
	# Transfert Learning 
	model = tf.keras.Sequential()
	model.add(NASNetLarge)
	model.add(Dense(NUMB_CLASSES, activation = DENSE_LAYER_ACTIVATION))
	model.layers[0].trainable = False

	model.load_weights(dir+"/static/best_DenseNet201.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224
	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = dir+'/static/test',
	target_size = (img_size,img_size),
	batch_size = BATCH_SIZE_TESTING,
	class_mode = None,
	shuffle = False,
	seed = 123
	)

	test_generator.reset()
	pred = model.predict_generator(test_generator, steps = len(test_generator), verbose = 1)
	predicted_class_indices = np.argmax(pred, axis = 1)
	prediction_img=""
	moy_violente = 0.0
	moy_Nviolente = 0.0
	for i in range(len(pred)) : 
		#prediction_img =prediction_img + " Frame "+str(i) +"est predite "+str(round(pred[i][1]*100,2))+"% non-violente et " + str(round(pred[i][0]*100))+"% violente"
		moy_violente = moy_violente + pred[i][1]
		moy_Nviolente = moy_Nviolente + pred[i][0]
	if(moy_violente>=moy_Nviolente):	
		prediction_img = prediction_img + "La vidéo est prédite : "+ str((moy_violente*100)/len(pred)) +" % Violente et " + str((moy_Nviolente*100)/len(pred)) + "% Non-violente . La vidéo est Violente"
	else :
		prediction_img = prediction_img + "La vidéo est prédite : "+ str((moy_violente*100)/len(pred)) +" % Violente et " + str((moy_Nviolente*100)/len(pred))  + "% Non-violente . La vidéo est Non Violente"
	return prediction_img

