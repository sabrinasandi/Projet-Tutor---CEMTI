#Bibliotheques flask
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory 
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


 

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

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
    
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_mobilenet.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224

	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static',
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
	img_path = "/Users/sandisabrina/CEMTI/flask-project/static/test/"+ str(v_nv)+"/"  + str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursMobileNet.png')	       
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
    
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_mobilenetv2.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224

	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
    directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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
	img_path = "/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/"+ str(v_nv)+"/"+ str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursMobileNetV2.png')	        
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
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_Vgg16.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
    directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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
	img_path = "/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/"+ str(v_nv)+"/"+ str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursVgg16.png')	       
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
    
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_Vgg19.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
    directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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
	img_path = "/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/"+ str(v_nv)+"/"+ str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursVgg19.png')	        
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
    
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_inceptionv3.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
    directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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
	img_path = "/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/"+ str(v_nv)+"/"+ str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursInceptionV3.png')	        
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
    
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_inception_resnetv2.hdf5")
    
	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
    directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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
	img_path = "/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/"+ str(v_nv)+"/" + str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursInceptionResNetV2.png')	        
	prediction_img = "L'image est predite "+str(pred[0][0]*100)+"% non-violente et " + str(pred[0][1]*100)+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violente"

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
    
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_ResNet_101_V2.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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
	img_path = "/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/"+ str(v_nv)+"/"+ str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursResNet101_V2.png')	    
	prediction_img = "L'image est predite "+str(pred[0][0]*100)+"% non-violente et " + str(pred[0][1]*100)+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violente"

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
    
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_ResNet_101.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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
	img_path = "/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/"+ str(v_nv)+"/"+ str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursResNet101.png')	    
	prediction_img = "L'image est predite "+str(round(pred[0][0]*100))+"% non-violente et " + str(round(pred[0][1]*100))+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violente"

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
    
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_ResNet152.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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
	img_path = "/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/"+ str(v_nv)+"/"+ str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursResNet_152.png')	    
	prediction_img = "L'image est predite "+str(round(pred[0][0]*100))+"% non-violente et " + str(round(pred[0][1]*100))+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violente"

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
    
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_DenseNet201.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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
	img_path = "/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/"+ str(v_nv)+"/"+ str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursDenseNet_201.png')	    
	prediction_img = "L'image est predite "+str(round(pred[0][0]*100,2))+"% non-violente et " + str(round(pred[0][1]*100))+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violente"

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
    
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_NasNetLarge.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224


	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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
	img_path = "/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/"+ str(v_nv)+"/"+ str(nom_image)

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
	plt.savefig('/Users/sandisabrina/Documents/CEMTI/flask-project/static/Descripteurs_image/DescripteursNASNetLarge.png')	    
	prediction_img = "L'image est predite "+str(round(pred[0][0]*100,2))+"% non-violente et " + str(round(pred[0][1]*100))+"% violente"


	if(predicted_class_indices[0]==1):
		prediction_img = prediction_img+" et donc cette image est violente"
	else :
		prediction_img = prediction_img+ "et donc cette image est non-violence"

	return prediction_img
	#---------------------Modèles vidéos---------------------#
def Segmentation_video(path_video,V_Nv):
    list_fichier = os.listdir("/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/violence")
    # Suppression des images figurant dans le repertoire test 
    for fichier in list_fichier:
        os.remove("/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/violence/"+fichier+"")
    list_fichier = os.listdir("/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/violence")
    for fichier in list_fichier:
        os.remove("/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/non-violence/"+fichier+"")
    # Segmentation de la vidéo 
    vidcap = cv2.VideoCapture(path_video)
    success,image = vidcap.read()
    count = 0
    while success:
        if(V_Nv == "non-violence"):     
            cv2.imwrite("/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/non-violence/frame%d.jpg" % count, image)     # save frame as JPEG file
        else:
            cv2.imwrite("/Users/sandisabrina/Documents/CEMTI/flask-project/static/test/violence/frame%d.jpg" % count, image)     # save frame as JPEG file
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
	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_ResNet_101.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224

	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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

	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_ResNet152.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224

	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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

	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_NasNetLarge.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224

	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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

	model.load_weights("/Users/sandisabrina/Documents/CEMTI/flask-project/static/best_DenseNet201.hdf5")

	BATCH_SIZE_TESTING = 1
	img_size = 224
	data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
	test_generator = data_generator.flow_from_directory(
	directory = '/Users/sandisabrina/Documents/CEMTI/flask-project/static/test',
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

				
#####################################################################################################################
@app.route('/' , methods =['GET','POST'])
def login():

	if 'loggedin' in session :
		# User is loggedin show them the home page
		cursor.execute('SELECT Nom,Prenom FROM Utilisateur WHERE Email = ?' ,(session['username'], ))
		account = cursor.fetchone()

		date = datetime.datetime.now()
		h = date.hour
		m = date.minute
		s = date.second
		return render_template('Accueil.html', nom = account[0] , prenom = account[1],heure=h,minute=m,seconde=s)

	else:
		# Output message if something goes wrong...
	    msg = ''
	    # Check if "username" and "password" POST requests exist (user submitted form)
	    if request.method == 'POST' and 'Email' in request.form and 'Mot_de_passe' in request.form:
	        # Create variables for easy access
	        username = request.form['Email']
	        password = request.form['Mot_de_passe']
	        # Check if account exists using sqlite3
	        cursor.execute('SELECT * FROM Utilisateur WHERE Email = ? AND Mot_de_passe = ?', (username, password,))
	        # Fetch one record and return result
	        account = cursor.fetchone()
	        # If account exists in accounts table in out database
	        if account:
	            # Create session data, we can access this data in other routes
	            session['loggedin'] = True
	            session['username'] = account[0] # email dans notre cas (dans la bdd)
	            # Redirect to home page

	            Nom = account[1]
	            Prenom = account[2]
	            return redirect(url_for('Accueil',Nom=Nom, Prenom=Prenom))
	        else:
	            # Account doesnt exist or username/password incorrect
	            msg = 'Incorrect username/password!'
	    # Show the login form with message (if any)
	    return render_template('index.html', msg=msg)
	    # http://localhost:5000/python/logout - this will be the logout page

@app.route("/logout")
def logout():
	return render_template('index.html')



@app.route('/register' , methods =['GET','POST'])
def register():
	 # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'Email' in request.form and 'Nom' in request.form and 'Prenom' in request.form and 'Age' in request.form and 'Sexe' in request.form and 'Profession' in request.form and 'Centre_interet' in request.form and 'Mot_de_passe' in request.form:

            
        # Create variables for easy access
        email = request.form['Email']
        nom = request.form['Nom']
        prenom = request.form['Prenom']
        age = request.form['Age']
        sexe = request.form['Sexe']
        profession = request.form['Profession']
        centre_interet = request.form['Centre_interet']
        password = request.form['Mot_de_passe']
       
        cursor.execute('SELECT * FROM Utilisateur WHERE Email = ?', (email,))
        account = cursor.fetchone()

        # If account exists show error and validation checks
        if account:
            msg = 'Compte deja existant !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Email invalide !'
        elif not re.match(r'[A-Za-z0-9]+', nom):
            msg = 'Nom invalide !'
        elif not re.match(r'[A-Za-z0-9]+', prenom):
            msg = 'Prenom invalide !'
        elif not re.match(r'[0-9]+', age):
            msg = 'Age invalide !'

        elif not email or not password or not age or not nom or not prenom or not sexe or not profession or not centre_interet:
            msg = 'Veuillez saisir tous les champs !'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO Utilisateur VALUES (?, ?, ?, ?,?,?,?,?)', (email, nom, prenom, age, sexe, profession, centre_interet, password,))
            connection.commit()
            msg = 'Votre enregistrement a bien été effectué !'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Veuillez remplir le formulaire !'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)



@app.route('/Accueil', methods =['GET','POST'])
def Accueil():
	date = datetime.datetime.now()
	h = date.hour
	m = date.minute
	s = date.second
	# Check if user is loggedin
	if 'loggedin' in session :
		if request.method == 'POST':
			# User is loggedin show them the home page
			cursor.execute('SELECT Nom,Prenom FROM Utilisateur WHERE Email = ?' ,(session['username'], ))
			account = cursor.fetchone()
			# A modifier apres avoir fait le trigger de la table statistique 
			cursor.execute('SELECT Type_media,URI,Label FROM Explicabilite GROUP BY URI')
			csv = cursor.fetchall()
			csv = pd.DataFrame(csv)
			csv.to_csv("/Users/sandisabrina/Documents/CEMTI/flask-project/test.csv",header=",Type_media, URI, Label")
			return render_template('Accueil.html', nom = account[0] , prenom = account[1],heure=h,minute=m,seconde=s)
		else :
			# User is loggedin show them the home page
			cursor.execute('SELECT Nom,Prenom FROM Utilisateur WHERE Email = ?' ,(session['username'], ))
			account = cursor.fetchone()
			#Displays(account)
			#msg = 'Accès accueil !'
			#return render_template('Accueil.html', nom = account[0] , prenom = account[1],heure=h,minute=m,seconde=s)
			return render_template('Accueil.html', nom=account[0] , prenom=account[1] ,heure=h,minute=m,seconde=s)
			#return("/Users/sandisabrina/Documents/CEMTI/flask-project/templates/Accueil.html")
			#return redirect(url_for('Accueil', nom=account[0] , prenom=account[1] ,heure=h,minute=m,seconde=s,msg=msg))

			#return render_template('Accueil.html')
	# User is not loggedin redirect to login page
	return  render_template('index.html')



@app.route("/Compte", methods =['GET','POST'])
def Compte():
	if 'loggedin' in session:
		cursor.execute('SELECT * FROM Utilisateur WHERE Email = ?' ,(session['username'],))
		account = cursor.fetchone()
		#return render_template('Compte.html')
	return render_template('Compte.html', Email = account[0] , Nom = account[1] , Prenom = account[2] , Age = account[3] , Profession = account[5] , Centre_interet = account[6] , Mot_de_passe = account[7] , sexe = account[4])
	



@app.route('/Prediction')
def Prediction():
	if 'loggedin' in session:
		# User is loggedin show them the home page
		cursor.execute('SELECT Nom,Prenom FROM Utilisateur WHERE Email = ?' ,(session['username'],))
		account = cursor.fetchone()

		return render_template('Prediction.html', nom = account[0] , prenom = account[1])
	# User is not loggedin redirect to login page
	return  render_template('index.html')


@app.route('/Historique', methods =['GET','POST'])
def Historique():
	
	if 'loggedin' in session:
		cursor.execute('SELECT Date,Modele_pred,Description_utilisateur,Description_systeme FROM Explicabilite WHERE Utilisateur_id = ? ',(session['username'],))
		Account = cursor.fetchall()
		cursor.execute('SELECT * FROM Explicabilite WHERE Utilisateur_id = ? ',(session['username'],))
		Acc = cursor.fetchone()
		cursor.execute('SELECT Utilisateur_id FROM Explicabilite WHERE Utilisateur_id = ? ',(session['username'],))
		user = cursor.fetchone()
		msg = 'non-violente'
		return render_template('Historique.html',User=user,Account=Account, Type_media = Acc[3] , Total_annotation = 28 , Violence_annotation = 0 , Nviolence_annotation = 3 , Violence_pourcentage = 0.0, Nviolence_pourcentage = 1.0 , Prediction = msg)
		#return render_template('Historique.html' , User = session['username'])
	return  render_template('index.html')




@app.route('/Accueil/Prediction/Descripteurs' , methods =['POST'])
def Descripteurs():
	if 'loggedin' in session:
		# User is loggedin show them the home page
		date = datetime.datetime.now()
		h = date.hour
		m = date.minute
		s = date.second
		email = session['username']
		# Le nom de l'image
		image = request.form['Image']
		

		# Le path  de l'image à revoir après 
		#img_file = request.files['file']
		#img_file.save('/home/nedra/projetCemti/flask-project/static/image_saved',image)
		annotation = request.form['Annotation']
		heure = "" + str(h) + "h:" + str(m) + "m:" + str(s) + "s"
		date = ""+str(date.day)+"/" +str(date.month)+"/"+str(date.year)
		id_prediction = str(uuid.uuid4())
		type_media = request.form['Type_fichier']
		modele = request.form['modele_image']
		# violence / non violence 
		label = request.form['Label']
		return render_template('Descripteurs.html' , Image = image , Annotation = annotation , Type_media = type_media , Modele = modele , AnnotationSYS = label)
	return  render_template('index.html')
		