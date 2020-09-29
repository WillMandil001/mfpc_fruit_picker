import cv2
import random
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(1000)
from sklearn import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications.resnet import preprocess_input

from skimage.io import imread
from skimage.transform import resize

################################## Hyper Parameters ##################################################################
data_location = "/home/will/Robotics/Data_sets/"
no_of_cameras = 8 + 1
data_set_length = 164
trajectory_length = 1
scale_up_value = 100000
single_sample_length = 199
train_size__ = 0.8
batch_size = 60

def _array_feature(value):
  value = np.nan_to_num(value.flatten())
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

with tf.device('/gpu:1'):
	################################## Generater to keep ram space Parameters ##################################################################
	# read data:
	robot_positions_train = _array_feature(np.array(pd.read_csv(data_location + 'processed_001/robot_positions_train_FOR_CLUSTER_001', header=None)))
	robot_positions_test = _array_feature(np.array(pd.read_csv(data_location + 'processed_001/robot_positions_test_FOR_CLUSTER_001', header=None)))
	robot_positions_validation = robot_positions_test[0:int(len(robot_positions_test) / 2)] 
	robot_positions_test = robot_positions_test[int(len(robot_positions_test) / 2):len(robot_positions_test)]

	robot_velocitys_train = _array_feature(np.array(pd.read_csv(data_location + 'processed_001/robot_velocitys_train_FOR_CLUSTER_001', header=None)))
	robot_velocitys_test = _array_feature(np.array(pd.read_csv(data_location + 'processed_001/robot_velocitys_test_FOR_CLUSTER_001', header=None)))
	robot_velocitys_validation = robot_velocitys_test[0:int(len(robot_velocitys_test) / 2)] 
	robot_velocitys_test = robot_velocitys_test[int(len(robot_velocitys_test) / 2):len(robot_velocitys_test)]

	strawberry_state_train = _array_feature(np.array(pd.read_csv(data_location + 'processed_001/strawberry_cluster_train_FOR_CLUSTER_001', header=None)))
	strawberry_state_test = _array_feature(np.array(pd.read_csv(data_location + 'processed_001/strawberry_cluster_test_FOR_CLUSTER_001', header=None)))
	strawberry_state_validation = strawberry_state_test[0:int(len(strawberry_state_test) / 2)] 
	strawberry_state_test = strawberry_state_test[int(len(strawberry_state_test) / 2):len(strawberry_state_test)]

	image_file_names_train = _array_feature(np.array([(data_location + "processed_001/images_shuffled/image_" + str(i) + ".png") for i in range(0, len(robot_positions_train))]))
	image_file_names_test = _array_feature(np.array([(data_location + "processed_001/images_shuffled/image_" + str(i) + ".png") for i in range(len(robot_positions_train), (len(robot_positions_train) + len(robot_positions_test)))]))
	image_file_names_validation = image_file_names_test[0:int(len(image_file_names_test) / 2)] 
	image_file_names_test = image_file_names_test[int(len(image_file_names_test) / 2):len(image_file_names_test)]


	class My_Custom_Generator(keras.utils.Sequence) :
	    def __init__(self, file_names, robot_position, robot_velocities, strawberry_state, batch_size) :
	        self.file_names = file_names
	        self.robot_position = robot_position
	        self.robot_velocities = robot_velocities
	        self.strawberry_state = strawberry_state
	        self.batch_size = batch_size

	    def __len__(self) :
	        return (np.ceil(len(self.file_names) / float(self.batch_size))).astype(np.int)

	    def __getitem__(self, idx) :
	        batch_x_img = self.file_names[idx * self.batch_size : (idx+1) * self.batch_size]
	        batch_x_robot_position = self.robot_position[idx * self.batch_size : (idx+1) * self.batch_size]
	        batch_x_robot_velocities = self.robot_velocities[idx * self.batch_size : (idx+1) * self.batch_size]
	        batch_y_strawberry_state = self.strawberry_state[idx * self.batch_size : (idx+1) * self.batch_size]

	        return [np.array([resize(imread(file_name), (224, 224, 3)) for file_name in batch_x_img]), np.array(batch_x_robot_position), np.array(batch_x_robot_velocities)], np.array(batch_y_strawberry_state)

	my_training_batch_generator = My_Custom_Generator(image_file_names_train, robot_positions_train, robot_velocitys_train, strawberry_state_train, batch_size)
	my_testing_batch_generator = My_Custom_Generator(image_file_names_test, robot_positions_test, robot_velocitys_test, strawberry_state_test, batch_size)
	my_validation_batch_generator = My_Custom_Generator(image_file_names_validation, robot_positions_validation, robot_velocitys_validation, strawberry_state_validation, batch_size)

	########################################################## Define AlexNet CNN ####################################################
	########################################################## Define AlexNet CNN ####################################################
	########################################################## Define AlexNet CNN ####################################################
	########################################################## Define AlexNet CNN ####################################################
	########################################################## Define AlexNet CNN ####################################################
	########################################################## Define AlexNet CNN ####################################################
	image_input_layer = keras.layers.Input(shape=(224,224,3))

	layer_conv_1 = keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation="relu")(image_input_layer)
	layer_pooling_1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(layer_conv_1)

	layer_conv_2 = keras.layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid', activation="relu")(layer_pooling_1)
	layer_pooling_2 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(layer_conv_2)

	layer_conv_3 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation="relu")(layer_pooling_2)
	layer_conv_4 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation="relu")(layer_conv_3)
	layer_conv_5 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation="relu")(layer_conv_4)

	layer_pooling_3 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(layer_conv_5)

	cnn_flatten = keras.layers.Flatten()(layer_pooling_3)

	dense_1 = keras.layers.Dense(4096, activation="relu")(cnn_flatten)
	drop_1 = keras.layers.Dropout(0.4)(dense_1)
	dense_2 = keras.layers.Dense(4096, activation="relu")(drop_1)
	drop_2 = keras.layers.Dropout(0.4)(dense_2)

	robot_pose_input_layer = keras.layers.Input(shape=((14),))
	dense_3_1 = keras.layers.Dense(64, activation="relu")(robot_pose_input_layer)
	dense_4_1 = keras.layers.Dense(32, activation="relu")(dense_3_1)

	robot_vel_input_layer = keras.layers.Input(shape=(14,))
	dense_3_2 = keras.layers.Dense(64, activation="relu")(robot_vel_input_layer)
	dense_4_2 = keras.layers.Dense(32, activation="relu")(dense_3_2)

	concat = keras.layers.concatenate([dense_3_1, dense_3_2, drop_2])

	dense_5 = keras.layers.Dense(64, activation="relu")(concat)
	dense_6 = keras.layers.Dense(128, activation="relu")(dense_5)
	dense_7 = keras.layers.Dense(64, activation="relu")(dense_6)
	dense_8 = keras.layers.Dense(32, activation="relu")(dense_7)
	output_layer = keras.layers.Dense(21, activation="linear")(dense_8)

	model = keras.models.Model(inputs=[image_input_layer, robot_pose_input_layer, robot_vel_input_layer] , outputs=output_layer)
	intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=dense_5) 

	# Compile the model
	model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mae'])
	model.summary()

	"""**Train the model**"""
	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='auto', restore_best_weights=True)

	history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=int(len(robot_positions_train) // batch_size), validation_data=my_validation_batch_generator, callbacks=[monitor], epochs=10)
	score = model.evaluate([images_test, robot_positions_test, robot_velocitys_test] , strawberry_cluster_test)

	##################################### save Model ###########################################################################
	model.save('/content/drive/My Drive/Masters Year/data_mfpc/AlexNet_full_001.h5')
	intermediate_layer_model.save('/content/drive/My Drive/Masters Year/data_mfpc/AlexNet_intermediate_layer_model_001.h5')
	print("model.summary()")
	model.summary()
	print("intermediate_layer_model.summary()")
	intermediate_layer_model.summary()
	try:
		predict_AlexNet_dense = model.predict([images_test, robot_positions_test, robot_velocitys_test])
		model.summary()
		err_matrix_customNet_dense = strawberry_cluster_test - predict_AlexNet_dense
		customNet_err_mean = np.mean(abs(err_matrix_customNet_dense))
		print("AlexNet mean error values for each output: ")
		print(customNet_err_mean)


		print([i / 100000 for i in predict_AlexNet_dense[250]])
		print([float(i / 100000) for i in strawberry_cluster_test[250]])
		print("------------------------------------------------------------------------------")
		print([i / 100000 for i in predict_AlexNet_dense[251]])
		print([float(i / 100000) for i in strawberry_cluster_test[251]])
	except:
		pass

	########################################################## Define ResNet CNN ####################################################
	########################################################## Define ResNet CNN ####################################################
	########################################################## Define ResNet CNN ####################################################
	########################################################## Define ResNet CNN ####################################################
	########################################################## Define ResNet CNN ####################################################
	########################################################## Define ResNet CNN ####################################################
	model = ResNet152V2(include_top=False, weights='imagenet', input_shape=(224 , 224 , 3))

	y1 = model.output
	y2 = GlobalAveragePooling2D()(y1)
	y3 = Dense(1024,activation='relu')(y2) 
	y4 = Dense(1024,activation='relu')(y3)
	new_model = Model(inputs=model.input,outputs=y4)

	for layer in new_model.layers[:561]:
	    layer.trainable=False
	for layer in new_model.layers[561:]:
	    layer.trainable=True
	cnn_out = new_model.output

	robot_pose_input_layer = keras.layers.Input(shape=((7*(trajectory_length+1)),))
	dense_3_1 = keras.layers.Dense(64, activation="relu")(robot_pose_input_layer)
	dense_4_1 = keras.layers.Dense(32, activation="relu")(dense_3_1)

	robot_vel_input_layer = keras.layers.Input(shape=((7*(trajectory_length+1)),))
	dense_3_2 = keras.layers.Dense(64, activation="relu")(robot_vel_input_layer)
	dense_4_2 = keras.layers.Dense(32, activation="relu")(dense_3_2)

	concat = keras.layers.concatenate([dense_3_1, dense_3_2, cnn_out])

	dense_5 = keras.layers.Dense(64, activation="relu")(concat)
	dense_6 = keras.layers.Dense(128, activation="relu")(dense_5)
	dense_7 = keras.layers.Dense(64, activation="relu")(dense_6)
	dense_8 = keras.layers.Dense(32, activation="relu")(dense_7)
	output_layer = keras.layers.Dense((21), activation="linear")(dense_8)

	model = keras.models.Model(inputs=[image_input_layer, robot_pose_input_layer, robot_vel_input_layer] , outputs=output_layer)
	intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=dense_5) 

	# Compile the model
	model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mae'])
	model.summary()

	"""**Train the model**"""
	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='auto', restore_best_weights=True)

	history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=int(len(robot_positions_train) // batch_size), validation_data=my_validation_batch_generator, callbacks=[monitor], epochs=10)
	score = model.evaluate([images_test, robot_positions_test, robot_velocitys_test] , strawberry_cluster_test)

	##################################### save Model ###########################################################################
	model.save('/content/drive/My Drive/Masters Year/data_mfpc/ResNet_full_001.h5')
	intermediate_layer_model.save('/content/drive/My Drive/Masters Year/data_mfpc/ResNet_intermediate_layer_model_001.h5')
	print("model.summary()")
	model.summary()
	print("intermediate_layer_model.summary()")
	intermediate_layer_model.summary()
	try:
		predict_AlexNet_dense = model.predict([images_test, robot_positions_test, robot_velocitys_test])
		model.summary()
		err_matrix_customNet_dense = strawberry_cluster_test - predict_AlexNet_dense
		customNet_err_mean = np.mean(abs(err_matrix_customNet_dense))
		print("AlexNet mean error values for each output: ")
		print(customNet_err_mean)


		print([i / 100000 for i in predict_AlexNet_dense[250]])
		print([float(i / 100000) for i in strawberry_cluster_test[250]])
		print("------------------------------------------------------------------------------")
		print([i / 100000 for i in predict_AlexNet_dense[251]])
		print([float(i / 100000) for i in strawberry_cluster_test[251]])
	except:
		pass

	########################################################## Define VGG16 CNN ####################################################
	########################################################## Define VGG16 CNN ####################################################
	########################################################## Define VGG16 CNN ####################################################
	########################################################## Define VGG16 CNN ####################################################
	########################################################## Define VGG16 CNN ####################################################
	########################################################## Define VGG16 CNN ####################################################

	model = VGG19(include_top=False, weights='imagenet', input_shape=(224 , 224 , 3))
	for layer in model.layers[:21]:
	    layer.trainable=False
	for layer in model.layers[21:]:
	    layer.trainable=True
	y1 = model.output
	y2 = GlobalAveragePooling2D()(y1)
	y3 = Dense(512,activation='relu')(y2) 
	y4 = Dense(512,activation='relu')(y3) 
	new_model = Model(inputs=model.input,outputs=y4)
	cnn_out = new_model.output

	robot_pose_input_layer = keras.layers.Input(shape=((7*(trajectory_length+1)),))
	dense_3_1 = keras.layers.Dense(64, activation="relu")(robot_pose_input_layer)
	dense_4_1 = keras.layers.Dense(32, activation="relu")(dense_3_1)

	robot_vel_input_layer = keras.layers.Input(shape=((7*(trajectory_length+1)),))
	dense_3_2 = keras.layers.Dense(64, activation="relu")(robot_vel_input_layer)
	dense_4_2 = keras.layers.Dense(32, activation="relu")(dense_3_2)

	concat = keras.layers.concatenate([dense_3_1, dense_3_2, cnn_out])

	dense_5 = keras.layers.Dense(64, activation="relu")(concat)
	dense_6 = keras.layers.Dense(128, activation="relu")(dense_5)
	dense_7 = keras.layers.Dense(64, activation="relu")(dense_6)
	dense_8 = keras.layers.Dense(32, activation="relu")(dense_7)
	output_layer = keras.layers.Dense((21), activation="linear")(dense_8)

	model = keras.models.Model(inputs=[image_input_layer, robot_pose_input_layer, robot_vel_input_layer] , outputs=output_layer)
	intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=dense_5) 

	# Compile the model
	model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mae'])
	model.summary()

	"""**Train the model**"""
	monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='auto', restore_best_weights=True)

	history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=int(len(robot_positions_train) // batch_size), validation_data=my_validation_batch_generator, callbacks=[monitor], epochs=10)
	score = model.evaluate([images_test, robot_positions_test, robot_velocitys_test] , strawberry_cluster_test)

	##################################### save Model ###########################################################################
	model.save('/content/drive/My Drive/Masters Year/data_mfpc/Vgg16_full_001.h5')
	intermediate_layer_model.save('/content/drive/My Drive/Masters Year/data_mfpc/Vgg16_intermediate_layer_model_001.h5')

	print("model.summary()")
	model.summary()
	print("intermediate_layer_model.summary()")
	intermediate_layer_model.summary()
	try:
		predict_AlexNet_dense = model.predict([images_test, robot_positions_test, robot_velocitys_test])
		model.summary()
		err_matrix_customNet_dense = strawberry_cluster_test - predict_AlexNet_dense
		customNet_err_mean = np.mean(abs(err_matrix_customNet_dense))
		print("AlexNet mean error values for each output: ")
		print(customNet_err_mean)


		print([i / 100000 for i in predict_AlexNet_dense[250]])
		print([float(i / 100000) for i in strawberry_cluster_test[250]])
		print("------------------------------------------------------------------------------")
		print([i / 100000 for i in predict_AlexNet_dense[251]])
		print([float(i / 100000) for i in strawberry_cluster_test[251]])
	except:
		pass
