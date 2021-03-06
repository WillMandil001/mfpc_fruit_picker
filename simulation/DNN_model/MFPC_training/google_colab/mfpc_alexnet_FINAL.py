import cv2
import random
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(1000)
from sklearn import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input


with tf.device('/gpu:1'):  
	################################## Hyper Parameters ##################################################################
	data_location = "sftp://wmandil@lch01/home/wmandil/MFPC/datasets"
	no_of_cameras = 8 + 1
	data_set_length = 3
	trajectory_length = 1
	scale_up_value = 100000
	single_sample_length = 199
	train_size__ = 0.8scale_up_value

	################################# Robot vel and pos ###################################################################
	robot_positions = []
	robot_velocitys = []
	for i in range(1, data_set_length+1):
		robot_positions__ = pd.read_csv(data_location + 'data_set_003/robot_pos/data_set_' + str(i) + '_robot_data_store_position.csv').values.flatten()
		robot_velocitys__ = pd.read_csv(data_location + 'data_set_003/robot_vel/data_set_' + str(i) + '_robot_data_store_velocity.csv').values.flatten()
		robot_positions_order = []
		robot_velocitys_order = []
		for j in range(0, len(robot_positions__), 7):
			robot_positions_order.append(robot_positions__[j:j+7])
			robot_velocitys_order.append(robot_velocitys__[j:j+7])
		for k in range(0, len(robot_positions_order), 4):
			robot_positions.append(robot_positions_order[k])
			robot_velocitys.append(robot_velocitys_order[k])
	print(len(robot_positions))

	############################################### CUT data with no movement (PART 1) ###############################################
	to_keep_index = []
	for i in range(0, len(robot_velocitys) - 1):
		dif = sum(abs(robot_velocitys[i] - robot_velocitys[i+1]))
		# print("here 3 >> ", i, " / ", len(robot_vel_frame_rate), " >> ", dif)
		if dif > 0.025:
			to_keep_index.append(i)
	print(len(to_keep_index))
	to_keep_index_all_cams = []
	for camera in range(1, no_of_cameras):
		to_keep_index_all_cams.append(to_keep_index)

	to_keep_index = [y for x in to_keep_index_all_cams for y in x]
	print(len(to_keep_index))

	################################## Standardization for Robot States ###################################################################
	for i in range(0, len(robot_positions)):
		robot_positions[i] = scale_up_value*robot_positions[i]
	print(robot_positions[0])
	scaler = preprocessing.StandardScaler()
	myScaler = scaler.fit(robot_positions)
	robot_positions = myScaler.transform(robot_positions)
	print(robot_positions.shape)

	for i in range(0, len(robot_velocitys)):
		robot_velocitys[i] = scale_up_value*robot_velocitys[i]
	scaler = preprocessing.StandardScaler()
	myScaler = scaler.fit(robot_velocitys)
	robot_velocitys = myScaler.transform(robot_velocitys)
	print(robot_velocitys.shape)

	############################################### Convert to trajectory && for each camera ###############################################
	robot_positions_trajectory = []
	robot_velocitys_trajectory = []
	for camera in range(1, no_of_cameras):
		for i in range(0, len(robot_positions)):
			robot_positions_trajectory.append(np.concatenate(robot_positions[i:i+2]).ravel())
			robot_velocitys_trajectory.append(np.concatenate(robot_velocitys[i:i+2]).ravel())

	print(robot_positions_trajectory[0])
	print(robot_positions_trajectory[1])

	################################## Load Strawberry Data ##################################################################
	straw_1 = []
	straw_2 = []
	straw_3 = []
	straw_4 = []
	straw_5 = []
	for camera in range(1, no_of_cameras):
		for i in range(1, data_set_length+1):
			straw_1__ = pd.read_csv(data_location + 'data_set_003/straw_1/data_set_' + str(i) + '_strawberry_data_store_1.csv', delimiter=',', error_bad_lines=False, header=None).values.flatten()
			straw_2__ = pd.read_csv(data_location + 'data_set_003/straw_2/data_set_' + str(i) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None).values.flatten()
			straw_3__ = pd.read_csv(data_location + 'data_set_003/straw_3/data_set_' + str(i) + '_strawberry_data_store_3.csv', delimiter=',', error_bad_lines=False, header=None).values.flatten()
			straw_4__ = pd.read_csv(data_location + 'data_set_003/straw_4/data_set_' + str(i) + '_strawberry_data_store_4.csv', delimiter=',', error_bad_lines=False, header=None).values.flatten()
			straw_5__ = pd.read_csv(data_location + 'data_set_003/straw_5/data_set_' + str(i) + '_strawberry_data_store_5.csv', delimiter=',', error_bad_lines=False, header=None).values.flatten()
			straw_1_order = []
			straw_2_order = []
			straw_3_order = []
			straw_4_order = []
			straw_5_order = []
			for j in range(0, len(straw_1__), 7):
				straw_1_order.append(straw_1__[j:j+7])
				straw_2_order.append(straw_2__[j:j+7])
				straw_3_order.append(straw_3__[j:j+7])
				straw_4_order.append(straw_4__[j:j+7])
				straw_5_order.append(straw_5__[j:j+7])
			for k in range(0, len(robot_positions_order), 4):
				straw_1.append(straw_1_order[k])
				straw_2.append(straw_2_order[k])
				straw_3.append(straw_3_order[k])
				straw_4.append(straw_4_order[k])
				straw_5.append(straw_5_order[k])

	strawberry_cluster_state = []
	blank = straw_2[0]
	for i in range(0, len(straw_1)):  # create cluster and remove empty strawbs
		strawberry_cluster = []
		strawberry_cluster.append([straw_1[i], blank, blank])
		if straw_2[0][0] != 100:
			if strawberry_cluster[1] == blank:
				strawberry_cluster.append(straw_2[i])
			elif strawberry_cluster[2] == blank:
				strawberry_cluster.append(straw_2[i])
		if straw_3[0][0] != 100:
			if strawberry_cluster[1] == blank:
				strawberry_cluster.append(straw_3[i])
			elif strawberry_cluster[2] == blank:
				strawberry_cluster.append(straw_3[i])
		if straw_4[0][0] != 100:
			if strawberry_cluster[1] == blank:
				strawberry_cluster.append(straw_4[i])
			elif strawberry_cluster[2] == blank:
				strawberry_cluster.append(straw_4[i])
		if straw_5[0][0] != 100:
			if strawberry_cluster[1] == blank:
				strawberry_cluster.append(straw_5[i])
			elif strawberry_cluster[2] == blank:
				strawberry_cluster.append(straw_5[i])
		strawberry_cluster_state.append(np.concatenate(strawberry_cluster).ravel())
	print(len(strawberry_cluster_state))

	############################################### Load image data #####################################################
	images = []
	for camera in range(1, no_of_cameras):
		print("camera: ", camera)
		for i in range(0, data_set_length):
			for j in range(0, 50):
				im_frame = cv2.imread(data_location + 'data_set_003/camera_' + str(camera) + '/rgb/sample_' + str(i) + '/time_step_' + str(j) + '.png')
				images.append(im_frame)

	############################################### Cut data with no movement (PART 2) ###############################################
	images_cut = []
	strawberry_cluster_state_cut = []
	robot_positions_cut = []
	robot_velocitys_cut = []

	for value in to_keep_index:
		images_cut.append(images[value])
		strawberry_cluster_state_cut.append(strawberry_cluster_state[value])
		robot_positions_cut.append(robot_positions_trajectory[value])
		robot_velocitys_cut.append(robot_velocitys_trajectory[value])

	############################################### Shuffle Data ###############################################
	random_shuffle_order = random.sample(range(len(images_cut)), len(images_cut))

	images_cut_shuffled = []
	strawberry_cluster_state_cut_shuffled = []
	robot_positions_cut_shuffled = []
	robot_velocitys_cut_shuffled = []

	for value in random_shuffle_order:
		images_cut_shuffled.append(images_cut[value])
		strawberry_cluster_state_cut_shuffled.append(strawberry_cluster_state_cut[value])
		robot_positions_cut_shuffled.append(robot_positions_cut[value])
		robot_velocitys_cut_shuffled.append(robot_velocitys_cut[value])

	############################################### Normalize image data ###############################################
	for i in range(0 , images_cut_shuffled.shape[0]):
		images_cut_shuffled[i, : , : , :] = images_cut_shuffled[i, : , : , :] / np.max(images_cut_shuffled[i, : , : , :])

	############################################### Cut into test and train data ###############################################
	data_set_lenth = len(images_cut_shuffled)
	images_train = images_cut_shuffled[0:int(data_set_lenth*train_size__)]
	images_test = images_cut_shuffled[int(data_set_lenth*train_size__):data_set_lenth]

	strawberry_cluster_train = strawberry_cluster_state_cut_shuffled[0:int(data_set_lenth*train_size__)]
	strawberry_cluster_test = strawberry_cluster_state_cut_shuffled[int(data_set_lenth*train_size__):data_set_lenth]

	robot_positions_train = robot_positions_cut_shuffled[0:int(data_set_lenth*train_size__)]
	robot_positions_test = robot_positions_cut_shuffled[int(data_set_lenth*train_size__):data_set_lenth]

	robot_velocitys_train = robot_velocitys_cut_shuffled[0:int(data_set_lenth*train_size__)]
	robot_velocitys_test = robot_velocitys_cut_shuffled[int(data_set_lenth*train_size__):data_set_lenth]

	print(len(robot_velocitys_train))
	print(len(robot_velocitys_test))
	print(len(images_test))

	# """**Import, sort and normalise the data sets.**"""

	# # ################################## Standardization for Strawberry States ###################################################################
	# # strawberry_cluster_state = np.array(strawberry_cluster_state) * scale_up_value
	# # scaler = preprocessing.StandardScaler()
	# # myScaler = scaler.fit(strawberry_cluster_state)
	# # strawberry_cluster_state = myScaler.transform(strawberry_cluster_state)
	# # print(strawberry_cluster_state.shape)
	# # print(strawberry_cluster_state[0])
	# # print(len(strawberry_cluster_state[0]))

	# """**Now to create and compile the AlexNet CNN:**"""

	# ########################################################## Define AlexNet CNN ####################################################
	# # swapped relu to tanh to stop getting only zero's as output
	# image_input_layer = keras.layers.Input(shape=(350,350,3))

	# layer_conv_1 = keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation="relu")(image_input_layer)
	# layer_pooling_1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(layer_conv_1)

	# layer_conv_2 = keras.layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid', activation="relu")(layer_pooling_1)
	# layer_pooling_2 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(layer_conv_2)

	# layer_conv_3 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation="relu")(layer_pooling_2)
	# layer_conv_4 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation="relu")(layer_conv_3)
	# layer_conv_5 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation="relu")(layer_conv_4)

	# layer_pooling_3 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(layer_conv_5)

	# cnn_flatten = keras.layers.Flatten()(layer_pooling_3)

	# dense_1 = keras.layers.Dense(4096, activation="relu")(cnn_flatten)
	# drop_1 = keras.layers.Dropout(0.4)(dense_1)
	# dense_2 = keras.layers.Dense(4096, activation="relu")(drop_1)
	# drop_2 = keras.layers.Dropout(0.4)(dense_2)


	# robot_pose_input_layer = keras.layers.Input(shape=(7,))
	# dense_3_1 = keras.layers.Dense(15, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(robot_pose_input_layer)
	# dense_4_1 = keras.layers.Dense(25, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(dense_3_1)

	# robot_vel_input_layer = keras.layers.Input(shape=(7,))
	# dense_3_2 = keras.layers.Dense(15, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(robot_vel_input_layer)
	# dense_4_2 = keras.layers.Dense(25, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(dense_3_2)

	# concat = keras.layers.concatenate([dense_4_1, dense_4_2, drop_2])

	# dense_5 = keras.layers.Dense(80, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(concat)
	# dense_6 = keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(dense_5)
	# output_layer = keras.layers.Dense(21, activation="linear")(dense_6)

	# model = keras.models.Model(inputs=[image_input_layer, robot_pose_input_layer, robot_vel_input_layer] , outputs=output_layer)

	# # Compile the model
	# model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mse'])
	# model.summary()

	# """**Train the model**"""

	# monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1, mode='auto', restore_best_weights=True)

	# history = model.fit([train_images, robot_pose_train_input, robot_vel_train_input], robot_state_train_label, callbacks=[monitor], batch_size=32, validation_split=0.2, epochs=15)
	# score = model.evaluate([test_images, robot_pose_test_input, robot_vel_test_input] , robot_state_test_label)

	# predict_AlexNet_dense = model.predict([test_images, robot_pose_test_input, robot_vel_test_input])

	# err_matrix_AlexNet_dense = robot_state_test_label - predict_AlexNet_dense
	# AlexNet_err_mean = np.mean(abs(err_matrix_AlexNet_dense))
	# print("AlexNet mean error values for each output: ")
	# print(AlexNet_err_mean)
	# a = np.where(err_matrix_AlexNet_dense > 0.01)
	# a = np.asarray(list(zip(*a)))
	# print("number of err elements higher than 0.01: {}".format(a.shape))

	# predict_AlexNet_dense.shape

	# ##################################### save Model ###########################################################################
	# model.save(data_location + 'drive/My Drive/Masters Year/data_mfpc/AlexNet_Ordered_leakyrelu_03.h5')

	# model.summary()

	# print(predict_AlexNet_dense[0])
	# print(robot_state_test_label.iloc[0])