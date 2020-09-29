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

################################ read in processed data ###########################################################
data_location = "/home/will/Robotics/Data_sets/data_set_003/"
cameras = 8 + 1
data_set_length = 165
trajectory_length = 1  # not including the start state...
batch_size = 32

################################# Robot pos ###################################################################
robot_positions = []
for i in range(0, data_set_length):
  print("here 0 >> ",  i)
  robot_positions_new_sample = pd.read_csv(data_location + '/robot_pos/data_set_' + str(i) + '_robot_data_store_position.csv', header=None)
  for j in range(0, 200, 4):
    robot_positions.append(robot_positions_new_sample.iloc[j:j+4].values.flatten())

################################## Load Strawberry Data ##################################################################
strawberry_states = []
blank = pd.read_csv(data_location + 'straw_2/data_set_' + str(0) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None)
blank_state = blank.iloc[2].values.flatten()
for i in range(0, data_set_length):
  print("here 1 >> ", i)
  strawberry_1 = pd.read_csv(data_location + 'straw_1/data_set_' + str(i) + '_strawberry_data_store_1.csv', delimiter=',', error_bad_lines=False, header=None)
  strawberry_2 = pd.read_csv(data_location + 'straw_2/data_set_' + str(i) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None)
  strawberry_3 = pd.read_csv(data_location + 'straw_3/data_set_' + str(i) + '_strawberry_data_store_3.csv', delimiter=',', error_bad_lines=False, header=None)
  strawberry_4 = pd.read_csv(data_location + 'straw_4/data_set_' + str(i) + '_strawberry_data_store_4.csv', delimiter=',', error_bad_lines=False, header=None)
  strawberry_5 = pd.read_csv(data_location + 'straw_5/data_set_' + str(i) + '_strawberry_data_store_5.csv', delimiter=',', error_bad_lines=False, header=None)
  for j in range(0, 200, 4):
    counter = 0
    strawberry_cluster_state_list = []
    strawberry_cluster_state_list.append(strawberry_1.iloc[j].values.flatten())
    if strawberry_2[0][0] != 100 and counter < 2:
      strawberry_cluster_state_list.append(strawberry_2.iloc[j].values.flatten())
      counter += 1
    if strawberry_3[0][0] != 100 and counter < 2:
      strawberry_cluster_state_list.append(strawberry_3.iloc[j].values.flatten())
      counter += 1
    if strawberry_4[0][0] != 100 and counter < 2:
      strawberry_cluster_state_list.append(strawberry_4.iloc[j].values.flatten())
      counter += 1
    if strawberry_5[0][0] != 100 and counter < 2:
      strawberry_cluster_state_list.append(strawberry_5.iloc[j].values.flatten())
      counter += 1
    for i in range(counter, 2):
      strawberry_cluster_state_list.append(blank_state)
    strawberry_states.append(np.concatenate(strawberry_cluster_state_list).ravel())
################################## Order data for time step prediction ###################################################################
camera_frames = [a for a in range(0, 50)]
robot_pos_trajectory_input = []
strawberry_state_input = []
strawberry_state_label = []
camera_name_list = []
sample = 0
camera = 1
for i in range(0, data_set_length * 50, 50):
  for j in range(0, 50 - trajectory_length - 1):
    print("here 2 >> ", i, " > ", str(i+j))
    try:
      camera_name_list.append(data_location + "camera_" + str(camera) + "/rgb/sample_" + str(sample) + "/time_step_" + str(j) + ".png")
      robot_pos_trajectory_input.append(robot_positions[i+j:i+j+trajectory_length + 1])
      strawberry_state_input.append(strawberry_states[i])
      strawberry_state_label.append(strawberry_states[i+j+trajectory_length + 1])  # does not include the start state of the strawberry. i+1:
    except:
      print("error")
      pass
  sample += 1

print(len(robot_pos_trajectory_input))
print(len(strawberry_state_input))
print(len(strawberry_state_label))
print(len(camera_name_list))

########################################################## lABEL NO ORIENTATION ####################################################
strawberry_state_label__ = []
for pose in strawberry_state_label:
  strawberry_state_label__.append(np.concatenate([pose[0:3], pose[7:10], pose[14:17]]).ravel())
strawberry_state_label = strawberry_state_label__

############################## random shuffle the data ##########################################
random_shuffle_order = random.sample(range(len(robot_pos_trajectory_input)), len(robot_pos_trajectory_input))

robot_pos_trajectory_input_shuffled = []
camera_name_list_shuffled = []
strawberry_state_input_shuffled = []
strawberry_state_label_shuffled = []

for value in random_shuffle_order:
  camera_name_list_shuffled.append(camera_name_list[value]) 
  robot_pos_trajectory_input_shuffled.append(np.concatenate(robot_pos_trajectory_input[value]).ravel()) 
  strawberry_state_input_shuffled.append(strawberry_state_input[value]) 
  strawberry_state_label_shuffled.append(strawberry_state_label[value])

############################## split the data into train, test and val the data ##########################################
camera_name_list_shuffled_train = camera_name_list_shuffled[0:int(len(camera_name_list_shuffled)*0.75)]
camera_name_list_shuffled_test = camera_name_list_shuffled[int(len(camera_name_list_shuffled)*0.75):len(camera_name_list_shuffled)]

robot_pos_trajectory_input_shuffled_train = tf.convert_to_tensor(robot_pos_trajectory_input_shuffled[0:int(len(robot_pos_trajectory_input_shuffled)*0.75)])
robot_pos_trajectory_input_shuffled_test = tf.convert_to_tensor(robot_pos_trajectory_input_shuffled[int(len(robot_pos_trajectory_input_shuffled)*0.75):len(robot_pos_trajectory_input_shuffled)])

strawberry_state_input_shuffled_train = tf.convert_to_tensor(strawberry_state_input_shuffled[0:int(len(strawberry_state_input_shuffled)*0.75)])
strawberry_state_input_shuffled_test = tf.convert_to_tensor(strawberry_state_input_shuffled[int(len(strawberry_state_input_shuffled)*0.75):len(strawberry_state_input_shuffled)])

strawberry_state_label_shuffled_train = tf.convert_to_tensor(strawberry_state_label_shuffled[0:int(len(strawberry_state_label_shuffled)*0.75)])
strawberry_state_label_shuffled_test = tf.convert_to_tensor(strawberry_state_label_shuffled[int(len(strawberry_state_label_shuffled)*0.75):len(strawberry_state_label_shuffled)])

batch_size = 32
class My_Custom_Generator(keras.utils.Sequence) :
  def __init__(self, file_names, robot_position, strawberry_state_label, batch_size) :
    self.file_names = file_names
    self.strawberry_state_label = strawberry_state_label
    self.robot_position = robot_position
    self.batch_size = batch_size

  def __len__(self) :
    return (np.ceil(len(self.file_names) / float(self.batch_size))).astype(np.int)

  def __getitem__(self, idx) :
    batch_x_img = self.file_names[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_x_robot_position = self.robot_position[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y_strawberry_state_label = self.strawberry_state_label[idx * self.batch_size : (idx+1) * self.batch_size]

    return [np.array([resize(imread(file_name), (224, 224, 3)) for file_name in batch_x_img]), np.array(batch_x_robot_position)], np.array(batch_y_strawberry_state_label)

my_training_batch_generator = My_Custom_Generator(camera_name_list_shuffled_train, robot_pos_trajectory_input_shuffled_train, strawberry_state_label_shuffled_train, batch_size)
my_test_batch_generator = My_Custom_Generator(robot_pos_trajectory_input_shuffled_test, robot_pos_trajectory_input_shuffled_test, strawberry_state_label_shuffled_test, batch_size)

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

robot_pose_input_layer = keras.layers.Input(shape=((56),))
dense_3_1 = keras.layers.Dense(64, activation="relu")(robot_pose_input_layer)
dense_4_1 = keras.layers.Dense(32, activation="relu")(dense_3_1)

concat = keras.layers.concatenate([dense_4_1, cnn_flatten])

dense_5 = keras.layers.Dense(128, activation="relu")(concat)
dense_6 = keras.layers.Dense(128, activation="relu")(dense_5)
dense_7 = keras.layers.Dense(64, activation="relu")(dense_6)
output_layer = keras.layers.Dense(9, activation="linear")(dense_7)

# inputs=[wide_model.input] + [deep_model.input]
model = keras.models.Model(inputs=[image_input_layer, robot_pose_input_layer] , outputs=output_layer)

# Compile the model
model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mae'])
model.summary()

"""**Train the model**"""
monitor = EarlyStopping(monitor='mae', min_delta=1e-3, patience=2, verbose=1, mode='auto', restore_best_weights=True)

history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=int(len(robot_pos_trajectory_input_shuffled_train) // batch_size), callbacks=[monitor], epochs=10)
# ##################################### save Model ###########################################################################
# model.save(data_location + 'FINAL_TEST_AlexNet_camera_1_full_001.h5')

# ########################################################## Define VGG19 CNN ####################################################
# ########################################################## Define VGG19 CNN ####################################################
# ########################################################## Define VGG19 CNN ####################################################
# ########################################################## Define VGG19 CNN ####################################################
# ########################################################## Define VGG19 CNN ####################################################
# ########################################################## Define VGG19 CNN ####################################################
# model = VGG19(include_top=False, weights='imagenet', input_shape=(224 , 224 , 3))
# #model.summary()

# for layer in model.layers[:21]:
#     layer.trainable=False
# for layer in model.layers[21:]:
#     layer.trainable=True

# y1 = model.output
# y2 = GlobalAveragePooling2D()(y1)
# y3 = Dense(512,activation='relu')(y2) 
# y4 = Dense(512,activation='relu')(y3) 

# new_model = Model(inputs=model.input,outputs=y4)
# ####################################################################################################################
# cnn_out = new_model.output

# robot_pose_input_layer = keras.layers.Input(shape=((28),))
# dense_3_1 = keras.layers.Dense(64, activation="relu")(robot_pose_input_layer)
# dense_4_1 = keras.layers.Dense(32, activation="relu")(dense_3_1)

# concat = keras.layers.concatenate([dense_4_1, cnn_out])

# dense_5 = keras.layers.Dense(128, activation="relu")(concat)
# dense_6 = keras.layers.Dense(128, activation="relu")(dense_5)
# dense_7 = keras.layers.Dense(64, activation="relu")(dense_6)
# output_layer = keras.layers.Dense(9, activation="linear")(dense_7)

# # inputs=[wide_model.input] + [deep_model.input]
# model = keras.models.Model(inputs=[model.input, robot_pose_input_layer] , outputs=output_layer)

# # Compile the model
# model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mae'])
# model.summary()

# """**Train the model**"""
# monitor = EarlyStopping(monitor='mae', min_delta=1e-3, patience=2, verbose=1, mode='auto', restore_best_weights=True)

# history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=int(len(robot_pos_trajectory_input_shuffled_train) // batch_size), callbacks=[monitor], epochs=10)
# ##################################### save Model ###########################################################################
# model.save(data_location + 'FINAL_TEST_VGG16_camera_1_full_001.h5')

# ########################################################## Define ResNet CNN ####################################################
# ########################################################## Define ResNet CNN ####################################################
# ########################################################## Define ResNet CNN ####################################################
# ########################################################## Define ResNet CNN ####################################################
# ########################################################## Define ResNet CNN ####################################################
# ########################################################## Define ResNet CNN ####################################################
# model = ResNet152V2(include_top=False, weights='imagenet', input_shape=(224 , 224 , 3))

# y1 = model.output
# y2 = GlobalAveragePooling2D()(y1)
# y3 = Dense(1024,activation='relu')(y2) 
# y4 = Dense(1024,activation='relu')(y3)
# new_model = Model(inputs=model.input,outputs=y4)

# for layer in new_model.layers[:561]:
#   layer.trainable=False
# for layer in new_model.layers[561:]:
#   layer.trainable=True
# cnn_out = new_model.output

# robot_pose_input_layer = keras.layers.Input(shape=((28),))
# dense_3_1 = keras.layers.Dense(64, activation="relu")(robot_pose_input_layer)
# dense_4_1 = keras.layers.Dense(32, activation="relu")(dense_3_1)

# concat = keras.layers.concatenate([dense_4_1, cnn_out])

# dense_5 = keras.layers.Dense(128, activation="relu")(concat)
# dense_6 = keras.layers.Dense(128, activation="relu")(dense_5)
# dense_7 = keras.layers.Dense(64, activation="relu")(dense_6)
# output_layer = keras.layers.Dense(9, activation="linear")(dense_7)

# # inputs=[wide_model.input] + [deep_model.input]
# model = keras.models.Model(inputs=[new_model.input, robot_pose_input_layer] , outputs=output_layer)

# # Compile the model
# model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mae'])
# model.summary()

# """**Train the model**"""
# monitor = EarlyStopping(monitor='mae', min_delta=1e-3, patience=2, verbose=1, mode='auto', restore_best_weights=True)

# history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=int(len(robot_pos_trajectory_input_shuffled_train) // batch_size), callbacks=[monitor], epochs=10)
# ##################################### save Model ###########################################################################
# model.save(data_location + 'FINAL_TEST_ResNet_camera_1_full_001.h5')
