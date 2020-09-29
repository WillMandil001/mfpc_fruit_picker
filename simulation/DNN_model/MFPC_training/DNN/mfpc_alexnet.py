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
batch_size = 32

with tf.device('/gpu:1'):  
    ################################## Generater to keep ram space Parameters ##################################################################
    # read data:
    robot_positions_train = np.array(pd.read_csv(data_location + 'data_set_003/processed_001/robot_positions_train_FOR_CLUSTER_001', header=None))
    robot_positions_test = np.array(pd.read_csv(data_location + 'data_set_003/processed_001/robot_positions_test_FOR_CLUSTER_001', header=None))

    robot_velocitys_train = np.array(pd.read_csv(data_location + 'data_set_003/processed_001/robot_velocitys_train_FOR_CLUSTER_001', header=None))
    robot_velocitys_test = np.array(pd.read_csv(data_location + 'data_set_003/processed_001/robot_velocitys_test_FOR_CLUSTER_001', header=None))

    strawberry_state_train = np.array(pd.read_csv(data_location + 'data_set_003/processed_001/strawberry_cluster_train_FOR_CLUSTER_001', header=None))
    strawberry_state_test = np.array(pd.read_csv(data_location + 'data_set_003/processed_001/strawberry_cluster_test_FOR_CLUSTER_001', header=None))

    image_file_names_train = np.array([(data_location + "data_set_003/processed_001/images_shuffled/image_" + str(i) + ".png") for i in range(0, len(robot_positions_train))])
    image_file_names_test = np.array([(data_location + "data_set_003/processed_001/images_shuffled/image_" + str(i) + ".png") for i in range(len(robot_positions_train), (len(robot_positions_train) + len(robot_positions_test)))])

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

            return [np.array([resize(imread(file_name), (224, 224, 3)) for file_name in batch_x_img]), np.array(batch_x_robot_position)], np.array(batch_x_robot_velocities), np.array(batch_y_strawberry_state)

    my_training_batch_generator = My_Custom_Generator(image_file_names_train, robot_positions_train, robot_velocitys_train, strawberry_state_train, batch_size)
    my_testing_batch_generator = My_Custom_Generator(image_file_names_test, robot_positions_test, robot_velocitys_test, strawberry_state_test, batch_size)


    print(strawberry_state_train.shape)
    print(type(strawberry_state_train[0]))
    print(strawberry_state_train[0].shape)

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

    history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch =int(len(robot_positions_train) // batch_size), callbacks=[monitor], epochs=10)
    # history = model.fit([images_train, robot_positions_train, robot_velocitys_train], strawberry_cluster_train, callbacks=[monitor], batch_size=32, validation_split=0.2, epochs=50)
    score = model.evaluate([images_test, robot_positions_test, robot_velocitys_test] , strawberry_cluster_test)

    ##################################### save Model ###########################################################################
    model.save(data_location + 'AlexNet_full_001.h5')
    intermediate_layer_model.save(data_location + 'AlexNet_intermediate_layer_model_001.h5')

    predict_AlexNet_dense = model.predict([images_test, robot_positions_test, robot_velocitys_test])
    model.summary()
    err_matrix_customNet_dense = strawberry_cluster_test - predict_AlexNet_dense
    customNet_err_mean = np.mean(abs(err_matrix_customNet_dense))
    print("AlexNet mean error values for each output: ")
    print(customNet_err_mean)

    model.summary()

    print([i / 100000 for i in predict_AlexNet_dense[250]])
    print([float(i / 100000) for i in strawberry_cluster_test[250]])
    print("------------------------------------------------------------------------------")
    print([i / 100000 for i in predict_AlexNet_dense[251]])
    print([float(i / 100000) for i in strawberry_cluster_test[251]])

    ########################################################## Define ResNet CNN ####################################################
    ########################################################## Define ResNet CNN ####################################################
    ########################################################## Define ResNet CNN ####################################################
    ########################################################## Define ResNet CNN ####################################################
    ########################################################## Define ResNet CNN ####################################################
    ########################################################## Define ResNet CNN ####################################################
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

    history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch =int(len(robot_positions_train) // batch_size), callbacks=[monitor], epochs=10)
    score = model.evaluate([images_test, robot_positions_test, robot_velocitys_test] , strawberry_cluster_test)

    ##################################### save Model ###########################################################################
    model.save(data_location + 'ResNet_full_001.h5')
    intermediate_layer_model.save(data_location + 'ResNet_intermediate_layer_model_001.h5')

    predict_AlexNet_dense = model.predict([images_test, robot_positions_test, robot_velocitys_test])
    model.summary()
    err_matrix_customNet_dense = strawberry_cluster_test - predict_AlexNet_dense
    customNet_err_mean = np.mean(abs(err_matrix_customNet_dense))
    print("AlexNet mean error values for each output: ")
    print(customNet_err_mean)

    model.summary()

    print([i / 100000 for i in predict_AlexNet_dense[250]])
    print([float(i / 100000) for i in strawberry_cluster_test[250]])
    print("------------------------------------------------------------------------------")
    print([i / 100000 for i in predict_AlexNet_dense[251]])
    print([float(i / 100000) for i in strawberry_cluster_test[251]])

    ########################################################## Define VGG16 CNN ####################################################
    ########################################################## Define VGG16 CNN ####################################################
    ########################################################## Define VGG16 CNN ####################################################
    ########################################################## Define VGG16 CNN ####################################################
    ########################################################## Define VGG16 CNN ####################################################
    ########################################################## Define VGG16 CNN ####################################################

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

    history = model.fit_generator(generator=my_training_batch_generator, steps_per_epoch =int(len(robot_positions_train) // batch_size), callbacks=[monitor], epochs=10)
    score = model.evaluate([images_test, robot_positions_test, robot_velocitys_test] , strawberry_cluster_test)

    ##################################### save Model ###########################################################################
    model.save(data_location + 'Vgg16_full_001.h5')
    intermediate_layer_model.save(data_location + 'Vgg16_intermediate_layer_model_001.h5')

    predict_AlexNet_dense = model.predict([images_test, robot_positions_test, robot_velocitys_test])
    model.summary()
    err_matrix_customNet_dense = strawberry_cluster_test - predict_AlexNet_dense
    customNet_err_mean = np.mean(abs(err_matrix_customNet_dense))
    print("AlexNet mean error values for each output: ")
    print(customNet_err_mean)

    model.summary()

    print([i / 100000 for i in predict_AlexNet_dense[250]])
    print([float(i / 100000) for i in strawberry_cluster_test[250]])
    print("------------------------------------------------------------------------------")
    print([i / 100000 for i in predict_AlexNet_dense[251]])
    print([float(i / 100000) for i in strawberry_cluster_test[251]])

