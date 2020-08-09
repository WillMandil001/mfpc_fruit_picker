import numpy as np
import matplotlib
import cv2
import pandas as pd
from PIL import Image
import matplotlib.image as mpimg
from sklearn import preprocessing
from matplotlib import pyplot as plt


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.callbacks import EarlyStopping
# #from tensorflow.keras.layers.normalization import BatchNormalization
# np.random.seed(1000)

# from tensorflow.keras.applications.resnet import preprocess_input
# from tensorflow.keras.preprocessing import image

# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

no_of_cameras = 8 + 1
data_set_length = 5
train = 3
test = 2

################################# Robot vel and pos ###################################################################
for camera in range(1, no_of_cameras):
    robot_positions = pd.read_csv('/vol/hd/mfpc_data/data_set_003/robot_pos/data_set_' + str(0) + '_robot_data_store_position.csv', header=None)
    robot_velocitys = pd.read_csv('/vol/hd/mfpc_data/data_set_003/robot_vel/data_set_' + str(0) + '_robot_data_store_velocity.csv', header=None)
    for i in range(1, data_set_length):
        robot_positions = pd.concat([robot_positions, pd.read_csv('/vol/hd/mfpc_data/data_set_003/robot_pos/data_set_' + str(i) + '_robot_data_store_position.csv', header=None)])
        robot_velocitys = pd.concat([robot_velocitys, pd.read_csv('/vol/hd/mfpc_data/data_set_003/robot_vel/data_set_' + str(i) + '_robot_data_store_velocity.csv', header=None)])
    if camera == 1:
        robot_state = pd.concat([robot_positions, robot_velocitys], axis=1)
    else:
        robot_state_single_cam = pd.concat([robot_positions, robot_velocitys], axis=1)
        robot_state = pd.concat([robot_state, robot_state_single_cam], axis=0)

robot_states_list = robot_state.values.tolist()
list_ = []
for i in range(0, len(robot_state[0]), 4):
    list_.append(robot_states_list[i])
robot_states_frame_rate = pd.DataFrame(list_)

################################## Standardization for Robot States ###################################################################
robot_state_names = robot_states_frame_rate.columns
scaler = preprocessing.StandardScaler()
myScaler = scaler.fit(robot_states_frame_rate)
robot_states_frame_rate = myScaler.transform(robot_states_frame_rate)
robot_states_frame_rate = pd.DataFrame(robot_states_frame_rate, columns=robot_state_names)
print(robot_states_frame_rate.shape)

################################## Load Strawberry Data ##################################################################
strawberry_1 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_1/data_set_' + str(0) + '_strawberry_data_store_1.csv', delimiter=',', error_bad_lines=False,  header=None)
strawberry_2 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_2/data_set_' + str(0) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_3 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_3/data_set_' + str(0) + '_strawberry_data_store_3.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_4 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_4/data_set_' + str(0) + '_strawberry_data_store_4.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_5 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_5/data_set_' + str(0) + '_strawberry_data_store_5.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_states = strawberry_1
blank = strawberry_2
counter = 0
for i in range(counter, 2):
    strawberry_states = pd.concat([strawberry_states, blank], axis=1)

for i in range(1, data_set_length):
    strawberry_cluster_state = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_1/data_set_' + str(i) + '_strawberry_data_store_1.csv', delimiter=',', error_bad_lines=False, header=None)
    strawberry_2 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_2/data_set_' + str(i) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None)
    strawberry_3 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_3/data_set_' + str(i) + '_strawberry_data_store_3.csv', delimiter=',', error_bad_lines=False, header=None)
    strawberry_4 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_4/data_set_' + str(i) + '_strawberry_data_store_4.csv', delimiter=',', error_bad_lines=False, header=None)
    strawberry_5 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_5/data_set_' + str(i) + '_strawberry_data_store_5.csv', delimiter=',', error_bad_lines=False, header=None)
    counter = 0
    if strawberry_2[0][0] != 100 and counter < 2:
        strawberry_cluster_state = pd.concat([strawberry_cluster_state, strawberry_2], axis=1)
        counter += 1
    if strawberry_3[0][0] != 100 and counter < 2:
        strawberry_cluster_state = pd.concat([strawberry_cluster_state, strawberry_3], axis=1)
        counter += 1
    if strawberry_4[0][0] != 100 and counter < 2:
        strawberry_cluster_state = pd.concat([strawberry_cluster_state, strawberry_4], axis=1)
        counter += 1
    if strawberry_5[0][0] != 100 and counter < 2:
        strawberry_cluster_state = pd.concat([strawberry_cluster_state, strawberry_5], axis=1)
        counter += 1
    for i in range(counter, 2):
        strawberry_cluster_state = pd.concat([strawberry_cluster_state, blank], axis=1)
    strawberry_states = pd.concat([strawberry_states, strawberry_cluster_state], axis=0)

strawberry_states_list = strawberry_states.values.tolist()
list_ = []
for i in range(0, len(strawberry_states[0]), 4):
    list_.append(strawberry_states_list[i])
strawberry_states_frame_rate = pd.DataFrame(list_)

strawberry_states_frame_rate__ = strawberry_states_frame_rate
for camera in range(1, no_of_cameras-1):
    strawberry_states_frame_rate = pd.concat([strawberry_states_frame_rate, strawberry_states_frame_rate__], axis=0)

################################## Standardization for Strawberry States ###################################################################
strawberry_state_names = strawberry_states_frame_rate.columns
scaler = preprocessing.StandardScaler()
myScaler = scaler.fit(strawberry_states_frame_rate)
strawberry_states_frame_rate = myScaler.transform(strawberry_states_frame_rate)
strawberry_states_frame_rate = pd.DataFrame(strawberry_states_frame_rate, columns=strawberry_state_names)
print(strawberry_states_frame_rate.shape)

############################################### Load image data #####################################################
images = []
for camera in range(1, no_of_cameras):
    print("camera: ", camera)
    for i in range(0, data_set_length):
        for j in range(0, 50):
            im_frame = cv2.imread('/vol/hd/mfpc_data/data_set_003/camera_' + str(camera) + '/rgb/sample_' + str(i) + '/time_step_' + str(j) + '.png')
            images.append(im_frame)

train_images = np.asarray(images[0:1500])
train_images = train_images.astype(np.float64)
print("train_imagesing dataset size : {}".format(train_images.shape[0]))


test_images = np.asarray(images[1500:2000])
test_images = test_images.astype(np.float64)
print("test_imagesing dataset size : {}".format(test_images.shape[0]))

############################################### Normalize image data ###############################################
for i in range(0 , train_images.shape[0]):
    train_images[i, : , : , :] = train_images[i, : , : , :] / np.max(train_images[i, : , : , :])

robot_state_train_input = robot_states_frame_rate[0:train_images.shape[0]]
print("Robot state input trainingset size: {}".format(robot_state_train_input.shape))
robot_state_train_label = strawberry_states_frame_rate[0:train_images.shape[0]]
print("Robot state label trainingset size: {}".format(robot_state_train_label.shape))

for i in range(0 , test_images.shape[0]):
    test_images[i, : , : , :] = test_images[i, : , : , :] / np.max(test_images[i, : , : , :])

robot_state_test_input = robot_states_frame_rate[train_images.shape[0]:train_images.shape[0]+test_images.shape[0]]
print("Robot state input testset size: {}".format(robot_state_test_input.shape))
robot_state_test_label = strawberry_states_frame_rate[train_images.shape[0]:train_images.shape[0]+test_images.shape[0]]
print("Robot state label testset size: {}".format(robot_state_test_label.shape))

########################################################## Define AlexNet CNN ####################################################
image_input_layer = keras.layers.Input(shape=(350,350,3))

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


robot_state_input_layer = keras.layers.Input(shape=(14,))

dense_3 = keras.layers.Dense(15, activation="relu")(robot_state_input_layer)
dense_4 = keras.layers.Dense(25, activation="relu")(dense_3)

concat = keras.layers.concatenate([dense_4 , drop_2])

dense_5 = keras.layers.Dense(80, activation="relu")(concat)
dense_6 = keras.layers.Dense(20, activation="relu")(dense_5)
output_layer = keras.layers.Dense(21, activation="linear")(dense_6)

model = keras.models.Model(inputs=[image_input_layer , robot_state_input_layer] , outputs=output_layer)

# Compile the model
model.compile(optimizer='adam',loss='mean_absolute_error', metrics=['mse','accuracy'])

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, 
        verbose=1, mode='auto', restore_best_weights=True)

history = model.fit([train_images, robot_state_train_input], robot_state_train_label, callbacks=[monitor], batch_size=32, validation_split=0.2, epochs=6)
score = model.evaluate([test_images, robot_state_test_input] , robot_state_test_label) 


predict_AlexNet_dense = model.predict([test_images, robot_state_test_input])


err_matrix_AlexNet_dense = robot_state_test_label - predict_AlexNet_dense
AlexNet_err_mean = np.mean(abs(err_matrix_AlexNet_dense))
print("AlexNet mean error values for each output: ")
print(AlexNet_err_mean)
a = np.where(err_matrix_AlexNet_dense > 0.01)
a = np.asarray(list(zip(*a)))
print("number of err elements higher than 0.01: {}".format(a.shape))


predict_AlexNet_dense.shape

##################################### save Model ###########################################################################
model.save('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Models/AlexNet_Ordered.h5')

model.summary()