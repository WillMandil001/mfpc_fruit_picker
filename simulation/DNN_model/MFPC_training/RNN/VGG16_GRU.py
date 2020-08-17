import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(1000)
from sklearn import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input


validation_set_size = 1500
timeWindow = 4





for camera in range(1, no_of_cameras):
    robot_positions = pd.read_csv('/content/data_set_003/robot_pos/data_set_' + str(0) + '_robot_data_store_position.csv', header=None)
    robot_velocitys = pd.read_csv('/content/data_set_003/robot_vel/data_set_' + str(0) + '_robot_data_store_velocity.csv', header=None)
    for i in range(1, data_set_length):
        robot_positions = pd.concat([robot_positions, pd.read_csv('/content/data_set_003/robot_pos/data_set_' + str(i) + '_robot_data_store_position.csv', header=None)])
        robot_velocitys = pd.concat([robot_velocitys, pd.read_csv('/content/data_set_003/robot_vel/data_set_' + str(i) + '_robot_data_store_velocity.csv', header=None)])
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
print("Done")

################################## Standardization for Robot States ###################################################################
robot_state_names = robot_states_frame_rate.columns
scaler = preprocessing.StandardScaler()
myScaler = scaler.fit(robot_states_frame_rate)
robot_states_frame_rate = myScaler.transform(robot_states_frame_rate)
robot_states_frame_rate = pd.DataFrame(robot_states_frame_rate, columns=robot_state_names)
print(robot_states_frame_rate.shape)

################################## Load Strawberry Data ##################################################################
strawberry_1 = pd.read_csv('/content/data_set_003/straw_1/data_set_' + str(0) + '_strawberry_data_store_1.csv', delimiter=',', error_bad_lines=False,  header=None)
strawberry_2 = pd.read_csv('/content/data_set_003/straw_2/data_set_' + str(0) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_3 = pd.read_csv('/content/data_set_003/straw_3/data_set_' + str(0) + '_strawberry_data_store_3.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_4 = pd.read_csv('/content/data_set_003/straw_4/data_set_' + str(0) + '_strawberry_data_store_4.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_5 = pd.read_csv('/content/data_set_003/straw_5/data_set_' + str(0) + '_strawberry_data_store_5.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_states = strawberry_1
blank = strawberry_2
counter = 0
for i in range(counter, 2):
    strawberry_states = pd.concat([strawberry_states, blank], axis=1)

for i in range(1, data_set_length):
    strawberry_cluster_state = pd.read_csv('/content/data_set_003/straw_1/data_set_' + str(i) + '_strawberry_data_store_1.csv', delimiter=',', error_bad_lines=False, header=None)
    strawberry_2 = pd.read_csv('/content/data_set_003/straw_2/data_set_' + str(i) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None)
    strawberry_3 = pd.read_csv('/content/data_set_003/straw_3/data_set_' + str(i) + '_strawberry_data_store_3.csv', delimiter=',', error_bad_lines=False, header=None)
    strawberry_4 = pd.read_csv('/content/data_set_003/straw_4/data_set_' + str(i) + '_strawberry_data_store_4.csv', delimiter=',', error_bad_lines=False, header=None)
    strawberry_5 = pd.read_csv('/content/data_set_003/straw_5/data_set_' + str(i) + '_strawberry_data_store_5.csv', delimiter=',', error_bad_lines=False, header=None)
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

print("Done")

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
            im_frame = cv2.imread('/content/data_set_003/camera_' + str(camera) + '/rgb/sample_' + str(i) + '/time_step_' + str(j) + '.png')
            im_frame = imutils.resize(im_frame, width=224)
            # im_frame = tf.convert_to_tensor(im_frame)
            images.append(im_frame)
train_size = int((len(images) / 4) * 3)
test_size = len(images) - train_size

train_images = np.asarray(images[0:train_size])
train_images = train_images.astype(np.float64)
print("train_imagesing dataset size : {}".format(train_images.shape[0]))

test_images = np.asarray(images[train_size:train_size+test_size])
test_images = test_images.astype(np.float64)
print("test_imagesing dataset size : {}".format(test_images.shape[0]))

############################################### Normalize image data ###############################################
for i in range(0 , train_images.shape[0]):
    train_images[i, : , : , :] = train_images[i, : , : , :] / np.max(train_images[i, : , : , :])

robot_state_train_input = robot_states_frame_rate[0:train_images.shape[0]]
robot_pose_train_input = robot_state_train_input.iloc[:, 0:7]
robot_vel_train_input = robot_state_train_input.iloc[:, 7:14]
print("Robot pose input trainingset size: {}".format(robot_pose_train_input.shape))
print("Robot vel input trainingset size: {}".format(robot_vel_train_input.shape))
robot_state_train_label = strawberry_states_frame_rate[0:train_images.shape[0]]
print("Robot state label trainset size: {}".format(robot_state_train_label.shape))

for i in range(0 , test_images.shape[0]):
    test_images[i, : , : , :] = test_images[i, : , : , :] / np.max(test_images[i, : , : , :])

robot_state_test_input = robot_states_frame_rate[train_images.shape[0]:train_images.shape[0]+test_images.shape[0]]
robot_pose_test_input = robot_state_test_input.iloc[:, 0:7]
robot_vel_test_input = robot_state_test_input.iloc[:, 7:14]
print("Robot pose input trainingset size: {}".format(robot_pose_test_input.shape))
print("Robot vel input trainingset size: {}".format(robot_vel_test_input.shape))
robot_state_test_label = strawberry_states_frame_rate[train_images.shape[0]:train_images.shape[0]+test_images.shape[0]]
print("Robot state label testset size: {}".format(robot_state_test_label.shape))






####################################################################################################################
model = VGG19(include_top=False, weights='imagenet', input_shape=(224 , 224 , 3))
#model.summary()

for layer in model.layers[:21]:
    layer.trainable=False
for layer in model.layers[21:]:
    layer.trainable=True

y1 = model.output
y2 = GlobalAveragePooling2D()(y1)
y3 = Dense(512,activation='relu')(y2) 
y4 = Dense(512,activation='relu')(y3) 

new_model = Model(inputs=model.input,outputs=y4)
####################################################################################################################

intermediate_layer_model = load_model('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Models/CNN_intermediate_layer.h5')
intermediate_output_train = intermediate_layer_model.predict([train, robot_state_train_input])
intermediate_output_test = intermediate_layer_model.predict([test, robot_state_test_input])

intermediate_output_train.shape

train_set = intermediate_output_train[0:intermediate_output_train.shape[0]-validation_set_size , : ]
validation_set = intermediate_output_train[intermediate_output_train.shape[0]-validation_set_size: , : ]
test_set = intermediate_output_test[: , :]

train = keras.preprocessing.sequence.TimeseriesGenerator(train_set, train_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)
validation = keras.preprocessing.sequence.TimeseriesGenerator(validation_set, validation_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)
test = keras.preprocessing.sequence.TimeseriesGenerator(test_set, test_set, length=timeWindow, sampling_rate=1, stride=1, batch_size=1)

train[0][0].shape

print("train_set shape: {}".format(train_set.shape))

train_matrix = np.zeros((train_set.shape[0]-timeWindow, timeWindow, 80))
for i in range(timeWindow,train_set.shape[0]):
  train_matrix[i-timeWindow, : , : ] = train[i-timeWindow][0][0]

validation_matrix = np.zeros((validation_set.shape[0]-timeWindow, timeWindow, 80))
for i in range(timeWindow,validation_set.shape[0]):
  validation_matrix[i-timeWindow, : , : ] = validation[i-timeWindow][0][0]

test_matrix = np.zeros((test_set.shape[0]-timeWindow, timeWindow, 80))
for i in range(timeWindow,test_set.shape[0]):
  test_matrix[i-timeWindow, : , : ] = test[i-timeWindow][0][0]

output_train_matrix = np.zeros((train_set.shape[0]-timeWindow, 7))
for i in range(timeWindow,train_set.shape[0]):
  output_train_matrix[i-timeWindow , :] = robot_state_train_label[i-timeWindow:(i-timeWindow+1)]

output_validation_matrix = np.zeros((validation_set.shape[0]-timeWindow, 7))
for i in range(timeWindow,validation_set.shape[0]):
  output_validation_matrix[i-timeWindow, : ] = robot_state_train_label[(intermediate_output_train.shape[0] - validation_set_size + i - timeWindow):(intermediate_output_train.shape[0] - validation_set_size + i - timeWindow)+1]

output_test_matrix = np.zeros((test_set.shape[0]-timeWindow, 7))
for i in range(timeWindow,test_set.shape[0]):
  output_test_matrix[i - timeWindow,:] = robot_state_test_label[(i - timeWindow):(i - timeWindow)+1]

print("output_test_matrix: {}".format(output_test_matrix.shape))


monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, 
        verbose=1, mode='auto', restore_best_weights=True)


GRU_Model = keras.models.Sequential()
GRU_Model.add(keras.layers.GRU(150, return_sequences=True, input_shape=(timeWindow,80 )))
GRU_Model.add(keras.layers.GRU(100, input_shape=(timeWindow,80 )))
GRU_Model.add(keras.layers.Dense(50))
GRU_Model.add(keras.layers.Dense(40))
GRU_Model.add(keras.layers.Dense(7,activation="linear"))
GRU_Model.compile(loss='mae', optimizer=keras.optimizers.Adam(), metrics=['mse','accuracy'])
GRU_Model.fit(train_matrix ,output_train_matrix, callbacks=[monitor], epochs=40, validation_data=(validation_matrix, output_validation_matrix),verbose=2)
score= GRU_Model.evaluate(test_matrix, output_test_matrix) 



test_predict_gru = GRU_Model.predict(test_matrix)

err_GRU_Model= test_predict_gru - output_test_matrix
Gru_err_mean = np.mean(abs(err_GRU_Model))
a = np.where(err_GRU_Model > 0.01)
a = np.asarray(list(zip(*a)))
print("number of err elements higher than 0.01")
print(a.shape)

GRU_Model.save('/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Models/RCN_GRU.h5')

GRU_Model.summary()