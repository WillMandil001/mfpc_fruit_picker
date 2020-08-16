#### Model that doesnt use the image data - just the current strawberry states.

import cv2
import matplotlib
import numpy as np
import pandas as pd
# from PIL import Image
# import tensorflow as tf
# from tensorflow import keras
# import matplotlib.image as mpimg
# from matplotlib import pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.callbacks import EarlyStopping
# np.random.seed(1000)
# from sklearn import preprocessing
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet import preprocess_input

################################## Hyper Parameters ##################################################################
data_set_length = 20
trajectory_length = 5

################################# Robot vel and pos ###################################################################
robot_positions = pd.read_csv('/home/will/Robotics/data_set_003/robot_pos/data_set_' + str(0) + '_robot_data_store_position.csv', header=None)
robot_velocitys = pd.read_csv('/home/will/Robotics/data_set_003/robot_vel/data_set_' + str(0) + '_robot_data_store_velocity.csv', header=None)
for i in range(1, data_set_length):
    robot_positions = pd.concat([robot_positions, pd.read_csv('/home/will/Robotics/data_set_003/robot_pos/data_set_' + str(i) + '_robot_data_store_position.csv', header=None)])
    robot_velocitys = pd.concat([robot_velocitys, pd.read_csv('/home/will/Robotics/data_set_003/robot_vel/data_set_' + str(i) + '_robot_data_store_velocity.csv', header=None)])
if camera == 1:
    robot_state = pd.concat([robot_positions, robot_velocitys], axis=1)
else:
    robot_state_single_cam = pd.concat([robot_positions, robot_velocitys], axis=1)
    robot_state = pd.concat([robot_state, robot_state_single_cam], axis=0)

# robot_states_list = robot_state.values.tolist()
# list_ = []
# for i in range(0, len(robot_state[0]), 4):
#     list_.append(robot_states_list[i])
# robot_states_frame_rate = pd.DataFrame(list_)
# print("Done")

# ################################## Standardization for Robot States ###################################################################
# robot_state_names = robot_states_frame_rate.columns
# scaler = preprocessing.StandardScaler()
# myScaler = scaler.fit(robot_states_frame_rate)
# robot_states_frame_rate = myScaler.transform(robot_states_frame_rate)
# robot_states_frame_rate = pd.DataFrame(robot_states_frame_rate, columns=robot_state_names)
# print(robot_states_frame_rate.shape)

# ################################## Load Strawberry Data ##################################################################
# strawberry_1 = pd.read_csv('/home/will/Robotics/data_set_003/straw_1/data_set_' + str(0) + '_strawberry_data_store_1.csv', delimiter=',', error_bad_lines=False,  header=None)
# strawberry_2 = pd.read_csv('/home/will/Robotics/data_set_003/straw_2/data_set_' + str(0) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None)
# strawberry_3 = pd.read_csv('/home/will/Robotics/data_set_003/straw_3/data_set_' + str(0) + '_strawberry_data_store_3.csv', delimiter=',', error_bad_lines=False, header=None)
# strawberry_4 = pd.read_csv('/home/will/Robotics/data_set_003/straw_4/data_set_' + str(0) + '_strawberry_data_store_4.csv', delimiter=',', error_bad_lines=False, header=None)
# strawberry_5 = pd.read_csv('/home/will/Robotics/data_set_003/straw_5/data_set_' + str(0) + '_strawberry_data_store_5.csv', delimiter=',', error_bad_lines=False, header=None)
# strawberry_states = strawberry_1
# blank = strawberry_2
# counter = 0
# for i in range(counter, 2):
#     strawberry_states = pd.concat([strawberry_states, blank], axis=1)

# for i in range(1, data_set_length):
#     strawberry_cluster_state = pd.read_csv('/home/will/Robotics/data_set_003/straw_1/data_set_' + str(i) + '_strawberry_data_store_1.csv', delimiter=',', error_bad_lines=False, header=None)
#     strawberry_2 = pd.read_csv('/home/will/Robotics/data_set_003/straw_2/data_set_' + str(i) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None)
#     strawberry_3 = pd.read_csv('/home/will/Robotics/data_set_003/straw_3/data_set_' + str(i) + '_strawberry_data_store_3.csv', delimiter=',', error_bad_lines=False, header=None)
#     strawberry_4 = pd.read_csv('/home/will/Robotics/data_set_003/straw_4/data_set_' + str(i) + '_strawberry_data_store_4.csv', delimiter=',', error_bad_lines=False, header=None)
#     strawberry_5 = pd.read_csv('/home/will/Robotics/data_set_003/straw_5/data_set_' + str(i) + '_strawberry_data_store_5.csv', delimiter=',', error_bad_lines=False, header=None)
#     counter = 0
#     if strawberry_2[0][0] != 100 and counter < 2:
#         strawberry_cluster_state = pd.concat([strawberry_cluster_state, strawberry_2], axis=1)
#         counter += 1
#     if strawberry_3[0][0] != 100 and counter < 2:
#         strawberry_cluster_state = pd.concat([strawberry_cluster_state, strawberry_3], axis=1)
#         counter += 1
#     if strawberry_4[0][0] != 100 and counter < 2:
#         strawberry_cluster_state = pd.concat([strawberry_cluster_state, strawberry_4], axis=1)
#         counter += 1
#     if strawberry_5[0][0] != 100 and counter < 2:
#         strawberry_cluster_state = pd.concat([strawberry_cluster_state, strawberry_5], axis=1)
#         counter += 1
#     for i in range(counter, 2):
#         strawberry_cluster_state = pd.concat([strawberry_cluster_state, blank], axis=1)
#     strawberry_states = pd.concat([strawberry_states, strawberry_cluster_state], axis=0)

# strawberry_states_list = strawberry_states.values.tolist()
# list_ = []
# for i in range(0, len(strawberry_states[0]), 4):
#     list_.append(strawberry_states_list[i])
# strawberry_states_frame_rate = pd.DataFrame(list_)

# ################################## Standardization for Strawberry States ###################################################################
# strawberry_state_names = strawberry_states_frame_rate.columns
# scaler = preprocessing.StandardScaler()
# myScaler = scaler.fit(strawberry_states_frame_rate)
# strawberry_states_frame_rate = myScaler.transform(strawberry_states_frame_rate)
# strawberry_states_frame_rate = pd.DataFrame(strawberry_states_frame_rate, columns=strawberry_state_names)
# print(strawberry_states_frame_rate.shape)

# ################################## Order data for time step prediction ###################################################################
# robot_pos_trajectory_input = []
# robot_vel_trajectory_input = []
# strawberry_state_input = []
# strawberry_state_label = []
# for i in range(0, trajectory_length):
# 	