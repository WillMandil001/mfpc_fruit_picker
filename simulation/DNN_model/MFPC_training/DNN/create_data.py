import cv2
import csv
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


################################## Hyper Parameters ##################################################################
data_location = "/home/will/Robotics/Data_sets/"
no_of_cameras = 8 + 1
data_set_length = 164
trajectory_length = 1
scale_up_value = 100000
single_sample_length = 199
train_size__ = 0.8

################################# Robot vel and pos ###################################################################
robot_positions = []
robot_velocitys = []
for i in range(0, data_set_length+1):
    print("stage 1: ", i, " / ", data_set_length)
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

############################################### CUT data with no movement (PART 1) ###############################################
# to_keep_index = []
# for i in range(0, len(robot_velocitys) - 1):
#     dif = sum(abs(robot_velocitys[i] - robot_velocitys[i+1]))
#     # print("here 3 >> ", i, " / ", len(robot_vel_frame_rate), " >> ", dif)
#     if dif >= 0:
#         to_keep_index.append(i)
# print(len(to_keep_index))
# to_keep_index_all_cams = []
# for camera in range(1, no_of_cameras):
#     to_keep_index_all_cams.append(to_keep_index)

# to_keep_index_singlecam__ = to_keep_index
# to_keep_index = [y for x in to_keep_index_all_cams for y in x]
# print(len(to_keep_index))
# print("cut data part 1")

# ################################## Standardization for Robot States ###################################################################
print("standardise: 1")
print(len(robot_positions))
for i in range(0, len(robot_positions)):
    robot_positions[i] = scale_up_value*robot_positions[i]
print(robot_positions[0])
scaler = preprocessing.StandardScaler()
myScaler = scaler.fit(robot_positions)
robot_positions = myScaler.transform(robot_positions)
print(robot_positions.shape)

print("standardise: 2")
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
        print("stage 2: ", i, " / ", no_of_cameras * len(robot_positions))
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
    print("Stage 3: camera ", camera)
    for i in range(0, data_set_length+1):
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
        for k in range(1, len(robot_positions_order), 4):
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

############################################### Cut data with no movement (PART 2) ###############################################
# images_cut = []
# strawberry_cluster_state_cut = []
# robot_positions_cut = []
# robot_velocitys_cut = []

# for value in to_keep_index:
#     print("Stage 5: cut data >> ")
#     # images_cut.append(images[value])
#     strawberry_cluster_state_cut.append(strawberry_cluster_state[value])
#     robot_positions_cut.append(robot_positions_trajectory[value])
#     robot_velocitys_cut.append(robot_velocitys_trajectory[value])

strawberry_cluster_state_cut = strawberry_cluster_state 
robot_positions_cut = robot_positions_trajectory 
robot_velocitys_cut = robot_velocitys_trajectory 

############################################### Shuffle Data ###############################################
random_shuffle_order = random.sample(range(len(strawberry_cluster_state_cut)), len(strawberry_cluster_state_cut))

# images_cut_shuffled = []
strawberry_cluster_state_cut_shuffled = []
robot_positions_cut_shuffled = []
robot_velocitys_cut_shuffled = []

print("Stage 6: shuffling data")
for value in random_shuffle_order:
    # images_cut_shuffled.append(images_cut[value])
    strawberry_cluster_state_cut_shuffled.append(strawberry_cluster_state_cut[value])
    robot_positions_cut_shuffled.append(robot_positions_cut[value])
    robot_velocitys_cut_shuffled.append(robot_velocitys_cut[value])

strawberry_cluster_state_cut = None
robot_positions_cut = None
robot_velocitys_cut = None
############################################### Normalize image data ###############################################
# for i in range(0 , len(images_cut_shuffled)):
#     print("Stage 7: Normalize >> ", i, " / ", len(images_cut_shuffled))
#     images_cut_shuffled[i] = images_cut_shuffled[i] / np.max(images_cut_shuffled[i])

############################################### Cut into test and train data ###############################################
data_set_lenth = len(robot_positions_cut_shuffled)
# images_train = images_cut_shuffled[0:int(data_set_lenth*train_size__)]
# images_test = images_cut_shuffled[int(data_set_lenth*train_size__):data_set_lenth]

strawberry_cluster_train = strawberry_cluster_state_cut_shuffled[0:int(data_set_lenth*train_size__)]
strawberry_cluster_test = strawberry_cluster_state_cut_shuffled[int(data_set_lenth*train_size__):data_set_lenth]

robot_positions_train = robot_positions_cut_shuffled[0:int(data_set_lenth*train_size__)]
robot_positions_test = robot_positions_cut_shuffled[int(data_set_lenth*train_size__):data_set_lenth]

robot_velocitys_train = robot_velocitys_cut_shuffled[0:int(data_set_lenth*train_size__)]
robot_velocitys_test = robot_velocitys_cut_shuffled[int(data_set_lenth*train_size__):data_set_lenth]

############################################### TF needs numpy arrays ###############################################
# images_train = np.asarray(images_train)
robot_positions_train = np.asarray(robot_positions_train)
robot_velocitys_train = np.asarray(robot_velocitys_train)
strawberry_cluster_train = np.asarray(strawberry_cluster_train)
# images_test = np.asarray(images_test)
robot_positions_test = np.asarray(robot_positions_test)
robot_velocitys_test = np.asarray(robot_velocitys_test)
strawberry_cluster_test = np.asarray(strawberry_cluster_test)

################################## SAVE DATA AS CSV ###################################################################
with open(data_location + 'data_set_003/processed_001/robot_positions_train_FOR_CLUSTER_001', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(robot_positions_train)
with open(data_location + 'data_set_003/processed_001/robot_velocitys_train_FOR_CLUSTER_001', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(robot_velocitys_train)
with open(data_location + 'data_set_003/processed_001/strawberry_cluster_train_FOR_CLUSTER_001', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(strawberry_cluster_train)
with open(data_location + 'data_set_003/processed_001/robot_positions_test_FOR_CLUSTER_001', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(robot_positions_test)
with open(data_location + 'data_set_003/processed_001/robot_velocitys_test_FOR_CLUSTER_001', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(robot_velocitys_test)
with open(data_location + 'data_set_003/processed_001/strawberry_cluster_test_FOR_CLUSTER_001', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(strawberry_cluster_test)
# with open(data_location + 'data_set_003/processed_001/to_keep_index', "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(to_keep_index_all_cams)
with open(data_location + 'data_set_003/processed_001/random_shuffle_order', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(map(lambda x: [x], random_shuffle_order))


robot_positions_train = None
robot_velocitys_train = None
strawberry_cluster_train = None
robot_positions_test = None
robot_velocitys_test = None
strawberry_cluster_test = None

print(len(random_shuffle_order), 49*(data_set_length+1)*8)
############################################## Load image data #####################################################
order_ = 0
for camera in range(1, no_of_cameras):
    for i in range(0, data_set_length+1):
        print("Stage 4: camera: ", camera, " data_set_length >> ", i, " / ", data_set_length)
        for j in range(0, 50):
            im_frame = cv2.imread(data_location + 'data_set_003/camera_' + str(camera) + '/rgb/sample_' + str(i) + '/time_step_' + str(j) + '.png')
            im_frame = im_frame / np.max(im_frame)
            cv2.imwrite((data_location + "data_set_003/processed_001/images_shuffled/image_" + str(int(random_shuffle_order[order_])) + ".png"), im_frame)
            order_ +=1


    # # images_cut = []
    # # for value in to_keep_index_singlecam__:  # cut the no movement data
    # #     print("Stage 5: camera: ", len(to_keep_index_singlecam__))
    # #     images_cut.append(images[value])
    # images = []
    # for i in range(0 , len(images_cut)):  # normalise
    #     print("Stage 6: camera: ", i, "/", len(images_cut))
    #     images_cut[i] = images_cut[i] / np.max(images_cut[i])

    # for j in range(0, len(images_cut)):  # save with shuffled name:
    #     print("Stage 7: camera: ", j, "/", len(images_cut))
    #     pd.DataFrame(images_cut[j]).to_csv(data_location + "data_set_003/processed_001/images_shuffled/image_" + int(random_shuffle_order[order_]) + ".csv")
    #     order_ +=1



