import numpy as np
import matplotlib
import pandas as pd
from PIL import Image
import matplotlib.image as mpimg
from sklearn import preprocessing
from matplotlib import pyplot as plt


################################# Robot vel and pos ###################################################################
data_set_length = 5
# robot_positions = pd.read_csv('/vol/hd/mfpc_data/data_set_003/robot_pos/data_set_' + str(0) + '_robot_data_store_position.csv', header=None)
# robot_velocitys = pd.read_csv('/vol/hd/mfpc_data/data_set_003/robot_vel/data_set_' + str(0) + '_robot_data_store_velocity.csv', header=None)
# for i in range(1, data_set_length):
#     robot_positions = pd.concat([robot_positions, pd.read_csv('/vol/hd/mfpc_data/data_set_003/robot_pos/data_set_' + str(i) + '_robot_data_store_position.csv', header=None)])
#     robot_velocitys = pd.concat([robot_velocitys, pd.read_csv('/vol/hd/mfpc_data/data_set_003/robot_vel/data_set_' + str(i) + '_robot_data_store_velocity.csv', header=None)])
# robot_state = pd.concat([robot_positions, robot_velocitys], axis=1)

################################## Load Strawberry data ##################################################################
strawberry_1 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_1/data_set_' + str(0) + '_strawberry_data_store_1.csv', delimiter=',', error_bad_lines=False,  header=None)
strawberry_2 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_2/data_set_' + str(0) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_3 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_3/data_set_' + str(0) + '_strawberry_data_store_3.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_4 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_4/data_set_' + str(0) + '_strawberry_data_store_4.csv', delimiter=',', error_bad_lines=False, header=None)
strawberry_5 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_5/data_set_' + str(0) + '_strawberry_data_store_5.csv', delimiter=',', error_bad_lines=False, header=None)

# for i in range(1, data_set_length):
#     strawberry_1 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_1/data_set_' + str(i) + '_strawberry_data_store_1.csv', delimiter=',', error_bad_lines=False, header=None)
#     strawberry_2 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_2/data_set_' + str(i) + '_strawberry_data_store_2.csv', delimiter=',', error_bad_lines=False, header=None)
#     strawberry_3 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_3/data_set_' + str(i) + '_strawberry_data_store_3.csv', delimiter=',', error_bad_lines=False, header=None)
#     strawberry_4 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_4/data_set_' + str(i) + '_strawberry_data_store_4.csv', delimiter=',', error_bad_lines=False, header=None)
#     strawberry_5 = pd.read_csv('/vol/hd/mfpc_data/data_set_003/straw_5/data_set_' + str(i) + '_strawberry_data_store_5.csv', delimiter=',', error_bad_lines=False, header=None)
#     print(strawberry_2[0])
#     break

print strawberry_1 

# strawberry_states = pd.concat([robot_positions, robot_velocitys], axis=1)

# ################################## Standardization ###################################################################
# robot_position_names = robot_positions.columns
# robot_velocity_names = robot_velocitys.columns
# scaler = preprocessing.StandardScaler()
# myScaler = scaler.fit(robot_positions)
# robot_positions = myScaler.transform(robot_positions)
# robot_velocitys = myScaler.transform(robot_velocitys)

# robot_positions = pd.DataFrame(robot_positions, columns=robot_position_names)
# robot_velocitys = pd.DataFrame(robot_velocitys, columns=robot_velocity_names)

# ############################################### Load image data #####################################################
# train_images = []
# for i in range(0, data_set_length):
#     for j in range(0, 50):
#         im_frame = mpimg.imread('/vol/hd/mfpc_data/data_set_003/camera_1/rgb/sample_' + str(i) + '/time_step_' + str(j) + '.png')
#         train_images.append(im_frame)

# train_images = np.asarray(train_images)
# train_images = train_images.astype(np.float64)
# print("train_imagesing dataset size : {}".format(train_images.shape[0]))

# ############################################### normalize image data ###############################################
# for i in range(0 , train_images.shape[0]):
#   train_images[i, : , : , :] = train_images[i, : , : , :] / np.max(train_images[i, : , : , :])

# robot_state_train_input = robot_positions[0:train_images.shape[0]]
# print("Robot state input trainingset size: {}".format(robot_state_train_input.shape))
# robot_state_train_label = robot_velocitys[0:train_images.shape[0]]
# print("Robot state label trainingset size: {}".format(robot_state_train_label.shape))

# test = D3data[D3data.shape[0]-3000: , : , : , :]
# test = test.astype(np.float64)

# for i in range(0 , test.shape[0]):
#   test[i, : , : , :] = test[i, : , : , :] / np.max(test[i, : , : , :])

# robot_state_test_input = robot_positions[train_images.shape[0]:train_images.shape[0]+test.shape[0]]
# print("Robot state input testset size: {}".format(robot_state_test_input.shape))
# robot_state_test_label = robot_velocitys[train_images.shape[0]:train_images.shape[0]+test.shape[0]]
# print("Robot state label testset size: {}".format(robot_state_test_label.shape))