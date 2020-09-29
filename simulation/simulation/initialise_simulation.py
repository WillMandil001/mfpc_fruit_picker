import os
import csv
import time
import copy
import h5py
import socket
import random
import itertools
import franka_panda
import pybullet_data
import strawberry_cluster
import franka_panda_new_EE
import math as m
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image

p.connect(p.DIRECT)
# p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
timeStep=1./120.
p.setTimeStep(timeStep)
planeId = p.loadURDF("plane.urdf")

# Load franka:
start_pos = [0, 0, 0]
Joint_start_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02]
panda = franka_panda_new_EE.FrankaPanda(p, start_pos, timeStep, Joint_start_state)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# load CENTER strawberry:
start_pose_1 = []
start_pose_1 = copy.deepcopy(strawberry_pose)
stem_length = 0.15
strawberry_radius = 0.015 + 0.06  # 0.075
start_pose_1[2] = start_pose_1[2] + stem_length + strawberry_radius
start_pose_1[1] += (random.uniform(-1, 1) * 0.01)
start_pose_1[0] += (random.uniform(-1, 1) * 0.01)
r1 = (random.uniform(-10, 10) * m.pi) / 180
r2 = (random.uniform(-10, 10) * m.pi) / 180
r3 = (random.uniform(-180, 180) * m.pi) / 180
start_ori_1 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
strawberry_1 = strawberry_cluster.StrawberryCluster(p, start_pose_1, start_ori_1, "models/clusters_urdf/strawberry_cluster.urdf")


strawberry_data_store_1 = []

robot_data_store_position = []
robot_data_store_velocity = []

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# Camera_1
width = 350
height = 350
fov = 69.4
aspect = width / height
near = 0.02
far = 1.5

view_matrix_camera_1 = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0.65], distance=0.1, yaw=-90, pitch=0, roll=0, upAxisIndex=2)
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)


for i in range(0, 200):
    # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    # camera 1
    images = p.getCameraImage(width, height, view_matrix_camera_1, projection_matrix, shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.
    depth_buffer_opengl = np.reshape(images[3], [width, height])
    depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
    images_rgb1.append(rgb_opengl)
    images_depth1.append(depth_opengl)

    # j1p = p.getJointState(panda.franka, 0)[0]
    # j2p = p.getJointState(panda.franka, 1)[0]
    # j3p = p.getJointState(panda.franka, 2)[0]
    # j4p = p.getJointState(panda.franka, 3)[0]
    # j5p = p.getJointState(panda.franka, 4)[0]
    # j6p = p.getJointState(panda.franka, 5)[0]
    # j7p = p.getJointState(panda.franka, 6)[0]
    # robot_data_store_position.append([j1p, j2p, j3p, j4p, j5p, j6p, j7p])
    # j1v = p.getJointState(panda.franka, 0)[1]
    # j2v = p.getJointState(panda.franka, 1)[1]
    # j3v = p.getJointState(panda.franka, 2)[1]
    # j4v = p.getJointState(panda.franka, 3)[1]
    # j5v = p.getJointState(panda.franka, 4)[1]
    # j6v = p.getJointState(panda.franka, 5)[1]
    # j7v = p.getJointState(panda.franka, 6)[1]
    # robot_data_store_velocity.append([j1v, j2v, j3v, j4v, j5v, j6v, j7v])

    # if i < 50:
    #     panda.step_from_ros(trajectory_simple[0])
    # elif i > 50 and i-50 < len(trajectory_simple):
    #     panda.step_from_ros(trajectory_simple[i-50])
    # else:
    #     panda.step_from_ros(trajectory_simple[-1])

    # Strawberry PID controller:
    cluster_r_state = p.getJointState(strawberry_1.pendulum, 1)[0]
    cluster_p_state = p.getJointState(strawberry_1.pendulum, 2)[0]
    cluster_y_state = p.getJointState(strawberry_1.pendulum, 3)[0]
    strawberry_1.pd_controller_step(cluster_r_state, cluster_p_state, cluster_y_state)
    strawberry_data_store_1.append(list(p.getLinkState(strawberry_1.pendulum, 4)[0] + p.getLinkState(strawberry_1.pendulum, 4)[1]))

    p.stepSimulation()
    time.sleep(timeStep)

p.disconnect()
