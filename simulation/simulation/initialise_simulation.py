import os
import csv
import time
import copy
import h5py
import socket
import random
import itertools
import pybullet_data
import strawberry_plant
import franka_strawberry_harvester
import math as m
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image

def read_robot_joint_data():
    j1p = p.getJointState(panda.franka, 0)[0]
    j2p = p.getJointState(panda.franka, 1)[0]
    j3p = p.getJointState(panda.franka, 2)[0]
    j4p = p.getJointState(panda.franka, 3)[0]
    j5p = p.getJointState(panda.franka, 4)[0]
    j6p = p.getJointState(panda.franka, 5)[0]
    j7p = p.getJointState(panda.franka, 6)[0]

    j1v = p.getJointState(panda.franka, 0)[1]
    j2v = p.getJointState(panda.franka, 1)[1]
    j3v = p.getJointState(panda.franka, 2)[1]
    j4v = p.getJointState(panda.franka, 3)[1]
    j5v = p.getJointState(panda.franka, 4)[1]
    j6v = p.getJointState(panda.franka, 5)[1]
    j7v = p.getJointState(panda.franka, 6)[1]

    return [j1p, j2p, j3p, j4p, j5p, j6p, j7p], [j1v, j2v, j3v, j4v, j5v, j6v, j7v]

def get_camera_frame(view_matrix_camera_1, width, height, depth=True):
    images = p.getCameraImage(width, height, view_matrix_camera_1, projection_matrix, shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.
    if depth == True:
        depth_buffer_opengl = np.reshape(images[3], [width, height])
        depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
        return rgb_opengl, depth_opengl
    else:
        return rgb_opengl

def control_strawberry(strawberry):
    cluster_r_state = p.getJointState(strawberry.pendulum, 1)[0]
    cluster_p_state = p.getJointState(strawberry.pendulum, 2)[0]
    cluster_y_state = p.getJointState(strawberry.pendulum, 3)[0]
    strawberry.pd_controller_step(cluster_r_state, cluster_p_state, cluster_y_state)

def format_trajectory(trajectory):
    for index, item in enumerate(trajectory):
        try:
            trajectory[index] = item.split(')","')
        except:
            pass
    trajectory = list(itertools.chain(*trajectory))
    step = []
    trajectory_simple = []
    for index, item in enumerate(trajectory):
        if "(" in item:
            trajectory_simple.append(step)
            step = []
        step.append(float(trajectory[index].replace('"', "").replace("(", "").replace(",", "").replace(")", "")))
    trajectory_simple.pop(0)
    return trajectory_simple

################################################### Read in ROS generated trajectories ###################################################
trajectories = []
with open(os.path.expanduser('~/trajectories_cartesian_circle.csv'), newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 0
    for row in spamreader:
        if i == 0:
            strawberry_pose = row[0]
        else:
            trajectories.append(row)
        i+=1
trajectory = format_trajectory(trajectories[random.randint(0, 9999)])  # just one random trajectory to loop through

################################################### Set initial variables for simulation #################################################
# p.connect(p.DIRECT)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
timeStep=1./240.
p.setTimeStep(timeStep)
planeId = p.loadURDF("plane.urdf")

################################################### Load Franka robot ####################################################################
start_pos = [0, 0, 0]
Joint_start_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02]
robot_file_name = "/home/will/Robotics/mfpc_fruit_picking/src/simulation/models/panda_no_gripper/panda.urdf"
panda = franka_strawberry_harvester.FrankaPanda(p, start_pos, timeStep, trajectory[0], robot_file_name)

robot_data_store_position = []
robot_data_store_velocity = []

################################################### Load strawberries ####################################################################
strawberry_file_name = "/home/will/Robotics/mfpc_fruit_picking/src/simulation/models/clusters_urdf/strawberry_cluster.urdf"

# Load CENTER strawberry:
start_pose_1 = []
start_pose_1 = [0.5,0,0.5]
# start_pose_1 = copy.deepcopy(strawberry_pose)
stem_length = 0.15
strawberry_radius = 0.015 + 0.06  # 0.075
start_pose_1[2] = start_pose_1[2] + stem_length + strawberry_radius
start_pose_1[1] += (random.uniform(-1, 1) * 0.01)
start_pose_1[0] += (random.uniform(-1, 1) * 0.01)
r1 = (random.uniform(-10, 10) * m.pi) / 180
r2 = (random.uniform(-10, 10) * m.pi) / 180
r3 = (random.uniform(-180, 180) * m.pi) / 180
start_ori_1 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
strawberry_1 = strawberry_plant.StrawberryCluster(p, start_pose_1, start_ori_1, strawberry_file_name)
strawberry_data_store_1 = []

# Load Second strawberry:
start_pose_2 = []
start_pose_2 = [0.5,0,0.5]
# start_pose_2 = copy.deepcopy(strawberry_pose)
stem_length = 0.15
strawberry_radius = 0.015 + 0.06  # 0.075
start_pose_2[2] = start_pose_2[2] + stem_length + strawberry_radius + (random.uniform(0, -2) * 0.01)
start_pose_2[1] += (random.uniform(-1, 1) * 0.01)
start_pose_2[0] += 0.02 + (random.uniform(-1, 1) * 0.01)
r1 = (random.uniform(-10, 10) * m.pi) / 180
r2 = (random.uniform(-10, 10) * m.pi) / 180
r3 = (random.uniform(-180, 180) * m.pi) / 180
start_ori_2 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
strawberry_2 = strawberry_plant.StrawberryCluster(p, start_pose_2, start_ori_2, strawberry_file_name)
strawberry_data_store_2 = []

# Load Third strawberry:
start_pose_3 = []
start_pose_3 = [0.5,0,0.5]
# start_pose_3 = copy.deepcopy(strawberry_pose)
stem_length = 0.15
strawberry_radius = 0.015 + 0.06  # 0.075
start_pose_3[2] = start_pose_3[2] + stem_length + strawberry_radius + (random.uniform(0, -2) * 0.01)
start_pose_3[1] += (random.uniform(-1, 1) * 0.01)
start_pose_3[0] -= 0.02 + (random.uniform(-1, 1) * 0.01)
r1 = (random.uniform(-10, 10) * m.pi) / 180
r2 = (random.uniform(-10, 10) * m.pi) / 180
r3 = (random.uniform(-180, 180) * m.pi) / 180
start_ori_3 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
strawberry_3 = strawberry_plant.StrawberryCluster(p, start_pose_3, start_ori_3, strawberry_file_name)
strawberry_data_store_3 = []

################################################### Load cameras #########################################################################
# Camera_1
width = 500
height = 500
fov = 69.4
aspect = width / height
near = 0.02
far = 1.5
view_matrix_camera_1 = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5,0,0.45], distance=0.05, yaw=0, pitch=90, roll=0, upAxisIndex=2)
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

images_rgb1 = []
images_depth1 = []

################################################### Start the simulation ##################################################################
for i in range(0, 200):
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)  # Comment out to remove visualisation of simulation

    ## Read Camera data:
    rgb_opengl, depth_opengl = get_camera_frame(view_matrix_camera_1, width, height, depth=True)
    images_rgb1.append(rgb_opengl)
    images_depth1.append(depth_opengl)

    ## Read robot data:
    robot_pos, robot_vel = read_robot_joint_data()
    robot_data_store_position.append(robot_pos)
    robot_data_store_velocity.append(robot_vel)

    ## Control the robot:
    if i < 50:
        panda.step_from_ros(trajectory[0])
    elif i > 50 and i-50 < len(trajectory):
        panda.step_from_ros(trajectory[i-50])
    else:
        panda.step_from_ros(trajectory[-1])

    ### Control the strawberries PID controller:
    control_strawberry(strawberry_1)
    strawberry_data_store_1.append(list(p.getLinkState(strawberry_1.pendulum, 4)[0] + p.getLinkState(strawberry_1.pendulum, 4)[1]))

    control_strawberry(strawberry_2)
    strawberry_data_store_2.append(list(p.getLinkState(strawberry_2.pendulum, 4)[0] + p.getLinkState(strawberry_2.pendulum, 4)[1]))

    control_strawberry(strawberry_3)
    strawberry_data_store_3.append(list(p.getLinkState(strawberry_3.pendulum, 4)[0] + p.getLinkState(strawberry_3.pendulum, 4)[1]))

    p.stepSimulation()
    time.sleep(timeStep)

p.disconnect()
