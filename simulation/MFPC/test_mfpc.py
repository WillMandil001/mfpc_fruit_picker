import os
import csv
import random
import time
import math as m
import copy
import socket
import pybullet_data
import strawberry_cluster_mfpc
import franka_panda_new_ee_mfpc
import numpy as np
import pybullet as p


file_name = 0
ee_length = 0.25
MPC_traj_length = 7
# Format trajectories from ros to pybullet:
trajectories = []


def strawberry_pd_controller():
	cluster_r_state = p.getJointState(strawberry_1.pendulum, 1)[0]
	cluster_p_state = p.getJointState(strawberry_1.pendulum, 2)[0]
	cluster_y_state = p.getJointState(strawberry_1.pendulum, 3)[0]
	strawberry_1.pd_controller_step(cluster_r_state, cluster_p_state, cluster_y_state)
	cluster_r_state = p.getJointState(strawberry_2.pendulum, 1)[0]
	cluster_p_state = p.getJointState(strawberry_2.pendulum, 2)[0]
	cluster_y_state = p.getJointState(strawberry_2.pendulum, 3)[0]
	strawberry_2.pd_controller_step(cluster_r_state, cluster_p_state, cluster_y_state)
	cluster_r_state = p.getJointState(strawberry_3.pendulum, 1)[0]
	cluster_p_state = p.getJointState(strawberry_3.pendulum, 2)[0]
	cluster_y_state = p.getJointState(strawberry_3.pendulum, 3)[0]
	strawberry_3.pd_controller_step(cluster_r_state, cluster_p_state, cluster_y_state)

def get_states():
	j1p = p.getJointState(panda.franka, 0)[0]
	j2p = p.getJointState(panda.franka, 1)[0]
	j3p = p.getJointState(panda.franka, 2)[0]
	j4p = p.getJointState(panda.franka, 3)[0]
	j5p = p.getJointState(panda.franka, 4)[0]
	j6p = p.getJointState(panda.franka, 5)[0]
	j7p = p.getJointState(panda.franka, 6)[0]
	robot_data_store_position = [j1p, j2p, j3p, j4p, j5p, j6p, j7p]
	j1v = p.getJointState(panda.franka, 0)[1]
	j2v = p.getJointState(panda.franka, 1)[1]
	j3v = p.getJointState(panda.franka, 2)[1]
	j4v = p.getJointState(panda.franka, 3)[1]
	j5v = p.getJointState(panda.franka, 4)[1]
	j6v = p.getJointState(panda.franka, 5)[1]
	j7v = p.getJointState(panda.franka, 6)[1]
	robot_data_store_velocity = [j1v, j2v, j3v, j4v, j5v, j6v, j7v]

	robot_ee_position = p.getLinkState(panda.franka, 7)[0]
	# strawberry states
	staw_1_pos = p.getLinkState(strawberry_1.pendulum, 4)[0]
	staw_1_ori = p.getLinkState(strawberry_1.pendulum, 4)[1]

	staw_2_pos = p.getLinkState(strawberry_2.pendulum, 4)[0] 
	staw_2_ori = p.getLinkState(strawberry_2.pendulum, 4)[1]

	staw_3_pos = p.getLinkState(strawberry_3.pendulum, 4)[0] 
	staw_3_ori = p.getLinkState(strawberry_3.pendulum, 4)[1]

	return robot_ee_position, robot_data_store_position, robot_data_store_velocity, [staw_1_pos, staw_1_ori], [staw_2_pos, staw_2_ori], [staw_3_pos, staw_3_ori]

def create_trajectory(robot_ee_position, robot_goal_pose):
	if robot_ee_position[0] >= robot_goal_pose[0]:
		x_traj = np.arange(robot_ee_position[0], robot_goal_pose[0], -0.005).tolist()
	else:
		x_traj = np.arange(robot_ee_position[0], robot_goal_pose[0], 0.005).tolist()
	if robot_ee_position[1] >= robot_goal_pose[1]:
		y_traj = np.arange(robot_ee_position[1], robot_goal_pose[1], -0.005).tolist()
	else:
		y_traj = np.arange(robot_ee_position[1], robot_goal_pose[1], 0.005).tolist()
	if robot_ee_position[2] >= robot_goal_pose[2]:
		z_traj = np.arange(robot_ee_position[2], robot_goal_pose[2], -0.005).tolist()
	else:
		z_traj = np.arange(robot_ee_position[2], robot_goal_pose[2], 0.005).tolist()

	if x_traj == []:
		x_traj = [robot_goal_pose[0], robot_goal_pose[0]]
	if y_traj == []:
		y_traj = [robot_goal_pose[1], robot_goal_pose[1]]
	if z_traj == []:
		z_traj = [robot_goal_pose[2], robot_goal_pose[2]]

	if len(x_traj) == 1:
		x_traj = [x_traj[0], x_traj[0]]
	if len(y_traj) == 1:
		y_traj = [y_traj[0], y_traj[0]]
	if len(z_traj) == 1:
		z_traj = [z_traj[0], z_traj[0]]

	return x_traj, y_traj, z_traj

def get_distances(robot_ee_position, straw_1_pose, straw_2_pose, straw_3_pose):
	d_s1 = m.sqrt(((straw_1_pose[0][0] - robot_ee_position[0])**2) + ((straw_1_pose[0][1] - robot_ee_position[1])**2) + ((straw_1_pose[0][2] + robot_z_offset - robot_ee_position[2])**2))
	d_s2 = m.sqrt(((straw_2_pose[0][0] - robot_ee_position[0])**2) + ((straw_2_pose[0][1] - robot_ee_position[1])**2) + ((straw_2_pose[0][2] + robot_z_offset - robot_ee_position[2])**2))
	d_s3 = m.sqrt(((straw_3_pose[0][0] - robot_ee_position[0])**2) + ((straw_3_pose[0][1] - robot_ee_position[1])**2) + ((straw_3_pose[0][2] + robot_z_offset - robot_ee_position[2])**2))
	return  d_s1, d_s2, d_s3

def calculate_priority_strawberry(robot_ee_position, straw_1_pose, straw_2_pose, straw_3_pose):
	priority = [1,1,1]
	main_axis_s1 = robot_ee_position[0] - straw_1_pose[0][0]
	main_axis_s2 = robot_ee_position[0] - straw_2_pose[0][0]
	main_axis_s3 = robot_ee_position[0] - straw_3_pose[0][0]

	if main_axis_s1 > 0.0:
		priority[2] = 0
	if main_axis_s2 > 0.0:
		priority[1] = 0
	if main_axis_s3 > 0.0:
		priority[0] = 0

	return priority


with open(os.path.expanduser('~/trajectories_cartesian_circle.csv'), newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	i = 0
	for row in spamreader:
		if i == 0:
			strawberry_pose = row[0]
		else:
			trajectories.append(row)
		i+=1

strawberry_pose = strawberry_pose.split(",")
for index, item in enumerate(strawberry_pose):
	strawberry_pose[index] = float(item)

with_ros = True

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
timeStep=1./120.
p.setTimeStep(timeStep)
planeId = p.loadURDF("plane.urdf")

# Load franka:
start_pos = [0,0,0]
Joint_start_state = [-1.433995977309579, -1.7617816763032867, 1.4325302980067844, -2.5136029912385243, -1.5686485073323206, 1.8073010767995468, -2.34824539264451, 0.02, 0.02]
panda = franka_panda_new_ee_mfpc.FrankaPanda(p, start_pos, timeStep, Joint_start_state, MPC_traj_length)

## load strawberry cluster problem:
start_pose_1 = []
start_pose_1 = copy.deepcopy(strawberry_pose)
stem_length = 0.15
strawberry_radius = 0.015 + 0.06  # 0.075
start_pose_1[2] = start_pose_1[2] + stem_length + strawberry_radius
start_pose_1[0] += 0.02
r1 = 0
r2 = 0
r3 = 0
start_ori_1 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
strawberry_1 = strawberry_cluster_mfpc.StrawberryCluster(p, start_pose_1, start_ori_1, "models/clusters_urdf/strawberry_cluster.urdf")

start_pose_2 = []
start_pose_2 = copy.deepcopy(strawberry_pose)
stem_length = 0.15
strawberry_radius = 0.015 + 0.06  # 0.075
start_pose_2[2] = start_pose_2[2] + stem_length + strawberry_radius - 0.015
start_pose_2[0] -= 0.025
r1 = 0
r2 = 0
r3 = 0
start_ori_2 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
strawberry_2 = strawberry_cluster_mfpc.StrawberryCluster(p, start_pose_2, start_ori_2, "models/clusters_urdf/strawberry_cluster_unripe.urdf")

start_pose_3 = []
start_pose_3 = copy.deepcopy(strawberry_pose)
stem_length = 0.15
strawberry_radius = 0.015 + 0.06  # 0.075
start_pose_3[2] = start_pose_3[2] + stem_length + strawberry_radius - 0.0075
start_pose_3[0] -= 0.01

r1 = 0
r2 = 0
r3 = 0
start_ori_3 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
strawberry_3 = strawberry_cluster_mfpc.StrawberryCluster(p, start_pose_3, start_ori_3, "models/clusters_urdf/strawberry_cluster_unripe.urdf")

# Need to define goal states for obstical strawberrys:
straw_1_goal = [start_pose_1[0], start_pose_1[1], start_pose_1[2]-0.12]  # stay the same for the goal strawb
straw_1_goal_vis = p.loadURDF("models/clusters_urdf/goal_state_visual.urdf", np.array(straw_1_goal), [0,0,0,1], useFixedBase=True)

straw_2_goal = [start_pose_2[0], start_pose_2[1]+ 0.02, start_pose_2[2]-0.12]
straw_2_goal_vis = p.loadURDF("models/clusters_urdf/goal_state_visual.urdf", np.array(straw_2_goal), [0,0,0,1], useFixedBase=True)

straw_3_goal = [start_pose_3[0], start_pose_3[1]+ 0.02, start_pose_3[2]-0.12]
straw_3_goal_vis = p.loadURDF("models/clusters_urdf/goal_state_visual.urdf", np.array(straw_3_goal), [0,0,0,1], useFixedBase=True)

# generate initial trajectory.
robot_start_pose = [start_pose_1[0]-0.2, start_pose_1[1] - 0.025, start_pose_1[2]-0.02 - ee_length]
robot_goal_pose = [start_pose_1[0], start_pose_1[1] - 0.025, start_pose_1[2]-0.02 - ee_length]

# offset for ee (start_pose_1[0], start_pose_1[1], start_pose_1[2]-0.02 - ee_length)

trajectory_length = 100
trajectory_x = list(np.linspace(robot_start_pose[0], robot_goal_pose[0], num=trajectory_length, retstep=True))[0]
trajectory_y = list(np.linspace(robot_start_pose[1], robot_goal_pose[1], num=trajectory_length, retstep=True))[0]
trajectory_z = list(np.linspace(robot_start_pose[2], robot_goal_pose[2], num=trajectory_length, retstep=True))[0]
count = 0
run_trajectory = 0
robot_z_offset = -0.12
for i in range(0,1000):
	p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
	strawberry_pd_controller()

	if i > 10:
		print(">>>>>>>>>>>>")
		robot_ee_position, robot_j_pos, robot_j_vel, straw_1_pose, straw_2_pose, straw_3_pose = get_states()
		x_traj, y_traj, z_traj = create_trajectory(robot_ee_position, robot_goal_pose)
		priority = calculate_priority_strawberry(robot_ee_position, straw_1_pose, straw_2_pose, straw_3_pose)
		d_s1, d_s2, d_s3 = get_distances(robot_ee_position, straw_1_pose, straw_2_pose, straw_3_pose)
		if priority == [0, 0, 1]:
			ee_state = panda.step([x_traj[1], y_traj[1], z_traj[1]])	
		elif d_s1 < 0.03 or d_s2 < 0.03 or d_s3 < 0.03:  # if dist < thresh start MFPC
			print("MFPC", priority, d_s1, d_s2, d_s3)
			print(straw_3_pose)
			# ee_state = panda.step([x_traj[1], y_traj[1], z_traj[1]])  # SWAP FOR MFPC
			trajectory_list_task, joint = panda.step_MFPC(priority, robot_ee_position, [x_traj, y_traj, z_traj], [straw_1_pose, straw_2_pose, straw_3_pose], [straw_1_goal, straw_2_goal, straw_3_goal])
			ee_state = panda.step([trajectory_list_task[1][0], (trajectory_list_task[1][1]), trajectory_list_task[1][2]])
			# ee_state = panda.step_joint(joint)
			print("===============================")
			print("===============================")
			print(trajectory_list_task[1][0], (trajectory_list_task[1][1]), trajectory_list_task[1][2])
			print("================")
			print([x_traj[1], y_traj[1], z_traj[1]])
			print("===============================")
			print("===============================")
		else:
			print("Initial trajectory", priority, d_s1, d_s2, d_s3)
			ee_state = panda.step([x_traj[1], y_traj[1], z_traj[1]])	
	else:
		ee_state = panda.step(robot_start_pose)


	# if i > 10:
	# 	print(">>>>>>>>>>>>")
	# 	robot_ee_position, robot_j_pos, robot_j_vel, straw_1_pose, straw_2_pose, straw_3_pose = get_states()
	# 	priority = calculate_priority_strawberry(robot_ee_position, straw_1_pose, straw_2_pose, straw_3_pose)
	# 	x_traj, y_traj, z_traj = create_trajectory(robot_ee_position, robot_goal_pose)
	# 	d_s1, d_s2, d_s3 = get_distances(robot_ee_position, straw_1_pose, straw_2_pose, straw_3_pose)

	# 	if d_s1 < 0.075 or d_s2 < 0.075 or d_s3 < 0.075:  # if dist < thresh start MFPC
	# 		print("MFPC", d_s1, d_s2, d_s3)
	# 		trajectory_list_joint, trajectory_list_task = panda.step_MFPC(priority, [x_traj, y_traj, z_traj], [straw_1_pose, straw_2_pose, straw_3_pose], [straw_1_goal, straw_2_goal, straw_3_goal])

	# 	else:
	# 		print("Initial trajectory", d_s1, d_s2, d_s3)
	# 		ee_state = panda.step([x_traj[1], y_traj[1], z_traj[1]])

	# else:
	# 	ee_state = panda.step(robot_start_pose)
	
	p.stepSimulation()
	time.sleep(timeStep)





# # generate initial trajectory.
# robot_start_pose = [start_pose_1[0]-0.2, start_pose_1[1], start_pose_1[2]-0.01 - ee_length]
# robot_goal_pose = [start_pose_1[0], start_pose_1[1], start_pose_1[2]-0.01 - ee_length]

# trajectory_length = 100
# trajectory_x = list(np.linspace(robot_start_pose[0], robot_goal_pose[0], num=trajectory_length, retstep=True))[0]
# trajectory_y = list(np.linspace(robot_start_pose[1], robot_goal_pose[1], num=trajectory_length, retstep=True))[0]
# trajectory_z = list(np.linspace(robot_start_pose[2], robot_goal_pose[2], num=trajectory_length, retstep=True))[0]

# for i in range(0,1000):
# 	p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
# 	strawberry_pd_controller()

# 	if i > 50:
# 		robot_ee_position, robot_j_pos, robot_j_vel, straw_1_pose, straw_2_pose, straw_3_pose = get_states()
# 		x_traj, y_traj, z_traj = create_trajectory(robot_ee_position, robot_goal_pose)
# 		panda.step([x_traj[1], y_traj[1], z_traj[1]])
# 		print("here")
# 		print(len(x_traj))
# 		print(len(y_traj))
# 		print(len(z_traj))

# 	# if i > 20:
# 	# 	robot_ee_position, robot_j_pos, robot_j_vel, straw_1_pose, straw_2_pose, straw_3_pose = get_states()
# 	# 	x_traj, y_traj, z_traj = create_trajectory(robot_ee_position, robot_goal_pose)
# 	# 	panda.step([x_traj[1], y_traj[1], z_traj[1]])

# 	# if i < len(trajectory_x):
# 	# 	# Calculate euclidean distance for robot EE to strawberries
# 	# 	robot_j_pos, robot_j_vel, straw_1_pose, straw_2_pose, straw_3_pose = get_states()
# 	# 	robot_ee_position = p.getLinkState(panda.franka, 9)[0]

# 	# 	d_s1 = m.sqrt(((robot_ee_position[0] - straw_1_pose[0][0])**2) + ((robot_ee_position[1] - straw_1_pose[0][1])**2) + (((robot_ee_position[2]+ 0.05) - straw_1_pose[0][2])**2))
# 	# 	d_s2 = m.sqrt(((robot_ee_position[0] - straw_2_pose[0][0])**2) + ((robot_ee_position[1] - straw_2_pose[0][1])**2) + (((robot_ee_position[2]+ 0.05) - straw_2_pose[0][2])**2))
# 	# 	d_s3 = m.sqrt(((robot_ee_position[0] - straw_3_pose[0][0])**2) + ((robot_ee_position[1] - straw_3_pose[0][1])**2) + (((robot_ee_position[2]+ 0.05) - straw_3_pose[0][2])**2))

# 	# 	# if dist < thresh start MFPC
# 	# 	if d_s1 < 0.05 or d_s2 < 0.05 or d_s3 < 0.05:
# 	# 		print("MFPC", d_s1, d_s2, d_s3)
# 	# 		panda.step_MFPC([trajectory_x[i],trajectory_y[i],trajectory_z[i]])
# 	# 	else:
# 	# 		print("Initial trajectory", d_s1, d_s2, d_s3)
# 	# 		panda.step([trajectory_x[i],trajectory_y[i],trajectory_z[i]])
# 	# 	trajectory_length -= 1
# 	# else:
# 	# 	pass
# 	# 	# panda.step([trajectory_x[-1],trajectory_y[-1],trajectory_z[-1]])
# 	# 	# print("Initial trajectory", d_s1, d_s2, d_s3)

# 	p.stepSimulation()
# 	time.sleep(timeStep)
