import os
import csv
import math as m
import time
import copy
import socket
import random
import itertools
import franka_panda
import pybullet_data
import strawberry_cluster
import franka_panda_new_EE
import numpy as np
import pybullet as p

file_name = 0
############# Format trajectories from ros to pybullet:
trajectories = []
clusters = ["I", "H", "F", "D", "B"]
cluster_order = [
				[1,1,0,0,0],
				[1,1,1,0,0],
				[1,1,0,1,0],
				[1,1,0,0,1],
				[1,0,1,0,0],
				[1,0,1,1,0],
				[1,0,1,0,1],
				[1,0,0,1,0],
				[1,0,0,1,1],
				[1,0,0,0,1],]

with open(os.path.expanduser('~/trajectories_cartesian_0_05_upsidedown_ee0_003.csv'), newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	i = 0
	for row in spamreader:
		if i == 0:
			strawberry_pose = row[0]
		else:
			break
			trajectories.append(row)
		i+=1
		break

strawberry_pose = strawberry_pose.split(",")
for index, item in enumerate(strawberry_pose):
	strawberry_pose[index] = float(item)

# trajectories = trajectories[6550:6551]
for current_cluster in cluster_order:
	# p.connect(p.DIRECT)
	p.connect(p.GUI)
	p.setAdditionalSearchPath(pybullet_data.getDataPath())

	p.setGravity(0, 0, -9.81)
	timeStep=1./240.
	p.setTimeStep(timeStep)
	planeId = p.loadURDF("plane.urdf")

	## Load franka:
	start_pos = [0, 0, 0]
	Joint_start_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02]
	panda = franka_panda_new_EE.FrankaPanda(p, start_pos, timeStep, Joint_start_state)

	## load CENTER strawberry:
	if current_cluster[0] == 1:
		start_pose_1 = []
		start_pose_1 = copy.deepcopy(strawberry_pose)
		stem_length = 0.15
		strawberry_radius = 0.015 + 0.06  # 0.075
		start_pose_1[2] = start_pose_1[2] + stem_length + strawberry_radius + (random.uniform(-1, 1) * 0.01)
		start_pose_1[1] += (random.uniform(-1, 1) * 0.01)
		start_pose_1[0] += (random.uniform(-1, 1) * 0.01)
		r1 = (random.uniform(-10, 10) * m.pi) / 180
		r2 = (random.uniform(-10, 10) * m.pi) / 180
		r3 = (random.uniform(-180, 180) * m.pi) / 180
		start_ori_1 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
		strawberry_1 = strawberry_cluster.StrawberryCluster(p, start_pose_1, start_ori_1, "models/clusters_urdf/strawberry_cluster.urdf")
	else:
		start_pose_1 = [False]
		start_ori_1 = [False]

	## load H - LEFT strawberry:
	if current_cluster[1] == 1:
		start_pose_2 = []
		start_pose_2 = copy.deepcopy(strawberry_pose)
		stem_length = 0.15
		strawberry_radius = 0.015 + 0.06  # 0.075
		start_pose_2[2] = start_pose_2[2] + stem_length + strawberry_radius + (random.uniform(0, -2) * 0.01)
		start_pose_2[1] += (random.uniform(-1, 1) * 0.01)
		start_pose_2[0] += 0.02 + (random.uniform(-1, 1) * 0.01)
		r1 = (random.uniform(-10, 10) * m.pi) / 180
		r2 = (random.uniform(-10, 10) * m.pi) / 180
		r3 = (random.uniform(-180, 180) * m.pi) / 180
		start_ori_2 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
		strawberry_2 = strawberry_cluster.StrawberryCluster(p, start_pose_2, start_ori_2, "models/clusters_urdf/strawberry_cluster.urdf")
	else:
		start_pose_2 = [False]
		start_ori_2 = [False]

	## load F - INFRONT strawberry:
	if current_cluster[2] == 1:
		start_pose_3 = []
		start_pose_3 = copy.deepcopy(strawberry_pose)
		stem_length = 0.15
		strawberry_radius = 0.015 + 0.06  # 0.075 
		start_pose_3[2] = start_pose_3[2] + stem_length + strawberry_radius  + (random.uniform(0, -2) * 0.01)
		start_pose_3[1] += 0.02 + (random.uniform(-1, 1) * 0.01)
		start_pose_3[0] += (random.uniform(-1, 1) * 0.01)
		r1 = (random.uniform(-10, 10) * m.pi) / 180
		r2 = (random.uniform(-10, 10) * m.pi) / 180
		r3 = (random.uniform(-180, 180) * m.pi) / 180
		start_ori_3 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
		strawberry_3 = strawberry_cluster.StrawberryCluster(p, start_pose_3, start_ori_3, "models/clusters_urdf/strawberry_cluster.urdf")
	else:
		start_pose_3 = [False]
		start_ori_3 = [False]

	## load D - RIGHT strawberry:
	if current_cluster[3] == 1:
		start_pose_4 = []
		start_pose_4 = copy.deepcopy(strawberry_pose)
		stem_length = 0.15
		strawberry_radius = 0.015 + 0.06  # 0.075
		start_pose_4[2] = start_pose_4[2] + stem_length + strawberry_radius + (random.uniform(0, -2) * 0.01)
		start_pose_4[1] += (random.uniform(-1, 1) * 0.01)
		start_pose_4[0] -= 0.02 + (random.uniform(-1, 1) * 0.01)
		r1 = (random.uniform(-10, 10) * m.pi) / 180
		r2 = (random.uniform(-10, 10) * m.pi) / 180
		r3 = (random.uniform(-180, 180) * m.pi) / 180
		start_ori_4 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
		strawberry_4 = strawberry_cluster.StrawberryCluster(p, start_pose_4, start_ori_4, "models/clusters_urdf/strawberry_cluster.urdf")
	else:
		start_pose_4 = [False]
		start_ori_4 = [False]

	## load B - BEHIND strawberry:
	if current_cluster[4] == 1:
		start_pose_5 = []
		start_pose_5 = copy.deepcopy(strawberry_pose)
		stem_length = 0.15
		strawberry_radius = 0.015 + 0.06  # 0.075
		start_pose_5[2] = start_pose_5[2] + stem_length + strawberry_radius + (random.uniform(0, -2) * 0.01) 
		start_pose_5[1] -= 0.02 + (random.uniform(-1, 1) * 0.01)
		start_pose_5[0] += (random.uniform(-1, 1) * 0.01)
		r1 = (random.uniform(-10, 10) * m.pi) / 180
		r2 = (random.uniform(-10, 10) * m.pi) / 180
		r3 = (random.uniform(-180, 180) * m.pi) / 180
		start_ori_5 = p.getQuaternionFromEuler([(-m.pi / 2) + r1, r2, r3])
		strawberry_5 = strawberry_cluster.StrawberryCluster(p, start_pose_5, start_ori_5, "models/clusters_urdf/strawberry_cluster.urdf")
	else:
		start_pose_5 = [False]
		start_ori_5 = [False]

	strawberry_data_store_1 = [[start_pose_1, start_ori_1]]
	strawberry_data_store_2 = [[start_pose_2, start_ori_2]]
	strawberry_data_store_3 = [[start_pose_3, start_ori_3]]
	strawberry_data_store_4 = [[start_pose_4, start_ori_4]]
	strawberry_data_store_5 = [[start_pose_5, start_ori_5]]

	robot_data_store_position = []
	robot_data_store_velocity = []

	for i in range(0,400):
		# p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

		p.stepSimulation()
		time.sleep(timeStep)

	p.disconnect()
