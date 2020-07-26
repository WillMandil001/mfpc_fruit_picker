import csv
import math as m
import time
import copy
import socket
import itertools
import franka_panda
import pybullet_data
import strawberry_cluster
import franka_panda_new_EE
import numpy as np
import pybullet as p


############# Format trajectories from ros to pybullet:
trajectories = []
# trajectories_cartesian_0_05_upsidedown_ee0_003
# trajectories_cartesian_0_05
with open('trajectories_cartesian_0_05_upsidedown_ee0_003.csv', newline='') as csvfile:
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

trajectories = trajectories[6550:6551]

for traj in trajectories:
	trajectory = traj
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

	# ## load strawberry cluster:
	# start_pose = []
	# start_pose = copy.deepcopy(strawberry_pose)
	# stem_length = 0.15
	# strawberry_radius = 0.015 + 0.06  # 0.075
	# start_pose[2] = start_pose[2] + stem_length + strawberry_radius
	# start_ori = p.getQuaternionFromEuler([(-m.pi / 2.2), 0, 0])
	# strawberry_1 = strawberry_cluster.StrawberryCluster(p, start_pose, start_ori, "models/clusters_urdf/strawberry_cluster.urdf")

	# ## load strawberry cluster:
	# start_pose = []
	# start_pose = copy.deepcopy(strawberry_pose)
	# stem_length = 0.15
	# strawberry_radius = 0.015 + 0.06  # 0.075
	# start_pose[2] = start_pose[2] + stem_length + strawberry_radius
	# start_ori = p.getQuaternionFromEuler([(-m.pi / 1.8), 0, 0])
	# strawberry_2 = strawberry_cluster.StrawberryCluster(p, start_pose, start_ori, "models/clusters_urdf/strawberry_cluster.urdf")

	## load strawberry cluster:
	start_pose = []
	start_pose = copy.deepcopy(strawberry_pose)
	stem_length = 0.15
	strawberry_radius = 0.015 + 0.06  # 0.075
	start_pose[2] = start_pose[2] + stem_length + strawberry_radius
	start_pose[0] += 0.02
	start_ori = p.getQuaternionFromEuler([(-m.pi / 2), 0, 0])
	strawberry_3 = strawberry_cluster.StrawberryCluster(p, start_pose, start_ori, "models/clusters_urdf/strawberry_cluster.urdf")

	# ## load strawberry cluster:
	# start_pose = []
	# start_pose = copy.deepcopy(strawberry_pose)
	# stem_length = 0.15
	# strawberry_radius = 0.015 + 0.06  # 0.075
	# start_pose[2] = start_pose[2] + stem_length + strawberry_radius
	# start_pose[0] -= 0.02
	# start_ori = p.getQuaternionFromEuler([(-m.pi / 2), 0, 0])
	# strawberry_4 = strawberry_cluster.StrawberryCluster(p, start_pose, start_ori, "models/clusters_urdf/strawberry_cluster.urdf")


	for i in range(0,1000):
		finished = False
		p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

		if i < 200:
			panda.step_from_ros(trajectory_simple[0])
		elif i > 200 and i-200 < len(trajectory_simple):
			panda.step_from_ros(trajectory_simple[i-200])
		else:
			panda.step_from_ros(trajectory_simple[-1])

		p.stepSimulation()
		time.sleep(timeStep)

		# # Strawberry2 PID controller:
		# cluster_r_state = p.getJointState(strawberry_1.pendulum, 1)[0]
		# cluster_p_state = p.getJointState(strawberry_1.pendulum, 2)[0]
		# cluster_y_state = p.getJointState(strawberry_1.pendulum, 3)[0]
		# strawberry_1.pd_controller_step(cluster_r_state, cluster_p_state, cluster_y_state)

		# # Strawberry2 PID controller:
		# cluster_r_state = p.getJointState(strawberry_2.pendulum, 1)[0]
		# cluster_p_state = p.getJointState(strawberry_2.pendulum, 2)[0]
		# cluster_y_state = p.getJointState(strawberry_2.pendulum, 3)[0]
		# strawberry_2.pd_controller_step(cluster_r_state, cluster_p_state, cluster_y_state)

		# Strawberry3 PID controller:
		cluster_r_state = p.getJointState(strawberry_3.pendulum, 1)[0]
		cluster_p_state = p.getJointState(strawberry_3.pendulum, 2)[0]
		cluster_y_state = p.getJointState(strawberry_3.pendulum, 3)[0]
		strawberry_3.pd_controller_step(cluster_r_state, cluster_p_state, cluster_y_state)

		# # Strawberry4 PID controller:
		# cluster_r_state = p.getJointState(strawberry_4.pendulum, 1)[0]
		# cluster_p_state = p.getJointState(strawberry_4.pendulum, 2)[0]
		# cluster_y_state = p.getJointState(strawberry_4.pendulum, 3)[0]
		# strawberry_4.pd_controller_step(cluster_r_state, cluster_p_state, cluster_y_state)

	p.disconnect()
	break
