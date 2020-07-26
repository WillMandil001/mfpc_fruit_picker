import time
import socket
import franka_panda
import pybullet_data
import strawberry_cluster
import franka_panda_new_EE
import numpy as np
import pybullet as p


with_ros = True

if with_ros == True:
	s = socket.socket()
	port = 12346
	s.bind(('127.0.0.11', port))
	s.listen(5)

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
timeStep=1./240.
p.setTimeStep(timeStep)
planeId = p.loadURDF("plane.urdf")

## Load franka:
start_pos = [0,0,0]
Joint_start_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02]
panda = franka_panda_new_EE.FrankaPanda(p, start_pos, timeStep, Joint_start_state)

## load strawberry cluster:
start_pose = [0.5, 0.5, 1.0]
start_ori = [0, 0, 1, 0]
cluster = strawberry_cluster.StrawberryCluster(p, start_pose, start_ori, "models/clusters_urdf/cluster_1.urdf")

# start_pose = [0.53, 0.53, 1.0]
# start_ori = [0, 0, 0, 1]
# cluster2 = strawberry_cluster.StrawberryCluster(p, start_pose, start_ori, "models/strawberry_cluster.urdf")

if with_ros == True:
	c, addr = s.accept()
	print('Got connection from', addr)

for i in range(0,10000):
	if with_ros == True:
		j1 = p.getJointState(panda.franka, 0)[0]
		j2 = p.getJointState(panda.franka, 1)[0]
		j3 = p.getJointState(panda.franka, 2)[0]
		j4 = p.getJointState(panda.franka, 3)[0]
		j5 = p.getJointState(panda.franka, 4)[0]
		j6 = p.getJointState(panda.franka, 5)[0]
		j7 = p.getJointState(panda.franka, 6)[0]

		# print([j1, j2, j3, j4, j5, j6, j7])
		joint_state = str([j1, j2, j3, j4, j5, j6, j7])
		c.send(joint_state.encode('utf-8')) 

	p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
	# if i == 500 and with_ros == True:
	# 	trajectory = panda.wait_for_trajectory()
	# 	print(trajectory)
	# 	trajectory = trajectory.replace("[", "")
	# 	trajectory = trajectory.replace("]", "")
	# 	steps = trajectory.split(")")
	# 	steps = steps[:-1]
	# 	for i in range(0, len(steps)):
	# 		steps[i] = steps[i].replace(", (", "")
	# 		steps[i] = steps[i].replace("(", "")
	# 		steps[i] = steps[i].split(", ")
	# 		print(steps[i])
	# 		for j in range(0, len(steps[i])):
	# 			steps[i][j] = float(steps[i][j])
	# elif i > 500 and with_ros == True:
	# 	try:
	# 		panda.step_from_ros(steps[i-501])
	# 	except:
	# 		pass
	# else:
	panda.step()

	p.stepSimulation()
	time.sleep(timeStep)

if with_ros == True:
	s.close()