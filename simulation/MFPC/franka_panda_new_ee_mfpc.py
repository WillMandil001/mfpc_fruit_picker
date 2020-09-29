import os
import time
import math
import random
import numpy as np
import itertools
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


import cv2
import math
import random
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(1000)
from sklearn import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input

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



import pybullet_data

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 7 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs  # upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs  # joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs  # restposes for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]
rp = jointPositions

class FrankaPanda(object):
	def __init__(self, p, offset, time_step, start_state, trajectory_length):
		self.model = load_model("/home/will/Robotics/mfpc_fruit_picking/src/simulation/MFPC/RNN_models/custom_model_simple_no_scaleNN_ni_nv_t6_011.h5")
		self.time_step = time_step
		self.offset = np.array(offset)
		self.p = p
		self.t = 0.
		self.trajectory_length = trajectory_length
		self.franka = self.p.loadURDF("/models/panda_no_gripper/panda.urdf", np.array([0,0,0]), [0,0,0,1], useFixedBase=True)
		# self.franka = self.p.loadURDF("franka_panda/panda.urdf", np.array([0,0,0]), [0,0,0,1], useFixedBase=True)  # , flags=flags # flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
		jointPositions=start_state
		# jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
		index = 0
		for j in range(self.p.getNumJoints(self.franka)):
			self.p.changeDynamics(self.franka, j, linearDamping=0, angularDamping=0)
			info = self.p.getJointInfo(self.franka, j)
			jointName = info[1]
			jointType = info[2]
			if (jointType == self.p.JOINT_PRISMATIC):
				self.p.resetJointState(self.franka, j, jointPositions[index]) 
				index=index+1
			if (jointType == self.p.JOINT_REVOLUTE):
				self.p.resetJointState(self.franka, j, jointPositions[index]) 
				index=index+1

	def reset(self):
		pass

	def step(self, start_state_x_y_z):
		t = self.t
		self.t += self.time_step
		pos = [start_state_x_y_z[0], start_state_x_y_z[1], start_state_x_y_z[2]]
		# orn = self.p.getQuaternionFromEuler([math.pi/2.,0.,0.])
		orn = [0, 0, 1, 0]
		jointPoses = self.p.calculateInverseKinematics(self.franka, pandaEndEffectorIndex, pos, orn)
		for i in range(pandaNumDofs):
				self.p.setJointMotorControl2(self.franka, i, self.p.POSITION_CONTROL, jointPoses[i],force=5 * 120.)
		return self.p.getLinkState(self.franka, pandaEndEffectorIndex)[0]

	def step_joint(self, start_state_x_y_z):
		for i in range(pandaNumDofs):
				self.p.setJointMotorControl2(self.franka, i, self.p.POSITION_CONTROL, start_state_x_y_z[i],force=5 * 120.)

	def step_MFPC(self, priority, ee_pose, origional_trajectory, current_strawberry_states, goal_strawberry_states):
		# Generate list of trajectorys 
		start_point = origional_trajectory[0]
		trajectory_pos_list_joint, trajectory_list_task, end_points = self.create_full_trajectory_list(ee_pose, 1000)

		# Run trajectory list on CNN & RNN to output list of predicted strawberry states:
		predicted_env_state_list = self.estimate_env_state(trajectory_pos_list_joint, current_strawberry_states)

		## plotting the predicted state:
		print(predicted_env_state_list)
		print(predicted_env_state_list[:][0])

		fig = plt.figure() 
		ax = plt.axes(projection ="3d") 		  
		# Creating plot 
		for state, ee in zip(predicted_env_state_list, end_points):
			ax.scatter3D(state[0], state[1], state[2], color = "green"); 
			ax.scatter3D(ee[0], ee[1], ee[2], color = "orange"); 
		ax.scatter3D(current_strawberry_states[0][0], current_strawberry_states[0][1], current_strawberry_states[0][2], color = "red"); 

		ax.scatter3D(goal_strawberry_states[2][0], goal_strawberry_states[2][1], goal_strawberry_states[2][2], color = "blue"); 
		ax.scatter3D(ee_pose[0], ee_pose[1], ee_pose[2], color = "black"); 

		# ax.scatter3D(current_strawberry_states[3], current_strawberry_states[4], current_strawberry_states[5], color = "red"); 
		# ax.scatter3D(current_strawberry_states[6], current_strawberry_states[7], current_strawberry_states[8], color = "red"); 

		plt.title("simple 3D scatter plot") 
		# show plot
		plt.show()

		# choose trajectory with best predicted environment state:
		final_trajectory_index = self.best_state(predicted_env_state_list, goal_strawberry_states, priority)
		print("current state >> ", current_strawberry_states[2])

		return trajectory_list_task[final_trajectory_index], trajectory_pos_list_joint[final_trajectory_index]

	def find_nearest(self, array, value):
	    return array[(np.abs(np.asarray(array) - value)).argmin()]

	def best_state(self, predicted_env_state_list, goal_strawberry_states, priority):
		print("goal_strawberry_states >> ", goal_strawberry_states[2])
		distance_list = []
		if priority[0] == 1:
			for state in predicted_env_state_list:
				distance_list.append(np.linalg.norm(state[6:9][1] - goal_strawberry_states[2][1]))
			min_index = distance_list.index((self.find_nearest(np.asarray(distance_list), 0.0)))
		elif priority[1] == 1:
			for state in predicted_env_state_list:
				distance_list.append(np.linalg.norm(state[3:6][1] - goal_strawberry_states[1][1]))
			min_index = distance_list.index((self.find_nearest(np.asarray(distance_list), 0.0)))
		print("Best state >> ", predicted_env_state_list[min_index][6:9])

		return min_index

		# 		distance_list = []
		# if priority[0] == 1:
		# 	for state in predicted_env_state_list:
		# 		print(state)
		# 		distance_list.append(math.sqrt(((state[0] - goal_strawberry_states[0])**2) + ((state[1] - goal_strawberry_states[1])**2) + ((state[2] - goal_strawberry_states[2])**2)))
		# 	min_index = np.argmin(np.asarray(distance_list))
		# elif priority[1] == 1:
		# 	for state in predicted_env_state_list:
		# 		distance_list.append(math.sqrt(((state[0] - goal_strawberry_states[0])**2) + ((state[1] - goal_strawberry_states[1])**2) + ((state[2] - goal_strawberry_states[2])**2)))
		# 	min_index = np.argmin(np.asarray(distance_list))

		# return

	def estimate_env_state(self, trajectory_pos_list_joint, current_strawberry_states):
		current_strawberry_states[0] = current_strawberry_states[0][0] + current_strawberry_states[0][1]
		current_strawberry_states[1] = current_strawberry_states[1][0] + current_strawberry_states[1][1]
		current_strawberry_states[2] = current_strawberry_states[2][0] + current_strawberry_states[2][1]
		current_strawberry_states = current_strawberry_states[0] + current_strawberry_states[1] + current_strawberry_states[2]

		formatted_trajectories = self.format_trajectories(trajectory_pos_list_joint)
		current_strawberry_states_list = list(itertools.repeat(current_strawberry_states, len(formatted_trajectories)))

		#################################### STANDARDISE THE ROBOT STATES!!   #################################### 
		scaler = preprocessing.StandardScaler()
		myScaler = scaler.fit(formatted_trajectories)
		formatted_trajectories = myScaler.transform(formatted_trajectories)

		predicted_states_list = self.model.predict([np.asarray(current_strawberry_states_list), np.asarray(formatted_trajectories)])

		return predicted_states_list

	def format_trajectories(self, trajectory_list):
		for i in range(0, len(trajectory_list)):
			trajectory_list[i] = list(trajectory_list[i][0] + trajectory_list[i][1] + trajectory_list[i][2] + trajectory_list[i][3] + trajectory_list[i][4] + trajectory_list[i][5] + trajectory_list[i][6])
		return trajectory_list

	def create_full_trajectory_list(self, start_point, number_of_trajectories):
		end_points = self.create_final_points(number_of_trajectories, start_point)

		trajectory_list_joint = []
		trajectory_list_task = []
		for goal_point in end_points:
			trajectory_joint, trajectory_task = self.create_trajectory(start_point, goal_point)
			trajectory_list_joint.append(trajectory_joint)
			trajectory_list_task.append(trajectory_task)
		return trajectory_list_joint, trajectory_list_task, end_points

	def create_final_points(self, npoints, start_point):
		final_point = []
		r = 0.05
		x__ = []
		y__ = []
		z__ = []
		for i in range(0, npoints):
			# r = random.uniform(0.005, 0.05)
			theta = random.uniform(0, 2*math.pi)
			phi = random.uniform(0, math.pi)
			x = r * math.cos(theta) * math.sin(phi)
			y = r * math.sin(theta) * math.sin(phi)
			z = r * math.cos(phi)
			# x__.append((start_point[0] + (r * math.cos(theta) * math.sin(phi))))
			# y__.append((start_point[1] + (r * math.sin(theta) * math.sin(phi))))
			# z__.append((start_point[2] + (r * math.cos(phi))))
			final_point.append([(start_point[0] + x), (start_point[1] + y), (start_point[2] + z)])
		# fig = plt.figure() 
		# ax = plt.axes(projection ="3d") 		  
		# # Creating plot 
		# ax.scatter3D(x__, y__, z__, color = "green"); 
		# plt.title("simple 3D scatter plot") 
  # 		# show plot
		# plt.show()
		return(final_point)

	def create_trajectory(self, start_point, end_point):
		traj_x = list(np.linspace(start_point[0], end_point[0], num=self.trajectory_length, retstep=True))[0]
		traj_y = list(np.linspace(start_point[1], end_point[1], num=self.trajectory_length, retstep=True))[0]
		traj_z = list(np.linspace(start_point[2], end_point[2], num=self.trajectory_length, retstep=True))[0]

		# Convert to joint state trajectory:
		trajectory_joint = []
		trajectory_task = []
		for i in range(0, self.trajectory_length):
			trajectory_joint.append(self.p.calculateInverseKinematics(self.franka, pandaEndEffectorIndex, [traj_x[i], traj_y[i], traj_z[i]], [0, 0, 1, 0]))
			trajectory_task.append([traj_x[i], traj_y[i], traj_z[i]])
		return trajectory_joint, trajectory_task

	def step_from_ros(self, trajectory):
		try:
			# print(trajectory)
			for i in range(pandaNumDofs):
					self.p.setJointMotorControl2(self.franka, i, self.p.POSITION_CONTROL, trajectory[i],force=5 * 120.)
		except:
			print("performed total trajectory")
