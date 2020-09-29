import os
import time
import math
import socket
import numpy as np
import pybullet_data
from socket import error as SocketError

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs  # upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs  # joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs  # restposes for null space
jointPositions=[0.98, 0.458, 0.31, -1.5, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class FrankaPanda(object):
	def __init__(self, p, offset, time_step, start_state):
		self.s = socket.socket()
		self.port = 22342
		self.time_step = time_step
		self.offset = np.array(offset)
		self.p = p
		self.t = 0.
		self.franka = self.p.loadURDF("franka_panda/panda.urdf", np.array([0,0,0]), [0,0,0,1], useFixedBase=True)  # , flags=flags # flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
		jointPositions=start_state
		# jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
		print(">>>>><><>>>>>>>>", str(self.p.getNumJoints(self.franka)))
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

	def step(self):
		t = self.t
		self.t += self.time_step
		pos = [0.4, 0.4, 1]
		# orn = self.p.getQuaternionFromEuler([math.pi/2.,0.,0.])
		orn = [1,0,0,0]
		jointPoses = self.p.calculateInverseKinematics(self.franka,pandaEndEffectorIndex, pos, orn, ll, ul,
			jr, rp, maxNumIterations=5)
		for i in range(pandaNumDofs):
				self.p.setJointMotorControl2(self.franka, i, self.p.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
		pass

	def wait_for_trajectory(self):
		self.s.connect(('127.0.0.2', self.port))
		trajectory = str(self.s.recv(4096).decode('utf-8'))
		self.s.close()
		return trajectory

	def step_from_ros(self, trajectory):
		try:
			print(trajectory, "\n")
			for i in range(pandaNumDofs):
					self.p.setJointMotorControl2(self.franka, i, self.p.POSITION_CONTROL, trajectory[i],force=5 * 240.)
		except:
			print("performed total trajectory")
