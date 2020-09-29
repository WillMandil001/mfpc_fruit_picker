import time
import math
import numpy as np


class StrawberryCluster(object):
	def __init__(self, p, position, orientation, file):
		self.pendulum = p.loadURDF(file, np.array(position), orientation, useFixedBase=True)
		self.j0 = StrawberryJoint(self.pendulum, p, 1, 2, 3)
		self.j1 = StrawberryJoint(self.pendulum, p, 4, 5, 6)

class StrawberryJoint(object):
	def __init__(self, pendulum, p, joint_id1, joint_id2, joint_id3):
		print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> here")
		self.pendulum = pendulum
		p.setJointMotorControl2(self.pendulum, joint_id1, p.VELOCITY_CONTROL, force=0.000)
		p.setJointMotorControl2(self.pendulum, joint_id2, p.VELOCITY_CONTROL, force=0.000)
		p.setJointMotorControl2(self.pendulum, joint_id3, p.VELOCITY_CONTROL, force=0.000)

		p.changeDynamics(self.pendulum, joint_id1, linearDamping=1, angularDamping=1, contactStiffness=1, contactDamping=0.5, 
						lateralFriction=1.0, spinningFriction=1.0, rollingFriction=1.0, jointDamping=0.0)
		p.changeDynamics(self.pendulum, joint_id2, linearDamping=1, angularDamping=1, contactStiffness=1, contactDamping=0.5, 
						lateralFriction=1.0, spinningFriction=1.0, rollingFriction=1.0, jointDamping=0.0)
		p.changeDynamics(self.pendulum, joint_id3, linearDamping=1, angularDamping=1, contactStiffness=1, contactDamping=0.5, 
						lateralFriction=1.0, spinningFriction=1.0, rollingFriction=1.0, jointDamping=0.0)

		self.previous_error_roll = 0.0
		self.previous_error_sum_roll = 0.0
		self.previous_error_pitch = 0.0
		self.previous_error_sum_pitch = 0.0
		self.previous_error_yaw = 0.0
		self.previous_error_sum_yaw = 0.0
		self.p = p

	def pd_controller_step(self, roll, pitch, yaw):
		kp = 0.1
		ki = 0.0
		kd = 0.01
		maxForce = 0

		# Proportional Values
		prop_roll = self.proportional_output(roll, 0.0, kp)
		prop_pitch = self.proportional_output(pitch, 0.0, kp)
		prop_yaw = self.proportional_output(yaw, 0.0, kp)

		# Derivative Values
		deriv_roll = self.derivative_output(roll, 0.0, self.previous_error_roll, kd)
		deriv_pitch = self.derivative_output(pitch, 0.0, self.previous_error_pitch, kd)
		deriv_yaw = self.derivative_output(yaw, 0.0, self.previous_error_yaw, kd)

		new_pos_roll = prop_roll + deriv_roll
		new_pos_pitch = prop_pitch + deriv_pitch
		new_pos_yaw = prop_yaw + deriv_yaw

		self.p.setJointMotorControl2(self.pendulum, 1, self.p.POSITION_CONTROL, new_pos_roll, force=5 * 240.)
		self.p.setJointMotorControl2(self.pendulum, 2, self.p.POSITION_CONTROL, new_pos_pitch, force=5 * 240.)
		self.p.setJointMotorControl2(self.pendulum, 3, self.p.POSITION_CONTROL, new_pos_yaw, force=5 * 240.)

		self.previous_error_roll = (0.0 - roll)
		self.previous_error_pitch = (0.0 - pitch)
		self.previous_error_yaw = (0.0 - yaw)

	def proportional_output(self, current_state, desired_state, kp):
		control_value = kp * (desired_state - current_state)
		return(control_value)

	def derivative_output(self, current_state, desired_state, previous_error, kd):
		error = (desired_state - current_state)
		error_diff = error - previous_error
		control_value = kd*error_diff
		return(control_value)