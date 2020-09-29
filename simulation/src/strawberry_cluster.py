import time
import math
import numpy as np


class StrawberryCluster(object):
	def __init__(self, p, position, orientation, file):
		self.pendulum = p.loadURDF(file, np.array(position), orientation, useFixedBase=True)
		# self.pendulum = p.loadURDF("models/clusters_urdf/cluster_1_extras.xacro", np.array(position), orientation, useFixedBase=True)

		# p.resetJointState(pendulum, 0, 0.5)
		# p.setJointMotorControl2(pendulum, 0, controlMode=p.TORQUE_CONTROL, force=0.000)
		# p.resetJointState(pendulum, 1, 0.5)
		# p.setJointMotorControl2(self.pendulum, 1, controlMode=p.TORQUE_CONTROL, force=0.000)
		p.setJointMotorControl2(self.pendulum, 1, p.VELOCITY_CONTROL, force=0.000)
		p.setJointMotorControl2(self.pendulum, 2, p.VELOCITY_CONTROL, force=0.000)
		p.setJointMotorControl2(self.pendulum, 3, p.VELOCITY_CONTROL, force=0.000)

		p.changeDynamics(self.pendulum, 1, linearDamping=1, angularDamping=1, contactStiffness=1, contactDamping=0.5, 
						lateralFriction=1.0, spinningFriction=1.0, rollingFriction=1.0, jointDamping=0.0)
		p.changeDynamics(self.pendulum, 2, linearDamping=1, angularDamping=1, contactStiffness=1, contactDamping=0.5, 
						lateralFriction=1.0, spinningFriction=1.0, rollingFriction=1.0, jointDamping=0.0)
		p.changeDynamics(self.pendulum, 3, linearDamping=1, angularDamping=1, contactStiffness=1, contactDamping=0.5, 
						lateralFriction=1.0, spinningFriction=1.0, rollingFriction=1.0, jointDamping=0.0)

		self.previous_error_roll = 0.0
		self.previous_error_sum_roll = 0.0
		self.previous_error_pitch = 0.0
		self.previous_error_sum_pitch = 0.0
		self.previous_error_yaw = 0.0
		self.previous_error_sum_yaw = 0.0
		self.p = p

	def pd_controller_step(self, roll, pitch, yaw):
		kp = 0.01
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

		# self.previous_error_sum_roll += self.previous_error_roll
		# self.previous_error_sum_pitch += self.previous_error_pitch
		# self.previous_error_sum_yaw += self.previous_error_yaw


	def proportional_output(self, current_state, desired_state, kp):
		control_value = kp * (desired_state - current_state)
		return(control_value)

	def derivative_output(self, current_state, desired_state, previous_error, kd):
		error = (desired_state - current_state)
		error_diff = error - previous_error
		control_value = kd*error_diff
		return(control_value)

	# def integral_output(self, x1, x2, t1, t2):
	# 	control_value = (self.integral - self.mu*(t2 - t1) * (x1 - self.set_point + x2 - self.set_point)/2.0)
	# 	self.integral = control_value
	# 	return(control_value)
