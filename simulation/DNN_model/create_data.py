import os
import csv
import math as m
import time
import copy
import socket
import random
import itertools
import numpy as np
import pybullet as p

def build_trajectory_data_for_DNN():
	# Robot pose data file:
	trajectory_full_robot_pose = []
	for i in range(0, 11000):
		trajectory_single = []
		with open(os.path.expanduser('data_set_001/robot_pos/data_set_' + str(i) + '_robot_data_store_position.csv'), newline='') as csvfile:
		    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		    for state in spamreader:
		        trajectory_point = []
		        for point in state[0].split(","):
		            trajectory_point.append(float(point))
		        trajectory_single.append(trajectory_point)
		trajectory_full_robot_pose.append(trajectory_single)

	# Robot velocity data file:
	trajectory_full_robot_velocity = []
	for i in range(0, 11000):
		trajectory_single = []
		with open(os.path.expanduser('data_set_001/robot_vel/data_set_' + str(i) + '_robot_data_store_velocity.csv'), newline='') as csvfile:
		    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		    for state in spamreader:
		        trajectory_point = []
		        for point in state[0].split(","):
		            trajectory_point.append(float(point))
		        trajectory_single.append(trajectory_point)
		trajectory_full_robot_velocity.append(trajectory_single)

	# straw_1 data file:
	trajectory_full_straw_1 = []
	for i in range(0, 11000):
		trajectory_single = []
		with open(os.path.expanduser('data_set_001/straw_1/data_set_' + str(i) + '_strawberry_data_store_1.csv'), newline='') as csvfile:
		    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		    for state in spamreader:
		        trajectory_point = []
		        for point in state[0].split(","):
		            trajectory_point.append(float(point))
		        trajectory_single.append(trajectory_point)
		trajectory_full_straw_1.append(trajectory_single)

	# straw_2 data file:
	trajectory_full_straw_2 = []
	for i in range(0, 11000):
		trajectory_single = []
		with open(os.path.expanduser('data_set_001/straw_2/data_set_' + str(i) + '_strawberry_data_store_2.csv'), newline='') as csvfile:
		    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		    for state in spamreader:
		        trajectory_point = []
		        for point in state[0].split(","):
		            trajectory_point.append(float(point))
		        trajectory_single.append(trajectory_point)
		trajectory_full_straw_2.append(trajectory_single)

	# straw_3 data file:
	trajectory_full_straw_3 = []
	for i in range(0, 11000):
		trajectory_single = []
		with open(os.path.expanduser('data_set_001/straw_3/data_set_' + str(i) + '_strawberry_data_store_3.csv'), newline='') as csvfile:
		    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		    for state in spamreader:
		        trajectory_point = []
		        for point in state[0].split(","):
		            trajectory_point.append(float(point))
		        trajectory_single.append(trajectory_point)
		trajectory_full_straw_3.append(trajectory_single)

	# straw_4 data file:
	trajectory_full_straw_4 = []
	for i in range(0, 11000):
		trajectory_single = []
		with open(os.path.expanduser('data_set_001/straw_4/data_set_' + str(i) + '_strawberry_data_store_4.csv'), newline='') as csvfile:
		    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		    for state in spamreader:
		        trajectory_point = []
		        for point in state[0].split(","):
		            trajectory_point.append(float(point))
		        trajectory_single.append(trajectory_point)
		trajectory_full_straw_4.append(trajectory_single)

	# straw_5 data file:
	trajectory_full_straw_5 = []
	for i in range(0, 11000):
		trajectory_single = []
		with open(os.path.expanduser('data_set_001/straw_5/data_set_' + str(i) + '_strawberry_data_store_5.csv'), newline='') as csvfile:
		    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		    for state in spamreader:
		        trajectory_point = []
		        for point in state[0].split(","):
		            trajectory_point.append(float(point))
		        trajectory_single.append(trajectory_point)
		trajectory_full_straw_5.append(trajectory_single)

	return trajectory_full_robot_pose, trajectory_full_robot_velocity, trajectory_full_straw_1, trajectory_full_straw_2, trajectory_full_straw_3, trajectory_full_straw_4, trajectory_full_straw_5