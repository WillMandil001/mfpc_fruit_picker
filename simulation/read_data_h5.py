import os
import csv
import time
import copy
import h5py
import socket
import random
import itertools
import franka_panda
import pybullet_data
import strawberry_cluster
import franka_panda_new_EE
import math as m
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image


def read_hf5_images(hdf5_dir):
	with h5py.File(hdf5_dir + "/" + str(400) + '.h5', 'r') as hf:
	    data = hf['images'][:]
	return data

camera_1_rgb_path = "data_set_002/camera_1/rgb/sample_" + str(0)
camera_1_depth_path = "data_set_002/camera_1/depth/sample_" + str(0)

image_data = read_hf5_images(camera_1_rgb_path)
print(image_data[0].shape)
for i in range(0, len(image_data)):
	imgplot = plt.imshow(image_data[i])
	plt.show()
