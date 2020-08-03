#!/usr/bin/env python
import csv
import sys
import copy
import rospy
import socket
import random
import math as m
import numpy as np
from math import pi
import moveit_msgs.msg
import moveit_commander
import geometry_msgs.msg
from std_msgs.msg import String
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from moveit_commander.conversions import pose_to_list


def all_close(goal, actual, tolerance):
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True


class MoveGroupPythonIntefaceTutorial(object):
    def __init__(self):
        super(MoveGroupPythonIntefaceTutorial, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "panda_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)
        planning_frame = move_group.get_planning_frame()
        eef_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()

        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names


    def go_to_joint_state(self):
        move_group = self.move_group
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -pi/4
        joint_goal[2] = 0
        joint_goal[3] = -pi/2
        joint_goal[4] = 0
        joint_goal[5] = pi/3
        joint_goal[6] = 0

        plan = move_group.plan(joint_goal)
        return plan

    def plan_cartesian_path(self, pose, orientation):
        move_group = self.move_group
        waypoints = []

        # start with the current pose
        waypoints.append(move_group.get_current_pose().pose)

        # first orient gripper and move forward (+x)
        wpose = geometry_msgs.msg.Pose()
        wpose.orientation.x = orientation[0]
        wpose.orientation.y = orientation[1]
        wpose.orientation.z = orientation[2]
        wpose.orientation.w = orientation[3]
        wpose.position.x = pose[0]
        wpose.position.y = pose[1]
        wpose.position.z = pose[2]
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = move_group.compute_cartesian_path(
                                     waypoints,   # waypoints to follow
                                     0.003,        # eef_step
                                     0.0)         # jump_threshold

        return plan

    def plan_to__pose(self, final_pose_position, final_pose_orientation):
        move_group = self.move_group

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = final_pose_orientation[0]
        pose_goal.orientation.y = final_pose_orientation[1]
        pose_goal.orientation.z = final_pose_orientation[2]
        pose_goal.orientation.w = final_pose_orientation[3]

        pose_goal.position.x = final_pose_position[0]
        pose_goal.position.y = final_pose_position[1]
        pose_goal.position.z = final_pose_position[2]

        plan = move_group.plan(pose_goal)
        return plan

    def go_to__pose(self, final_pose_position, final_pose_orientation):
        move_group = self.move_group

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = final_pose_orientation[0]
        pose_goal.orientation.y = final_pose_orientation[1]
        pose_goal.orientation.z = final_pose_orientation[2]
        pose_goal.orientation.w = final_pose_orientation[3]

        pose_goal.position.x = final_pose_position[0]
        pose_goal.position.y = final_pose_position[1]
        pose_goal.position.z = final_pose_position[2]

        plan = move_group.go(pose_goal)


def filter_plan(plan):
    simple_trajectory = []
    trajectory = plan.joint_trajectory.points
    for i in range(0, len(trajectory)):
        simple_trajectory.append(trajectory[i].positions)

    return simple_trajectory

def send_trajectory(s, plan, port):
    s.bind(('127.0.0.2', port))
    s.listen(5)
    c, addr = s.accept()
    print('Got connection from', addr)
    message = str(plan)
    c.send(message.encode('utf-8')) 

def generate_data_set_movements_square(strawberry_pose, robot):
    trajectories = []
    ## ASSUMPTION:
    # 1. The mushroom is invariant in X axis.
    # 2. Constant velocity.

    # 1. Straight line - Variance in x y and z start and finish state
    # final_pose_orientation = [0, 1, 0, 0]
    # for y in range(-5, 5):
    #     for z in range(-5, 5):
    #         start_pose_position = [strawberry_pose[0] - 0.2,
    #                             (strawberry_pose[1] + (y * 0.01)),
    #                             (strawberry_pose[2] + (z * 0.01))] 
    #         robot.go_to__pose(start_pose_position, final_pose_orientation)
    #         final_pose_position = start_pose_position
    #         final_pose_position[0] = strawberry_pose[0] + 0.2
    #         trajectory = robot.plan_cartesian_path(final_pose_position, final_pose_orientation)
    #         simple_trajectory = filter_plan(trajectory)
    #         trajectories.append(simple_trajectory)
    #         break

    # 2. Straight line - Variant angle of pass through
    final_pose_orientation = [0, 0, 1, 0]
    start_poses = []
    final_poses = []
    for y in range(-7, 7):
        for z in range(-7, 7):
            start_poses.append([strawberry_pose[0] - 0.2 + (y * 0.01),
                                (strawberry_pose[1] + (y * 0.01)),
                                (strawberry_pose[2] + (z * 0.01))])

            final_poses.append([strawberry_pose[0] + 0.2,
                                (strawberry_pose[1] + (y * 0.01)),
                                (strawberry_pose[2] + (z * 0.01))])
    i = 0
    for start_state in start_poses:
        robot.go_to__pose(start_state, final_pose_orientation)
        for final_pose in final_poses:    
            trajectory = robot.plan_cartesian_path(final_pose, final_pose_orientation)
            simple_trajectory = filter_plan(trajectory)
            trajectories.append(simple_trajectory)
            print("this far through = ", i)
            i += 1
    return trajectories

def generate_data_circle(strawberry_pose, robot):
    # sample
    while(1):
        size = 800
        R = 2.5
        phi = np.random.uniform(low = (0.75 + (np.pi/2)), high=(np.pi - 0.75 + (np.pi/2)), size=size)
        r = np.sqrt(np.random.random(size=size)) * R

        # transform
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        remain_list = []
        for i in range(0, len(x) - 1):
            distance = m.sqrt(x[i]**2 + y[i]**2)
            if distance > 2.3:
                remain_list.append(i)

        x_copy_start = []
        y_copy_start = []

        x_copy_finish = []
        y_copy_finish = []
        print("HERERERERERERE >>> ", len(remain_list))

        for value in remain_list:
            x_copy_start.append(0.04 * x[value])
            y_copy_start.append(0.04 * y[value])

        # plot Behind
        for index, value in enumerate(x_copy_start):
            x_copy_finish.append(-1*x_copy_start[index])
            y_copy_finish.append(-1*y_copy_start[index])

        z_start = []
        z_finish = []
        for i in range(0, len(x_copy_start)):
            z_start.append(random.uniform(-0.04, 0.04))
            z_finish.append(random.uniform(-0.04, 0.04))


        if len(remain_list) == 100:
            # strawberry_center_x = 0.5
            # strawberry_center_y = 0
            # strawberry_center_z = 0.5
            # for index, value in enumerate(x_copy_start):
            #     z_start[index] = z_start[index] + strawberry_center_z 
            #     z_finish[index] = z_finish[index] + strawberry_center_z 

            #     x_copy_start[index] = x_copy_start[index] + strawberry_center_x 
            #     y_copy_start[index] = y_copy_start[index] + strawberry_center_y 

            #     x_copy_finish[index] = x_copy_finish[index] + strawberry_center_x 
            #     y_copy_finish[index] = y_copy_finish[index] + strawberry_center_y 

            fig = plt.figure()
            a = fig.add_subplot(111, projection='3d')
            a.scatter(x_copy_start, y_copy_start, z_start, color="r", marker='.')
            a.scatter(x_copy_finish, y_copy_finish, z_finish, color="b", marker='.')
            a.set_aspect('equal')
            plt.show()
            break


    # # 2. Straight line - Variant angle of pass through
    trajectories = []
    final_pose_orientation = [0, 0, 1, 0]
    start_poses = []
    final_poses = []
    for index, value in enumerate(x_copy_start):
            start_poses.append([strawberry_pose[0] + x_copy_start[index],
                                strawberry_pose[1] + y_copy_start[index],
                                strawberry_pose[2] + z_start[index]])

            final_poses.append([strawberry_pose[0] + x_copy_finish[index],
                                strawberry_pose[1] + y_copy_finish[index],
                                strawberry_pose[2] + z_finish[index]])
    i = 0
    for start_state in start_poses:
        robot.go_to__pose(start_state, final_pose_orientation)
        for final_pose in final_poses:    
            trajectory = robot.plan_cartesian_path(final_pose, final_pose_orientation)
            simple_trajectory = filter_plan(trajectory)
            trajectories.append(simple_trajectory)
            print("this far through = ", i)
            i += 1
    return trajectories


def main():
    # s = socket.socket()
    # port = 22342
    try:
        robot = MoveGroupPythonIntefaceTutorial()
        strawberry_state = [0.5, 0, 0.5]
        trajectories = generate_data_circle(strawberry_state, robot)
        # trajectories = generate_data_set_movements_square(strawberry_state, robot)
        trajectories.insert(0, strawberry_state)
        with open("Robotics/mfpc_fruit_picking/src/panda_moveit_config/src/trajectories_cartesian_circle.csv","w") as f:
            wr = csv.writer(f)
            wr.writerows(trajectories)

        # plan = robot.go_to_joint_state()
        # print plan
        # plan_simple = filter_plan(plan)
        # send_trajectory(s, plan_simple, port)
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return
    # s.close()

if __name__ == '__main__':
    main()