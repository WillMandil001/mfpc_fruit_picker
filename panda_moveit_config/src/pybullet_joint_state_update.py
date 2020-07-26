#!/usr/bin/env python
import rospy
import socket
from std_msgs.msg import String
from sensor_msgs.msg import JointState

def talker(s):
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rospy.init_node('joint_state_publisher', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    s.connect(('127.0.0.11', port))
    previous_state = []
    while not rospy.is_shutdown():
        try:
            hello_str = str(s.recv(1024).decode('utf-8'))
            pybullet_robot_state = hello_str.split("]")
            pybullet_robot_state = str(pybullet_robot_state[0]).replace("[", "").split(", ")
            for i in range(0, len(pybullet_robot_state)):
                pybullet_robot_state[i] = float(pybullet_robot_state[i]) 

            pybullet_robot_state.append(0.0)
            pybullet_robot_state.append(0.0)

            state = JointState()
            state.header.stamp = rospy.Time.now()
            # state.header.frame_id = "panda_link0"
            state.name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"]
            state.position = pybullet_robot_state
            previous_state = state

            rospy.loginfo(state)
            pub.publish(state)
            rate.sleep()
        except:
            print">>> No data recieved - publishning last known pose"
            previous_state.header.stamp = rospy.Time.now()
            rospy.loginfo(previous_state)
            pub.publish(previous_state)
            rate.sleep()
    s.close()

if __name__ == '__main__':
    s = socket.socket()
    port = 12346
    try:
        talker(s)
    except rospy.ROSInterruptException:
        s.close()
        pass