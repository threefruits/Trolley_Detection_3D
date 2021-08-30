#!/usr/bin/env python3

from sensor_msgs.msg import Image
import rospy
from TrolleyEstimator import TrolleyEstimator
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import time
from std_msgs.msg import Bool, Float64, Float32MultiArray

class Trolley_Estimator_ROS():
    def __init__(self):
        self.TrolleyEstimator = TrolleyEstimator()
        self.curr_state = np.zeros(3)
        self.__sub_img = rospy.Subscriber('/camera/color/image_raw', Image, self.__image_cb, queue_size=10)
        self.__sub_curr_state = rospy.Subscriber('/curr_state', Float32MultiArray, self.__curr_pose_cb, queue_size=10)
        self.__pub_target = rospy.Publisher('/target', Float32MultiArray, queue_size=10)
        self.__timer_publish_target = rospy.Timer(rospy.Duration(0.01), self.state_filter)
        self.last_target = np.zeros(3)
        self.target = np.zeros(3)
        self.observation = np.zeros(3)
        self.is_obser = False
        # self.__sub_curr_state = rospy.Subscriber('/rgb/image_raw', Image, self._image_cb)
        # self.__sub_curr_state = rospy.Subscriber('/device_0/sensor_1/Color_0/image/data', Image, self._image_cb)
        

		# self.ob_pub = rospy.Publisher('/ob_draw', MarkerArray, queue_size=10)

    def __curr_pose_cb(self, data):
        for i in range(3):
            self.curr_state[i] = data.data[i]

    def state_filter(self): 
        target_point = Float32MultiArray()
        alpha = 0.3
        if(self.is_obser):
            self.target = alpha * self.last_target + (1-alpha) * self.observation
            self.is_obser = False
        else:
            self.target = self.last_target
        self.last_target = self.target
        for i in range(3): 
            target_point.data.append(self.target[i])
        self.__pub_target(target_point)
        

    def __image_cb(self, msg):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        # print((msg.height,msg.width,3))
        image = image[:,:,[2,1,0]]
        # cv2.imshow("image",image)
        # cv2.waitKey(10)
        R, T, euler_angles = self.TrolleyEstimator.detect_3D(image)
        if(len(T)):
            x = self.curr_state[0] + T[0]*np.cos(self.curr_state[2]) - T[1]*np.sin(self.curr_state[2])
            y = self.curr_state[1] + T[0]*np.sin(self.curr_state[2]) + T[1]*np.cos(self.curr_state[2])
            yaw = self.curr_state[2] + euler_angles[2]
            self.observation = np.array(x,y,yaw)
            self.is_obser = True

        # print(T)
        # if(len(T)):
        #     self.TrolleyEstimator.publish_marker_simple(, euler_angles)
        
            # print("1")
            # cv2.imshow("image",image)
            # cv2.waitKey(10)

if __name__=="__main__":
	rospy.init_node("trolley_estimator")
	phri_planner = Trolley_Estimator_ROS()
	rospy.spin()