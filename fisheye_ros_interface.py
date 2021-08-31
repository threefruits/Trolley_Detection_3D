#!/usr/bin/env python3

from sensor_msgs.msg import Image
import rospy
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import time
from std_msgs.msg import Bool, Float64, Float32MultiArray
from fisheye_TrolleyEstimator import TrolleyEstimator

class Trolley_Estimator_Close_ROS():
    def __init__(self):
        self.TrolleyEstimator = TrolleyEstimator()
        self.__sub_img = rospy.Subscriber('/camera/fisheye2/image_raw', Image, self.__image_cb, queue_size=10)
    
    def __image_cb(self, msg):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        out_img = image.copy()
        DIM = (800,848)
        K = np.array([[286.5763854980469, 0.0, 424.664306640625], [0.0, 286.6034851074219, 389.56170654296875], [0.0, 0.0, 1.0]])
        D = np.array([-0.007487522903829813, 0.0455954484641552, -0.04332948848605156, 0.007895871065557003])
        Knew = K.copy()
        scale = 0.5
        Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
        print(Knew)
        map1,map2 = cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),Knew,DIM, cv2.CV_16SC2)
        out_img = cv2.remap(image,map1,map2,interpolation = cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # cv2.fisheye.undistortImage(image, out_img, K, D, K, DIM)
        R, T, euler_angles = self.TrolleyEstimator.detect_3D(image)
        if(len(T)):
            self.TrolleyEstimator.publish_marker_simple(T[0],T[1],euler_angles[2])
        # print(image.shape)
        # image = image[:,:]
        # cv2.imshow("image",out_img)
        # cv2.imshow("image2",image)
        # cv2.waitKey(10)

if __name__=="__main__":
	rospy.init_node("trolley_estimator_close")
	phri_planner = Trolley_Estimator_Close_ROS()
	rospy.spin()
