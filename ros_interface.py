#!/usr/bin/env python3

from sensor_msgs.msg import Image
import rospy
from TrolleyEstimator import TrolleyEstimator
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np


class Trolley_Estimator_ROS():
    def __init__(self):
        self.TrolleyEstimator = TrolleyEstimator()
        self.__sub_curr_state = rospy.Subscriber('/device_0/sensor_1/Color_0/image/data', Image, self._image_cb)
		# self.ob_pub = rospy.Publisher('/ob_draw', MarkerArray, queue_size=10)
    def _image_cb(self, msg):
        image = np.ndarray(shape=(msg.height,msg.width,3),dtype=np.uint8,buffer=msg.data)
        image = image[:,:,[2,1,0]]
        R, T, euler_angles = self.TrolleyEstimator.detect_3D(image)
        self.TrolleyEstimator.publish_marker(T, euler_angles)
        # print("1")
        # cv2.imshow("image",image)
        # cv2.waitKey(10)

if __name__=="__main__":
	rospy.init_node("trolley_estimator")
	phri_planner = Trolley_Estimator_ROS()
	rospy.spin()