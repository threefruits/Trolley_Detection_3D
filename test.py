import rospy
import cv2
from TrolleyEstimator import TrolleyEstimator
import time

if __name__=="__main__":
    img_input_source = './trolley/8.png'
    image = cv2.imread(img_input_source)
    rospy.init_node("trolley_estimator")
    trolley_estimator = TrolleyEstimator()
    while (not rospy.is_shutdown()):
        
        R, T, euler_angles = trolley_estimator.detect_3D(image)
        if(len(T)):
            trolley_estimator.publish_marker(T, euler_angles)
        
        
    rospy.spin()