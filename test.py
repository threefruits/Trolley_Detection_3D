import rospy
import cv2
from TrolleyEstimator import TrolleyEstimator


if __name__=="__main__":
    img_input_source = './trolley/2.png'
    image = cv2.imread(img_input_source)
    rospy.init_node("trolley_estimator")
    trolley_estimator = TrolleyEstimator()
    R, T, euler_angles = trolley_estimator.detect_3D(image)
    trolley_estimator.publish_marker(T, euler_angles)
    rospy.spin()