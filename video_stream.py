import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import rospy
import cv2
import numpy as np
from TrolleyEstimator import TrolleyEstimator
import pyrealsense2 as rs


if __name__=="__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    rospy.init_node("trolley_estimator")
    trolley_estimator = TrolleyEstimator()

    while(True):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        # Convert images to numpy arrays
        # image = cv2.imread(color_frame.get_data())
        image = np.asanyarray(color_frame.get_data())
        print(len(image), len(image[0]), len(image[0][0]), )


        R, T, euler_angles = trolley_estimator.detect_3D(image)
        if len(euler_angles) != 0:
            trolley_estimator.publish_marker(T,  euler_angles)


        
