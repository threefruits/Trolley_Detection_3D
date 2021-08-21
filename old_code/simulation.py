import rospy
from std_msgs.msg import Bool, Float64, Float32MultiArray
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point, Twist
from nav_msgs.msg import Path, Odometry
import numpy as np
# import tf
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt

from apply import detected_car_pos,detected_car
from car_infer import card_keypoints_init, card_infer

import cv2
from pnp import solve_pose

height = 640
weight = 640
dim = (weight, height)

model_path = r'./epoch_50.pth'
transform, model = card_keypoints_init(model_path)

def rpy2quaternion(roll, pitch, yaw):
    x=np.sin(pitch/2)*np.sin(yaw/2)*np.cos(roll/2)+np.cos(pitch/2)*np.cos(yaw/2)*np.sin(roll/2)
    y=np.sin(pitch/2)*np.cos(yaw/2)*np.cos(roll/2)+np.cos(pitch/2)*np.sin(yaw/2)*np.sin(roll/2)
    z=np.cos(pitch/2)*np.sin(yaw/2)*np.cos(roll/2)-np.sin(pitch/2)*np.cos(yaw/2)*np.sin(roll/2)
    w=np.cos(pitch/2)*np.cos(yaw/2)*np.cos(roll/2)-np.sin(pitch/2)*np.sin(yaw/2)*np.sin(roll/2)
    return x, y, z, w

class Sim():
	def __init__(self):
		self.L=0.75
		self.W=0.5
		self.H=1.0
		R = np.array([[0,1,0],[-1,0,0],[0,0,1]])
		
		point_in_original = []
		keypoint_in_original = []
		srcImg = cv2.imread("./trolley_detected/000002.jpg")
		
		srcImg_padding = cv2.copyMakeBorder(srcImg, 100, 100, 100, 100, cv2.BORDER_REPLICATE)
	
		h = srcImg_padding.shape[0]
		w = srcImg_padding.shape[1]
		w_ratio = w/640
		h_ratio = h/640
		srcImg_640 = cv2.resize(srcImg,dim)

		xyxy = detected_car_pos(srcImg_640)
		# xyxy = detected_car(srcImg_640)
		if(xyxy):
			point_in_original.append( ( (xyxy[0].cpu().numpy() - 320) * w_ratio + w/2).astype(int) )
			point_in_original.append( ( (xyxy[1].cpu().numpy() - 320) * h_ratio + h/2).astype(int) )
			point_in_original.append( ( (xyxy[2].cpu().numpy() - 320) * w_ratio + w/2).astype(int) )
			point_in_original.append( ( (xyxy[3].cpu().numpy() - 320) * h_ratio + h/2).astype(int) )
			h_d = (point_in_original[3] - point_in_original[1])
			w_d = (point_in_original[2] - point_in_original[0]) 
			h_crop = (point_in_original[3] - point_in_original[1]) * 1.56
			w_crop = (point_in_original[2] - point_in_original[0]) * 1.56
			if(0<point_in_original[0]-(w_d*0.28).astype(int) and 0<point_in_original[1]-(h_d*0.28).astype(int) and point_in_original[2]+(w_d*0.28).astype(int)<w and point_in_original[3]+(h_d*0.28).astype(int)<h):
				srcImg_crop = srcImg_padding[point_in_original[1]-(h_d*0.28).astype(int):point_in_original[3]+(h_d*0.28).astype(int),point_in_original[0]-(w_d*0.28).astype(int):point_in_original[2]+(w_d*0.28).astype(int)]
				# print(srcImg_padding.shape,srcImg_crop.shape,point_in_original)

				srcImg_crop = cv2.resize(srcImg_crop,(256,256))
				point_result = card_infer(transform, model, srcImg_crop)
				# print("point_result: ", point_result)
				for i in range(6):
					x = point_result[i][0] * (w_crop/256) + point_in_original[0]-(w_d*0.28).astype(int) -100
					y = point_result[i][1] * (h_crop/256) + point_in_original[1]-(h_d*0.28).astype(int) -100
					keypoint_in_original.append([x,y])
				print(keypoint_in_original)

		image_points = np.array(keypoint_in_original, dtype="double")

		# image_points = np.array([[692,962],[784,946],[740,894],[574,916],[741,566],[568,553]], dtype="double")
		T, euler_angles = solve_pose(image_points)
		print(T,euler_angles)
		self.T = R.dot(T)
		self.euler_angles = euler_angles

		self.car_marker = Marker()
		self.car_pub = rospy.Publisher('/car_maker', Marker, queue_size=10)

		self.sim_update = rospy.Timer(rospy.Duration(0.01), self.sim_update_fun)

	def sim_update_fun(self,event):
		self.car_marker.header.frame_id = "camera_base"
		self.car_marker.type = self.car_marker.CUBE
		self.car_marker.pose.position.x = self.T[0]
		self.car_marker.pose.position.y = self.T[1]
		self.car_marker.pose.position.z = self.T[2]+self.H/2
		# q =tf.transformations.quaternion_from_euler(self.euler_angles[0],self.euler_angles[1],self.euler_angles[2])
		q = rpy2quaternion(self.euler_angles[0],self.euler_angles[1],self.euler_angles[2])
		self.car_marker.pose.orientation.x = q[0]
		self.car_marker.pose.orientation.y = q[1]
		self.car_marker.pose.orientation.z = q[2]
		self.car_marker.pose.orientation.w = q[3]
		self.car_marker.scale.x = self.L
		self.car_marker.scale.y = self.W
		self.car_marker.scale.z = self.H
		self.car_marker.color.a= 1
		self.car_marker.color.r = 0
		self.car_marker.color.g = 1
		self.car_marker.color.b = 0

		# p = np.dot(self.intr, self.point1)/ 1.2
		# print(p)
		self.car_pub.publish(self.car_marker)

if __name__ == '__main__':
	rospy.init_node("sim")

	sim = Sim()
	rospy.spin()