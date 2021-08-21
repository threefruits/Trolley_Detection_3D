import cv2
import numpy as np
import math

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def solve_pose( image_points ):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        fx = 972.195
        fy = 972.023
        cx = 1023.77
        cy = 778.886
        R_ext = np.array([[1,0,0],[0,0,1],[0,-1,0]])

        camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        dist_coefs = np.array([[0.4808, -2.6471,-0.000164,-0.0001218,1.5565,0.361842,-2.47278,1.48308]]).T
        image_points = image_points
        model_points = np.array([[0.13662,0.39554,0.1],[-0.13662,0.39554,0.1],[-0.2535,-0.39554,0.1],[0.2535,-0.39554,0.1],[-0.2535,-0.39554,1],[0.2535,-0.39554,1]])
        (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coefs)
        R, _ = cv2.Rodrigues(rotation_vector)

        R_ = R_ext.dot(R)
        T = R_ext.dot(translation_vector)

        euler_angles = rot2eul(R_)
        
        return T, euler_angles

# image_points = np.array([[1367,1033],[1270,1021],[1113,1079],[1338,1116]], dtype="double")
# print(solve_pose(image_points))