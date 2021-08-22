
import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import rospy
import time
import math
import os

import cv2
import torch
import numpy as np
import torchvision
from torchvision import transforms
from visualization_msgs.msg import Marker

from models.experimental import attempt_load
from models.Stack import HourglassNet,Bottleneck
from utils.general import non_max_suppression



class TrolleyEstimator():
    """
    Function: Trolley Estimator

    Steps to Caculate the location:
             .  Crop the boundingbox  of trolley by  yolov5s.
             .  Get the 6  keypoints  in cropped  image  model  using stacked hourglass network. 
             .  Solve the R and T

    use detect_3D() to compute 3D pose
    """
    def __init__(self):
        self.L=0.75
        self.W=0.5
        self.H=1.0
        self.height = 640
        self. weight = 640
        self.dim = (self.weight, self. height)
        self.R = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        self.keypoint_model_path = r'./checkpoint/keypoint_ckpt.pth'
        self.yolo5_model_path=r'./checkpoint/yolo5_ckpt_500.pt'
        self.conf_thres=0.30
        self.iou_thres=0.45
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform, self.keypoint_model = self.trolley_keypoints_init()

    def publish_marker(self, T, euler_angles):
        """

        Publish maker in rviz

        """
        self.T = T
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
        q = self.rpy2quaternion(self.euler_angles[0],self.euler_angles[1],self.euler_angles[2])
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
    
    def detected_car_pos(self, img):
        """
        This is the interface of trolley pose detection.

        Input: 
            * RGB img after resize ........ numpy array (640,640,3) 
        
        Output:  
            * two points of bounding box (left-up and right-down) ......... (Tensor [4])
        
        """
        srcImg1 = img.copy()
        srcImg = srcImg1.transpose((2, 0, 1))[::-1]
        srcImg = np.ascontiguousarray(srcImg)
        img = torch.from_numpy(srcImg).to(self.device)
        img = img.float() 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        # print(img.shape)

        yolo_model = attempt_load(self.yolo5_model_path, map_location=self.device)
        pred = yolo_model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=1000)

        for i, det in enumerate(pred):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = det[:, :4].round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    return xyxy
                    
        return None

    def trolley_keypoints_init(self):
        """
        init keypoints detection model

        """
        transform = transforms.Compose([transforms.ToTensor()])
        model = HourglassNet(Bottleneck)
        model.load_state_dict(torch.load(self.keypoint_model_path))
        model.cuda()
        model.eval()
        return transform, model

    def trolley_infer(self, transform, model, test_img):
        """
        This is the interface of trolley pose detection.

        Input: 
            * RGB img after croped ........ numpy array (h_crop,w_crop,3) 
        
        Output:  
            * keypoints in the croped image ......... (list: len = 6, with [x,y])
        
        """
        t2 = time.time()
        test_img = cv2.resize(test_img, (256, 256))
        draw_img = test_img.copy()
        test_img = transform(test_img)
        test_img = torch.unsqueeze(test_img, 0).cuda()
        out = model(test_img)
        out = torch.sigmoid(out[-1])
        out = out.cpu().detach().numpy()
        # print("infer time: ", time.time() - t2)
        # show_six_feature(out, figure_num=1, img=test_img)
        # plt.figure(2)
        # plt.subplot(1, 2, 1)
        show_out = np.max(out[0], axis=0)
        # plt.imshow(show_out)
        # point_result_dict = {}
        point_result_dict = []
        for i in range(out.shape[1]):
            one_layer = out[0][i]
            max_point_value = np.max(one_layer)
            if max_point_value > 0:
                coor = np.where(one_layer == max_point_value)
                # print(i, coor[0][0], coor[1][0], "max: ", np.max(one_layer))
                x = coor[1][0] * 4
                y = coor[0][0] * 4
                draw_img = cv2.putText(draw_img, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255], 2)
                draw_img = cv2.circle(draw_img, (x, y), 3, [0, 255, 0], 2)
                point_result_dict.append([x, y])
        # plt.subplot(1, 2, 2)
        # cv2.imshow('frame',draw_img)
        # if cv2.waitKey(1) == ord('q'):
        #     cv2.destroyAllWindows()
        return point_result_dict

    def generate_key_points(self, image):
        """
        This is the interface of keypoints detection

        Input: 
            * RGB img ........ numpy array (H,W,3) 
        
        Output:  
            * keypoints in the orignal image ......... (list: len = 6, with [x,y])
        
        """
        point_in_original = []
        keypoint_in_original = []
        srcImg = image.copy()
        # import pdb;pdb.set_trace()

        srcImg_padding = cv2.copyMakeBorder(srcImg, 100, 100, 100, 100, cv2.BORDER_REPLICATE)

        h = srcImg_padding.shape[0]
        w = srcImg_padding.shape[1]
        w_ratio = w/640
        h_ratio = h/640
        count = 0
        srcImg_640 = cv2.resize(srcImg,self.dim)

        xyxy = self.detected_car_pos(srcImg_640)
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
                point_result = self.trolley_infer(self.transform, self.keypoint_model, srcImg_crop)
                # print("point_result: ", point_result)
                for i in range(6):
                    x = point_result[i][0] * (w_crop/256) + point_in_original[0]-(w_d*0.28).astype(int) -100
                    y = point_result[i][1] * (h_crop/256) + point_in_original[1]-(h_d*0.28).astype(int) -100
                    # srcImg = cv2.circle(srcImg, (x.astype(int) , y.astype(int) ), 10, [255, 255, 255], 4)
                    keypoint_in_original.append([x,y])
                print(keypoint_in_original)

                # new_name = 'test' + '.jpg'
                # dst = os.path.join(os.path.abspath(target_path), new_name)
                # cv2.imwrite(dst,srcImg)

        image_points = np.array(keypoint_in_original, dtype="double")
        return image_points
   
    def rpy2quaternion(self, roll, pitch, yaw):
        x=np.sin(pitch/2)*np.sin(yaw/2)*np.cos(roll/2)+np.cos(pitch/2)*np.cos(yaw/2)*np.sin(roll/2)
        y=np.sin(pitch/2)*np.cos(yaw/2)*np.cos(roll/2)+np.cos(pitch/2)*np.sin(yaw/2)*np.sin(roll/2)
        z=np.cos(pitch/2)*np.sin(yaw/2)*np.cos(roll/2)-np.sin(pitch/2)*np.cos(yaw/2)*np.sin(roll/2)
        w=np.cos(pitch/2)*np.cos(yaw/2)*np.cos(roll/2)-np.sin(pitch/2)*np.sin(yaw/2)*np.sin(roll/2)
        return x, y, z, w
   
    def rot2eul(self, R):
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        return np.array((alpha, beta, gamma))

    def solve_pose(self, image_points ):
        """
        This is the interface of PNP
        Input: 
            * keypoints in origninal image
        
        Output:  
            * R, T of robot in camera base
        
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

        euler_angles = self.rot2eul(R_)
        
        return R_,T, euler_angles

    def detect_3D(self, image):
        """
        This is the interface of 3D pose estimation
        Input: 
            * origninal rgb image  ........ numpy array (H,W,3) 
        
        Output:  
            * R, T of robot in camera base
        
        """
        points = self.generate_key_points(image)
        # image_points = np.array([[692,962],[784,946],[740,894],[574,916],[741,566],[568,553]], dtype="double")
        if len(points) > 0:
            print(points)
            R_, T, euler_angles = self.solve_pose(points)
            T = self.R.dot(T)
            return R_, T, euler_angles
        return [], [], []
