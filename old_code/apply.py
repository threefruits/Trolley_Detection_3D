import argparse
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
height = 640
weight = 640
conf_thres=0.30
iou_thres=0.45
line_thickness = 3
dim = (weight, height)
model = attempt_load('./best_500.pt', map_location=device)
names = model.module.names if hasattr(model, 'module') else model.names 


def detected_car_pos(img):
    
    srcImg1 = img.copy()
    srcImg = srcImg1.transpose((2, 0, 1))[::-1]
    srcImg = np.ascontiguousarray(srcImg)
    img = torch.from_numpy(srcImg).to(device)
    img = img.float() 
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    # print(img.shape)

    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)

    im0 = srcImg1.copy()
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = det[:, :4].round()

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                return xyxy
                
    return None

def detected_car(img):
    
    srcImg1 = cv2.resize(img,dim)
    srcImg = srcImg1.transpose((2, 0, 1))[::-1]
    srcImg = np.ascontiguousarray(srcImg)
    img = torch.from_numpy(srcImg).to(device)
    img = img.float() 
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    # print(img.shape)

    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)

    im0 = srcImg1.copy()
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = det[:, :4].round()

            for *xyxy, conf, cls in reversed(det):
                
                c = int(cls)
                print(xyxy)
                label = f'{names[c]} {conf:.2f}'
                # draw_0 = cv2.rectangle(image, (x_center-h, y_center-h), (x_center+h, y_center+h), (255, 0, 0), 2)
                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
    return im0



if __name__ == "__main__":

    cap = cv2.VideoCapture(1)
    while(cap.isOpened()):
        ret, frame = cap.read()
        # frame1=cv2.flip(frame,1)
        frame_detected = detected_car(frame)
        cv2.imshow('frame', frame_detected)
        if cv2.waitKey(1) == ord('q'):
            break

    # img = cv2.imread("./test.jpg")
    # print(img.shape)
    # img_2 = detected_car(img)
    # while True:
    #     cv2.imshow("ss", img_2)
    #     cv2.waitKey(1)
    # print(img_2.shape)
