import os
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')



from apply import detected_car_pos,detected_car
from car_infer import card_keypoints_init, card_infer
import cv2 
root_path = "./trolley/"
target_path = "./trolley_detected/"

count = 0
height = 640
weight = 640
dim = (weight, height)

model_path = r'./epoch_50.pth'
transform, model = card_keypoints_init(model_path)

for root,dir,files in os.walk(root_path):
    print(files)
    for file in files:
        point_in_original = []
        keypoint_in_original = []
        srcImg = cv2.imread(root_path+str(file))
        srcImg_padding = cv2.copyMakeBorder(srcImg, 100, 100, 100, 100, cv2.BORDER_REPLICATE)

        h = srcImg_padding.shape[0]
        w = srcImg_padding.shape[1]
        w_ratio = w/640
        h_ratio = h/640
        srcImg_640 = cv2.resize(srcImg,dim)

        xyxy = detected_car_pos(srcImg_640)
        # xyxy = detected_car(srcImg_640)
        if(xyxy):
            import pdb;pdb.set_trace()
            point_in_original.append( ( (xyxy[0].cpu().numpy() - 320) * w_ratio + w/2).astype(int) )
            point_in_original.append( ( (xyxy[1].cpu().numpy() - 320) * h_ratio + h/2).astype(int) )
            point_in_original.append( ( (xyxy[2].cpu().numpy() - 320) * w_ratio + w/2).astype(int) )
            point_in_original.append( ( (xyxy[3].cpu().numpy() - 320) * h_ratio + h/2).astype(int) )
            h_d = (point_in_original[3] - point_in_original[1])
            w_d = (point_in_original[2] - point_in_original[0]) 
            h_crop = (point_in_original[3] - point_in_original[1]) * 1.56
            w_crop = (point_in_original[2] - point_in_original[0]) * 1.56  #crop the  wheel
            #
            if(0<point_in_original[0]-(w_d*0.28).astype(int) and 0<point_in_original[1]-(h_d*0.28).astype(int) and point_in_original[2]+(w_d*0.28).astype(int)<w and point_in_original[3]+(h_d*0.28).astype(int)<h):
                srcImg_crop = srcImg_padding[point_in_original[1]-(h_d*0.28).astype(int):point_in_original[3]+(h_d*0.28).astype(int),point_in_original[0]-(w_d*0.28).astype(int):point_in_original[2]+(w_d*0.28).astype(int)]
                # print(srcImg_padding.shape,srcImg_crop.shape,point_in_original)

                srcImg_crop = cv2.resize(srcImg_crop,(256,256))
                point_result = card_infer(transform, model, srcImg_crop)

                new_name_gt = '00' + format(str(count), '0>4s') + '_gt.jpg'
                dst_gt = os.path.join(os.path.abspath(target_path), new_name_gt)
                cv2.imwrite(dst_gt,srcImg)

                # print("point_result: ", point_result)
                for i in range(6):
                    x = point_result[i][0] * (w_crop/256) + point_in_original[0]-(w_d*0.28).astype(int) -100
                    y = point_result[i][1] * (h_crop/256) + point_in_original[1]-(h_d*0.28).astype(int) -100
                    srcImg = cv2.circle(srcImg, (x.astype(int) , y.astype(int) ), 10, [255, 255, 255], 4)
                    keypoint_in_original.append([x,y])
                # print(keypoint_in_original)
                
                new_name = '00' + format(str(count), '0>4s') + '.jpg'
                dst = os.path.join(os.path.abspath(target_path), new_name)
                cv2.imwrite(dst,srcImg)

                
                count += 1
        # new_name = '00' + format(str(count), '0>4s') + '.jpg'
        # dst = os.path.join(os.path.abspath(target_path), new_name)

        # cv2.imwrite(dst,xyxy)
        # count += 1