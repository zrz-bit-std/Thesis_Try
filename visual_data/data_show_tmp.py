import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
from typing import Optional, List, Tuple
# import mmcv
from tqdm import tqdm
import json
import re
import sys
sys.path.append('/data1/turbo_data/wangruihao/code/4D_label')
from visual_data.utils.project_image import *
from visual_data.utils.load_param import load_yaml

name_combile_list = [
        'car', #0
        'bus', #1
        'truck', #2
        'rider', #3
        'bicycle', #4
        'person', #5
        'NotUsed' #6
        ]
name_old_dict = {
        'car':0,
        'bus':1,
        'truck':2,
        'bike':4,
        'bicycle':4,
        'person':5,
        'motor':4,
        'NotUsed':6,
        'rider': 3,
        'pedestrain':5,
        'pedestrian':5
    }

OBJECT_PALETTE_BEVFUSION = {
    "car":          (255, 0, 0),        #红色
    "truck":        (0, 255, 0),        #绿色
    "bus":          (0, 255, 255),      #青色
    "rider":        (255, 255, 0),      #黄色
    "bicycle":      (255, 0, 255),      #洋红
    "person":       (0, 0, 255),        #蓝色
    "NotUsed":      (160, 32, 240),     #紫色
}
OBJECT_PALETTE_FISHYE_DETECT = {
    "car":          (255, 0, 0),        #红色
    "truck":        (0, 255, 0),        #绿色
    "bus":          (0, 255, 255),      #青色
    "rider":        (255, 255, 0),      #黄色
    "bicycle":      (255, 0, 255),      #洋红
    "person":       (0, 0, 255),        #蓝色
    "motor":      (160, 32, 240),     #紫色
}


def get_palette(size):
    h, w = size
    canvas = np.zeros((h, w, 3), np.uint8)
    color_height = 30
    color_width = 60
    vertical_gap = 10
    palette_horizontal_start = 10
    horizontal_gap = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    font_color = (255, 255, 255)
    
    object_palette_bevfusion_append = {
        "bevfusion": None, **OBJECT_PALETTE_BEVFUSION
    }
    object_palette_fisheyedet_append = {
        "fisheyedet": None, **OBJECT_PALETTE_FISHYE_DETECT
    }
    palettes = [object_palette_bevfusion_append, object_palette_fisheyedet_append]
    
    i = 0
    for cur_palette in palettes:
        for name, color in cur_palette.items():
            top_left_u = palette_horizontal_start
            top_left_v = i * (color_height + vertical_gap) + vertical_gap
            bottom_right_u = top_left_u + color_width
            bottom_right_v = top_left_v + color_height
            if color is not None:
                color_bgr = color[::-1]
                cv2.rectangle(
                    canvas,
                    (top_left_u, top_left_v),
                    (bottom_right_u, bottom_right_v),
                    color_bgr,
                    -1,
                )
                cv2.putText(
                    canvas,
                    name,
                    (top_left_u + color_width + horizontal_gap, bottom_right_v),
                    font,
                    font_scale,
                    font_color,
                    thickness,
                )
            else:
                cv2.putText(
                    canvas,
                    f"{name}: ",
                    (top_left_u, bottom_right_v),
                    font,
                    font_scale,
                    font_color,
                    thickness,
                )
            i += 1

    return canvas




origin_dir = '/data1/turbo_data/4D_label_dataset/origin'
label_dir = '/data1/turbo_data/4D_label_dataset/labels'
merged_dir = '/data1/turbo_data/4D_label_dataset/models_res/merged'
bev_pro_dir = '/data1/turbo_data/4D_label_dataset/models_res/bevpro/' #train_hy_4d_road_7_20250208_lx/scences'
tracked_dir = '/data1/turbo_data/4D_label_dataset/models_res/offline_tracked'


def get_pick_data(sub_t):
    image_save_size = (800,800)
    color_bar = get_palette((1600,350))
    image_pinhole_list = ['camera_0_0','camera_1_0','camera_2_0','camera_3_0']
    image_fisheye_list = ['camera_0_8','camera_1_8','camera_2_8','camera_3_8']

    sub_origin_dir = os.path.join(origin_dir,sub_t)
    sub_label_dir = os.path.join(label_dir,sub_t)
    sub_tracked_dir = os.path.join(tracked_dir,sub_t)
    sub_bev_pro_dir = os.path.join(bev_pro_dir,sub_t)
    sub_merged_dir = os.path.join(merged_dir,sub_t)
    test_json_path = os.path.join(sub_bev_pro_dir,'scences','test.json')
    save_path = os.path.join(sub_label_dir,'selected')
    os.makedirs(save_path,exist_ok=True)

    with open(test_json_path) as f:
        test_json = json.load(f)

    clip_list = os.listdir(sub_merged_dir)
    for clip_name in tqdm(clip_list):
        clip_origin_dir = os.path.join(sub_origin_dir,clip_name)
        # clip_tracked_dir = os.path.join(sub_tracked_dir,clip_name,'splited')
        clip_merged_dir = os.path.join(sub_merged_dir,clip_name)
        frame_list = os.listdir(clip_merged_dir)
        for frame_name in tqdm(frame_list):
            # frame_tracted_name = os.path.join(clip_tracked_dir,frame_name)
            frame_merged_name = os.path.join(clip_merged_dir,frame_name)
            # boxes_label_tracked = []
            # with open(frame_tracted_name,'r') as f:
            #     for line_ in f.readlines():
            #         content_list = line_.strip().split('\t')
            #         # name_list.append(name_combile_dict[label_dict[content_list[0]]])
            #         h,w,l,x,y,z,yaw,confidence = [float(i) for i in content_list[1:]] #c, h, w, l, new_center[0], new_center[1], new_center[2], yaw_new, confidence
            #         yaw = math.radians(yaw)
            #         label_name = name_combile_list[name_old_dict[content_list[0]]]
            #         boxes_label_tracked.append([label_name,h,w,l,x,y,z,yaw,confidence])
            boxes_label_merged = []
            with open(frame_merged_name,'r') as f:
                for line_ in f.readlines():
                    content_list = line_.strip().split('\t')
                    # name_list.append(name_combile_dict[label_dict[content_list[0]]])
                    h,w,l,x,y,z,yaw,confidence = [float(i) for i in content_list[1:]] #c, h, w, l, new_center[0], new_center[1], new_center[2], yaw_new, confidence
                    yaw = math.radians(yaw)
                    label_name = name_combile_list[name_old_dict[content_list[0]]]
                    boxes_label_merged.append([label_name,h,w,l,x,y,z,yaw,confidence])
            # if len(boxes_label_merged) == len(boxes_label_tracked):
            #     continue

            boxes_label = boxes_label_merged
            lidar2cam_fisheye = test_json[0]['lidar2cam_fisheye']
            cam2img_fisheye = test_json[0]['cam2img_fisheye']
            distort_fisheye = test_json[0]['distort_fisheye']
            lidar2cam = test_json[0]['lidar2cam']
            cam2img = test_json[0]['cam2img']
            
            
            image_res_list = []  
            for image_name in image_pinhole_list:
                if image_name == 'camera_0_0':
                    image_res_list.append(np.zeros((800,800,3),np.uint8))
                    continue
                image = cv2.imread(os.path.join(clip_origin_dir,image_name,frame_name.replace('txt','jpg')))
                camera_matrix = np.array(cam2img[image_name])[:3,:3]
                camera_extrinsic = np.array(lidar2cam[image_name])
                for label in boxes_label:
                    class_name_label,h,w,l,x,y,z,yaw,confidence = label
                    # yaw = -1*math.pi/2.0 - math.radians(yaw)
                    draw_box_on_pinhole((w,l,h,x,y,z),yaw,image,camera_matrix,camera_extrinsic,color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1])
                image = cv2.resize(image,image_save_size)
                image_res_list.append(image)
            return_image_pinhole = np.concatenate((image_res_list[0],image_res_list[1],image_res_list[2],image_res_list[3]),axis=1)


            image_res_list = []  
            for image_name in image_fisheye_list:
                image = cv2.imread(os.path.join(clip_origin_dir,image_name,frame_name.replace('txt','jpg')))
                camera_matrix = np.array(cam2img_fisheye[image_name])[:3,:3]
                camera_extrinsic = np.array(lidar2cam_fisheye[image_name])
                camera_distort_param = np.array(distort_fisheye[image_name])
                for label in boxes_label:
                    class_name_label,h,w,l,x,y,z,yaw,confidence = label
                    # yaw = -1*math.pi/2.0 - math.radians(yaw)
                    draw_box_on_fisheye((w,l,h,x,y,z),yaw,image,camera_matrix,camera_extrinsic,distort=camera_distort_param,color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1])
                image = cv2.resize(image,image_save_size)
                image_res_list.append(image)
            return_image_fisheye = np.concatenate((image_res_list[0],image_res_list[1],image_res_list[2],image_res_list[3]),axis=1)

            image_concated = np.concatenate(([return_image_pinhole,return_image_fisheye]),axis=0)
            image_concated = np.concatenate((image_concated,color_bar),axis=1)
            cv2.imwrite(os.path.join(save_path,frame_name.replace('.txt','.jpg')),image_concated)



# def fisheye_tranform_point_to_origin(points, map1):
#     pts = np.copy(points)
#     shape = map1.shape[0:2]
#     for i, p in enumerate(pts):
#         x = min(max(0, int(p[0])), shape[0]-1)
#         y = min(max(0, int(p[1])), shape[1]-1)
#         new_x = map1[x, y][0]
#         new_y = map1[x, y][1]
#         # new_x = map1[x, y]
#         # new_y = map2[x, y]
#         pts[i] = np.array([new_y, new_x])
    
#     return pts


# def get_fishmap(calib_path, fish_cam_num):
#     # calib_path = "/data1/turbo_data/liangyaolin/road_dataset/8cam_bev_dataset/training/calib/1726199522000.txt"

#     # 图像尺寸 (width, height)
#     img_size = (1024, 1024)

#     P, V2C, C2V = get_calibration(calib_path, Px=f'P{fish_cam_num}')

#     K = P[:, 0:3]

#     D = np.array([1.0926628389307196e-01, -6.5713320780575097e-04, 8.4866561354316559e-03, -4.2045330300667406e-03])

#     new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, img_size, np.eye(3), balance=1.0)
    
#     # 手动缩小 new_K 的焦距（减少 fx 和 fy），扩大视场，减少裁剪
#     scaling_factor = 1.0
#     new_K[0, 0] *= scaling_factor  # fx
#     new_K[1, 1] *= scaling_factor  # fy
#     # 计算正向的去畸变映射矩阵
#     mapX, mapY = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, img_size, cv2.CV_16SC2)
#     # mapX, mapY = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, img_size, cv2.CV_32FC1)
    
#     return mapX, mapY

            

if __name__ == '__main__':
    get_pick_data('train_hy_4d_road_7_20250210_lx')