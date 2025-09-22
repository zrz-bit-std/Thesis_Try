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
from visual_data.utils.project_image import *
from visual_data.utils.load_param import load_yaml



OBJECT_PALETTE_BEVFUSION = {
    "car":          (255, 0, 0),        #红色
    "truck":        (0, 255, 0),        #绿色
    "bus":          (0, 255, 255),      #青色
    "rider":        (255, 255, 0),      #黄色
    "bicycle":      (255, 0, 255),      #洋红
    "pedestrian":   (0, 0, 255),        #蓝色
    "NotUsed":      (160, 32, 240),     #紫色
}
OBJECT_PALETTE_FISHYE_DETECT = {
    "car":          (255, 0, 0),        #红色
    "truck":        (0, 255, 0),        #绿色
    "bus":          (0, 255, 255),      #青色
    "rider":        (255, 255, 0),      #黄色
    "bicycle":         (255, 0, 255),      #洋红
    "pedestrian":       (0, 0, 255),        #蓝色
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


def read_param_from_txt(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(':')
            key = parts[0].strip()
            values = parts[1].strip().split(' ')
            # 将字符串类型的数值转换为对应的数值类型（float），如果不需要转换可以注释掉下一行
            values = [float(val) for val in values]
            result_dict[key] = values
    intri_param = [np.array(result_dict[k]).reshape(3,4)[:,:3] for k in ["P0","P1","P2","P3"]]
    extri_param = [np.array(result_dict[k] + [0,0,0,1]).reshape(4,4) for k in ['Tr_velo_to_cam_0','Tr_velo_to_cam_1','Tr_velo_to_cam_2','Tr_velo_to_cam_3']]
    return intri_param,extri_param


def read_param_from_yaml_fisheye(param_base_dir,param_base_dir_raw,name_list):
    intri_param_list = []
    extri_param_list = []
    distort_param_list = []
    for item in name_list:
        data_ = load_yaml(os.path.join(param_base_dir,item))
        T_common_cam = data_['T_common_cam']['data']
        matrix_param = np.linalg.inv(np.array(T_common_cam).reshape(4,4))
        extri_param_list.append(matrix_param)
        data_raw = load_yaml(os.path.join(param_base_dir_raw,item))
        projection_parameters = data_raw['projection_parameters']
        distort_param_list.append([projection_parameters[i] for i in ['k2','k3','k4','k5']])
        intri_param_matrix = np.eye(3)
        intri_param_matrix[0,0] = projection_parameters['mu']/2.0
        intri_param_matrix[1,1] = projection_parameters['mv']/2.0
        intri_param_matrix[0,2] = projection_parameters['u0']/2.0
        intri_param_matrix[1,2] = projection_parameters['v0']/2.0
        intri_param_list.append(intri_param_matrix)
    return intri_param_list,extri_param_list,distort_param_list


def get_label(label_file_path):
    class_info_list = []
    objects_data = []
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            object_info = [
                float(parts[-8]),  # h
                float(parts[-7]),  # w
                float(parts[-6]),  # l
                float(parts[-5]),  # x
                float(parts[-4]),  # y
                float(parts[-3]),  # z
                float(parts[-2])  # yaw
            ]
            class_info_list.append('car')
            objects_data.append(object_info) 
    return  class_info_list,objects_data


def get_image_param(fpath):
    return None


def main():
    image_save_size = (800,800)
    color_bar = get_palette((1600,350))
    train_data_path = '/data1/turbo_data/RALG/data/3.0_pro/Intersection/version/v4_fisheye/scences/train.json'
    image_base_dir = '/data1/turbo_data/RALG/data/3.0_pro/Intersection/version/v4_fisheye/samples'
    with open(train_data_path) as f:
        res = json.load(f)
    image_pinhole_list = ['camera_0_0','camera_1_0','camera_2_0','camera_3_0']
    image_fisheye_list = ['camera_0_8','camera_1_8','camera_2_8','camera_3_8']
    save_path = '/data1/turbo_data/wangruihao/data/4D_label/tmp_save'
    for clip_item in res:
        samples = clip_item['samples']
        lidar2cam_fisheye = clip_item['lidar2cam_fisheye']
        cam2img_fisheye = clip_item['cam2img_fisheye']
        distort_fisheye = clip_item['distort_fisheye']
        lidar2cam = clip_item['lidar2cam']
        cam2img = clip_item['cam2img']
        for frame in tqdm(samples):
            base_name = frame['token']
            image_name_dict = frame['cam_imgs']
            image_fisheye_name_dict = frame['cam_fisheye_imgs']
            boxes_label = frame['annos']
            image_res_list = []  
            for image_name in image_pinhole_list:
                image = cv2.imread(os.path.join(image_base_dir,image_name_dict[image_name]))
                camera_matrix = np.array(cam2img[image_name])[:3,:3]
                camera_extrinsic = np.array(lidar2cam[image_name])
                for label in boxes_label:
                    center = label['center']
                    size_ = label['size']
                    class_name_label = label['type']
                    yaw = -label['rotation']['yaw'] - (math.pi/2.0)
                    w, l,h, x, y, z = size_['x'],size_['y'],size_['z'],center['x'],center['y'],center['z']
                    draw_box_on_pinhole((w,l,h,x,y,z),yaw,image,camera_matrix,camera_extrinsic,color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1])
                image = cv2.resize(image,image_save_size)
                image_res_list.append(image)
            return_image_pinhole = np.concatenate((image_res_list[0],image_res_list[1],image_res_list[2],image_res_list[3]),axis=1)


            image_res_list = []  
            for image_name in image_fisheye_list:
                image = cv2.imread(os.path.join(image_base_dir,image_fisheye_name_dict[image_name]))
                camera_matrix = np.array(cam2img_fisheye[image_name])[:3,:3]
                camera_extrinsic = np.array(lidar2cam_fisheye[image_name])
                camera_distort_param = np.array(distort_fisheye[image_name])
                for label in boxes_label:
                    center = label['center']
                    size_ = label['size']
                    class_name_label = label['type']
                    yaw = -label['rotation']['yaw'] - (math.pi/2.0)
                    w, l,h, x, y, z = size_['x'],size_['y'],size_['z'],center['x'],center['y'],center['z']
                    draw_box_on_fisheye((w,l,h,x,y,z),yaw,image,camera_matrix,camera_extrinsic,distort=camera_distort_param,color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1])
                image = cv2.resize(image,image_save_size)
                image_res_list.append(image)
            return_image_fisheye = np.concatenate((image_res_list[0],image_res_list[1],image_res_list[2],image_res_list[3]),axis=1)

            image_concated = np.concatenate(([return_image_pinhole,return_image_fisheye]),axis=0)
            image_concated = np.concatenate((image_concated,color_bar),axis=1)
            cv2.imwrite(os.path.join(save_path,base_name+'.jpg'),image_concated)

if __name__ == '__main__':
    main()