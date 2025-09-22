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
# sys.path.append('/data1/turbo_data/wangruihao/code/auto_labeling/4D_label')
from visual_data.utils.project_image import *
from visual_data.utils.load_param import load_yaml
from visual_data.utils.nms import nms_angle

import argparse
import time

from mmcv import Config
from torchpack.utils.config import configs
from BEVFUSION.tools.utils import poolprocess, get_calib_params_v2
from BEVFUSION.tools.visualize.get_infer_res import recursive_eval
from BEVFUSION.tools.data_process.sensor_modules import (
    single_pole_roadside_sensor_modules,
    roadside_sensor_modules,
)
from BEVFUSION.tools.visualize.check_calib_params import undistort_img
from visual_data.pcr_boundary_ground_z_config import pcr_boundary_ground_z
from visual_data.base_config import class_score_thresholds_distancewise

DISTANCE_INTERVALS = list(
    class_score_thresholds_distancewise[
        list(class_score_thresholds_distancewise.keys())[0]
    ].keys()
)

OBJECT_PALETTE_BEVFUSION = {
    "car":          (255, 0, 0),        #红色
    "truck":        (0, 255, 0),        #绿色
    "bus":          (0, 255, 255),      #青色
    "rider":        (255, 255, 0),      #黄色
    "bicycle":      (255, 0, 255),      #洋红
    "person":       (0, 0, 255),        #蓝色
    "motorcycle":   (255, 105, 180),    # hot pink
    "NotUsed":      (160, 32, 240),     #紫色
}
OBJECT_PALETTE_FISHYE_DETECT = {
    "car":          (255, 0, 0),        #红色
    "truck":        (0, 255, 0),        #绿色
    "bus":          (0, 255, 255),      #青色
    "rider":        (255, 255, 0),      #黄色
    "bicycle":      (255, 0, 255),      #洋红
    "person":       (0, 0, 255),        #蓝色
    "motorcycle":   (255, 105, 180),    # hot pink
    "motor":      (160, 32, 240),     #紫色
}

# name_combile_list = [
#     'car', #0
#     'bus', #1
#     'truck', #2
#     'rider', #3
#     'bicycle', #4
#     'person', #5
#     'NotUsed' #6
    # ]
# name_old_dict = {
#     'car':0,
#     'bus':1,
#     'truck':2,
#     'bike':4,
#     'bicycle':4,
#     'person':5,
#     'motor':4,
#     'NotUsed':6,
#     'rider': 3,
#     'pedestrain':5,
#     'pedestrian':5
# }

# confidence_type = {
#         'car':0.8,
#         'bus':0.7,
#         'truck':0.99,
#         'bike':0.7,
#         'bicycle':0.7,
#         'person':0.7,
#         'motor':0.7,
#         'NotUsed':0.7,
#         'rider': 0.7,
#         'pedestrain':0.7,
#         'pedestrian':0.7
#     }


name_combile_list = [
    'car', #0
    'truck', # 1   
    'bus', # 2
    'rider', #3
    'bicycle', # 4
    'person', # 5
    'motorcycle', # 6
    'NotUsed' # 7
]
name_old_dict = {  # TODO:修改类别！！！！！！
    'car':0,
    'truck':1,
    'bus':2,
    'bike':4,
    'bicycle':4,
    'person':5,
    'motor':6,
    'motorcycle':6,
    'NotUsed':7,
    'rider': 3,
    'pedestrain':5,
    'pedestrian':5
}
confidence_type = {
    'car':0.8,
    'bus':0.7,
    'truck':0.99,
    'bike':0.7,
    'bicycle':0.7,
    'person':0.7,
    'motor':0.7,
    'motorcycle':0.7,
    'NotUsed':0.7,
    'rider': 0.7,
    'pedestrain':0.7,
    'pedestrian':0.7
}

# 临时检测sh-4号路口
param_info = get_calib_params_v2(
    "/data1/turbo_data/huben/projects/BevCalib",
    "./BEVFUSION/tools/data_process/calibration_configs/calib_file.yaml",
    "sh_4",
    "v20250109",
)


def get_bev_dist_interval(x, y, distance_intervals):
    dist = (x ** 2 + y ** 2) ** 0.5
    res = distance_intervals[0]
    for dist_interval in distance_intervals:
        if dist >= dist_interval[0] and dist < dist_interval[1]:
            res = dist_interval
            break
    return res


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


def get_dense_boundary_points(pcr, mode, road_id):
    cur_ground_z = pcr_boundary_ground_z[road_id][mode]
    x_min, y_min, z_min, x_max, y_max, z_min = pcr
    delta = 1.0
    top = []
    for y in np.arange(y_min, y_max, delta):
        top.append((x_min, y, cur_ground_z["top"]))
    right = []
    for x in np.arange(x_min, x_max, delta):
        right.append((x, y_max, cur_ground_z["right"]))
    bottom = []
    for y in np.arange(y_max, y_min, -delta):
        bottom.append((x_max, y, cur_ground_z["bottom"]))
    left = []
    for x in np.arange(x_max, x_min, -delta):
        left.append((x, y_min, cur_ground_z["left"]))
    dense_boundary_points = top + right + bottom + left
    return dense_boundary_points

def draw_pcr_on_fisheye(
    mode,
    road_id,
    img,
    pcr,
    inter_param,
    exter_param,
    distort,
    camera_name,
    color=(255, 0, 0),
):
    img_mask = np.ones_like(img[:, :, 0])
    if road_id in pcr_boundary_ground_z:
        points = get_dense_boundary_points(pcr, mode, road_id)
        pcr_corners = []
        for pt in points:
            u, v, valid = project_3d_to_fisheye(
                # pt, img.shape, inter_param, exter_param, distort, filter_z_camera=False
                pt, img.shape, inter_param, exter_param, distort, filter_z_camera=True,
            )
            if u is not None and v is not None:
                pcr_corners.append([int(u), int(v)])
        pcr_corners = np.array(pcr_corners)
        
        img_mask = np.zeros_like(img[:, :, 0])
        cv2.fillPoly(img_mask, [pcr_corners], 255)
    
    # 0.47半径
    img_circle_mask = np.zeros_like(img[:, :, 0])
    h_, w_, _ = img.shape
    circle_radius = int(w_*0.47)
    cv2.circle(img_circle_mask, (w_//2,h_//2), circle_radius, 255, -1)
    
    final_mask = np.where(
        np.logical_and(img_mask > 0, img_circle_mask > 0), 255, 0
    ).astype(np.uint8)

    # TODO:临时异常处理！！！
    if road_id == "sh_2" and mode == "intersection" and camera_name == "camera_0_8":
        abnormal_mask = np.ones_like(img[:, :, 0]) * 255
        abnormal_mask = np.tril(abnormal_mask)  # 获取下三角
        abnormal_mask[:, w_ // 2:] = 0
        abnormal_mask = np.where(
            np.logical_and(img_circle_mask > 0, abnormal_mask > 0), 255, 0
        ).astype(np.uint8)
        final_mask = np.where(
            np.logical_or(abnormal_mask > 0, final_mask > 0), 255, 0
        ).astype(np.uint8)

    
    contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, color, 2)


def draw_pcr_on_cam(
    mode,
    road_id,
    img,
    pcr,
    inter_param,
    exter_param,
    distort,
    camera_name,
    color=(255, 0, 0),
):
    if road_id in pcr_boundary_ground_z and mode in pcr_boundary_ground_z[road_id]:
        points = get_dense_boundary_points(pcr, mode, road_id)
        pcr_corners = []
        for pt in points:
            u, v, valid = project_3d_to_pinhole(
                pt, img.shape, inter_param, exter_param
            )
            if u is not None and v is not None:
                pcr_corners.append([int(u), int(v)])
        pcr_corners = np.array(pcr_corners)
        
        img_mask = np.zeros_like(img[:, :, 0])
        cv2.fillPoly(img_mask, [pcr_corners], 255)
    
        contours, _ = cv2.findContours(
            img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, color, 2)


def draw_object_id(image, object_id, corners_2d, color=(0, 255, 255)):
    # corners_2d: 8角点
    # 将object id写在顶面左后点
    # 绘制顶面四条边
    # for i in range(4, 8):
    i = 7  # 顶面左后点
    cv2.putText(
        image,
        str(object_id),
        tuple(corners_2d[i].astype(int)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
        fontScale=1.0,               # 字体大小
        color=color,           # 文本颜色（BGR格式，这里是绿色）
        thickness=2
    )


def draw_vehicle_num(boxes_label, color_bar):
    vehicle_num = 0  # 记录大车的数量，包括car, truck, bus
    for label in boxes_label:
        class_name_label,h,w,l,x,y,z,yaw,confidence = label
        if class_name_label in ["car", "truck", "bus"]:
            vehicle_num += 1
    h, w = color_bar.shape[:2]
    cv2.putText(
        color_bar,
        f"vehicle_num: {vehicle_num}",
        (10, h - 40),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(255, 255, 255),
        thickness=2,
    )


def worker(args):
    (
        clip_tracked_dir,
        clip_merged_dir,
        frame_name,
        save_path,
        save_path_big_objects,
        test_json,
        image_pinhole_list,
        clip_origin_dir,
        image_save_size,
        image_fisheye_list,
        color_bar_raw,
        pcr,
        road_id,
        mode,
        use_fisheye,
        extra_show_big_objects,
    ) = args

    frame_tracted_name = os.path.join(clip_tracked_dir,frame_name)
    frame_merged_name = os.path.join(clip_merged_dir,frame_name)
    save_image_name = os.path.join(save_path, frame_name.replace('.txt','.jpg'))

    pcr_x_min, pcr_y_min, _, pcr_x_max, pcr_y_max, _ = pcr
    
    save_image_name_big_objects = None
    if save_path_big_objects is not None:
        save_image_name_big_objects = os.path.join(
            save_path_big_objects, frame_name.replace('.txt','.jpg')
        )
    if not os.path.exists(save_image_name) or True:
        boxes_label_tracked = []
        boxes_label_tracked_raw = []
        vehicle_and_rider_person_num = 0
        with open(frame_tracted_name,'r') as f:
            for line_ in f.readlines():
                content_list = line_.strip().split('\t')
                # name_list.append(name_combile_dict[label_dict[content_list[0]]])
                is_deg = True
                if len(content_list) > 9:
                    content_list = content_list[1:]
                    is_deg = False
                h,w,l,x,y,z,yaw,confidence = [float(i) for i in content_list[1:]] #c, h, w, l, new_center[0], new_center[1], new_center[2], yaw_new, confidence
                if is_deg:
                    yaw = math.radians(yaw)
                label_name = name_combile_list[name_old_dict[content_list[0]]]
        
                # TODO: 支持路段！！！！
                cur_dist_interval = get_bev_dist_interval(x, y, DISTANCE_INTERVALS)
                if confidence < class_score_thresholds_distancewise[label_name][cur_dist_interval]:
                    continue

                if label_name in ["car", "truck", "bus", "rider", "person"]:
                    vehicle_and_rider_person_num += 1

    
                boxes_label_tracked.append([label_name,h,w,l,x,y,z,yaw,confidence_type[label_name]])# cx, cy, l, w, r
                boxes_label_tracked_raw.append([label_name,h,w,l,x,y,z,yaw, confidence])# cx, cy, l, w, r
        
        nms_prepare_inputs = [[i_[4],i_[5],i_[3],i_[2],np.degrees(i_[-2]),i_[-1]] for i_ in boxes_label_tracked]
        _,nms_idx = nms_angle(nms_prepare_inputs,iou_thres=0.2)
        boxes_label_tracked_new  = [boxes_label_tracked[i] for i in nms_idx]
        boxes_label_tracked_new_raw = [boxes_label_tracked_raw[i] for i in nms_idx]

        # TODO: 保存nms之后的跟踪结果 !!!
        nms_frame_tracted_name = frame_tracted_name.replace("/splited/", "/splited_nms/")
        if not os.path.exists(nms_frame_tracted_name):
            os.makedirs(os.path.dirname(nms_frame_tracted_name), exist_ok=True)
            with open(nms_frame_tracted_name, "w") as f:
                for cur_nms_id, cur_nms_data in enumerate(boxes_label_tracked_new_raw):
                    # write object id to delete error detected !!!
                    cur_nms_data.insert(0, cur_nms_id)
                    f.write('\t'.join([str(i) for i in cur_nms_data])+'\n')

        if vehicle_and_rider_person_num < 15 and mode == "intersection":  # 临时路口使用！！！
            return
        

        boxes_label = boxes_label_tracked_new
        if  use_fisheye:
            lidar2cam_fisheye = test_json[0]['lidar2cam_fisheye']
            cam2img_fisheye = test_json[0]['cam2img_fisheye']
            distort_fisheye = test_json[0]['distort_fisheye']
        lidar2cam = test_json[0]['lidar2cam']
        cam2img = test_json[0]['cam2img']
        
        # 显示大车数目（car/truck/bus）
        color_bar = color_bar_raw.copy()
        draw_vehicle_num(boxes_label, color_bar)

        image_res_list = []
        image_res_big_obj_list = []
        for image_name in image_pinhole_list:
            if image_name == 'camera_0_0' and image_name not in cam2img: # TODO: temp hard code for mising camera_0_0
                image_res_list.append(np.zeros((800,800,3),np.uint8))
                image_res_big_obj_list.append(np.zeros((800,800,3),np.uint8))
                continue

            image = cv2.imread(os.path.join(clip_origin_dir,image_name,frame_name.replace('txt','jpg')))
            image = cv2.resize(image, (1280, 720))
            image_big_objects = image.copy()
            camera_matrix = np.array(cam2img[image_name])[:3,:3]
            camera_extrinsic = np.array(lidar2cam[image_name])
            
            # 临时去畸变
            if road_id == "sh_4":
                cur_cam_distort = param_info["cam"][image_name]["distortion"]
                image = undistort_img(image, camera_matrix[:3, :3], cur_cam_distort)

            
            for obj_id, label in enumerate(boxes_label):
                class_name_label,h,w,l,x,y,z,yaw,confidence = label
                
                if mode == "intersection" and (x < pcr_x_min or x > pcr_x_max or y < pcr_y_min or y > pcr_y_max):  # 支持路段
                        continue

                image, corner_2d = draw_box_on_pinhole((w,l,h,x,y,z),yaw,image,camera_matrix,camera_extrinsic,color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1])
                # TODO: draw object id on image
                if corner_2d is not None:
                    # draw_object_id(image, obj_id, corner_2d)
                    pass

                if extra_show_big_objects and class_name_label in ["car", "truck", "bus"]:
                    draw_box_on_pinhole((w,l,h,x,y,z),yaw,image_big_objects,camera_matrix,camera_extrinsic,color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1])
            
            # TODO: 支持路段！！！！！
            if mode == "intersection":
                draw_pcr_on_cam(
                    mode,
                    road_id,
                    image,
                    pcr,
                    camera_matrix,
                    camera_extrinsic,
                    None,
                    image_name,
                    color=(255, 0, 0),
                )
                if extra_show_big_objects:
                    draw_pcr_on_cam(
                        mode,
                        road_id,
                        image_big_objects,
                        pcr,
                        camera_matrix,
                        camera_extrinsic,
                        None,
                        image_name,
                        color=(255, 0, 0),
                    )
            image = cv2.resize(image,image_save_size)
            image_res_list.append(image)
            if extra_show_big_objects:
                image_big_objects = cv2.resize(image_big_objects, image_save_size)
                image_res_big_obj_list.append(image_big_objects)

        # return_image_pinhole = np.concatenate((image_res_list[0],image_res_list[1],image_res_list[2],image_res_list[3]),axis=1)
        # TODO: 临时处理sh_6丁字路口，没有S2数据
        if road_id == "sh_6":
            image_res_list.insert(2, np.zeros((800, 800, 3), np.uint8))  # S2
        if road_id == "sh_4":
            image_res_list.insert(2, np.zeros((800, 800, 3), np.uint8))  # S2
        return_image_pinhole = np.concatenate(image_res_list, axis=1)
        if extra_show_big_objects:
            return_image_pinhole_big_obj = np.concatenate(image_res_big_obj_list, axis=1)

        if use_fisheye:
            image_res_list = []
            image_res_big_obj_list = []
            for image_name in image_fisheye_list:
                image = cv2.imread(os.path.join(clip_origin_dir,image_name,frame_name.replace('txt','jpg')))
                image = cv2.resize(image, (1024, 1024))
                image_big_objects = image.copy()

                camera_matrix = np.array(cam2img_fisheye[image_name])[:3,:3]
                camera_extrinsic = np.array(lidar2cam_fisheye[image_name])
                camera_distort_param = np.array(distort_fisheye[image_name])
                # camera_distort_param = np.array([1.0926628389307196e-01, -6.5713320780575097e-04, 8.4866561354316559e-03, -4.2045330300667406e-03])
                for obj_id, label in enumerate(boxes_label):
                    class_name_label,h,w,l,x,y,z,yaw,confidence = label
                    if mode == "intersection" and (x < pcr_x_min or x > pcr_x_max or y < pcr_y_min or y > pcr_y_max):
                        continue

                    image, corners_2d = draw_box_on_fisheye((w,l,h,x,y,z),yaw,image,camera_matrix,camera_extrinsic,distort=camera_distort_param,color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1])
                    if corners_2d is not None:
                        # draw_object_id(image, obj_id, corners_2d)
                        pass
                    if extra_show_big_objects and class_name_label in ["car", "truck", "bus"]:
                        draw_box_on_fisheye((w,l,h,x,y,z),yaw,image_big_objects,camera_matrix,camera_extrinsic,distort=camera_distort_param,color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1])


                # 可视化pcr边界
                # 临时不画sh-4
                if road_id == "sh_4" or road_id == "sh_3":
                    draw_pcr_on_fisheye(
                        mode,
                        "",
                        image,
                        pcr,
                        camera_matrix,
                        camera_extrinsic,
                        camera_distort_param,
                        image_name,
                        color=(255, 0, 0),
                    )
                else:
                    draw_pcr_on_fisheye(
                        mode,
                        road_id,
                        image,
                        pcr,
                        camera_matrix,
                        camera_extrinsic,
                        camera_distort_param,
                        image_name,
                        color=(255, 0, 0),
                    )
                if extra_show_big_objects:
                    draw_pcr_on_fisheye(
                        mode,
                        road_id,
                        image_big_objects,
                        pcr,
                        camera_matrix,
                        camera_extrinsic,
                        camera_distort_param,
                        image_name,
                        color=(255, 0, 0),
                    )

                image = cv2.resize(image,image_save_size)
                image_res_list.append(image)
                if extra_show_big_objects:
                    image_big_objects = cv2.resize(image_big_objects, image_save_size)
                    image_res_big_obj_list.append(image_big_objects)
            
            # TODO: 临时处理sh_6丁字路口，没有S2数据
            if road_id == "sh_6":
                image_res_list.insert(2, np.zeros((800, 800, 3), np.uint8))  # S2
            if road_id == "sh_4":
                image_res_list.insert(2, np.zeros((800, 800, 3), np.uint8))  # S2
            return_image_fisheye = np.concatenate(image_res_list, axis=1)

            image_concated = np.concatenate(([return_image_pinhole,return_image_fisheye]),axis=0)
            image_concated = np.concatenate((image_concated,color_bar),axis=1)
            if extra_show_big_objects:
                return_image_fisheye_big_obj = np.concatenate(image_res_big_obj_list, axis=1)
                image_concated_big_obj = np.concatenate([return_image_pinhole_big_obj, return_image_fisheye_big_obj], axis=0)
                image_concated_big_obj = np.concatenate([image_concated_big_obj, color_bar], axis=1)
        else:
            image_concated = np.concatenate([return_image_pinhole, color_bar], axis=1)
            if extra_show_big_objects:
                image_concated_big_obj = np.concatenate([return_image_pinhole_big_obj, color_bar], axis=1)
        cv2.imwrite(save_image_name,image_concated)
        if extra_show_big_objects:
            cv2.imwrite(save_image_name_big_objects, image_concated_big_obj)

def get_pick_data_mp(
    origin_dir,
    label_dir,
    tracked_dir,
    bev_pro_dir,
    merged_dir,
    sub_t,
    pnum,
    pcr,
    road_id,
    mode,
    use_fisheye,
    pole,
    extra_show_big_objects,
):
    if mode == "intersection":
        image_save_size = (800,800)
        color_bar = get_palette((1600,350))
        # image_pinhole_list = ['camera_0_0','camera_1_0','camera_2_0','camera_3_0']
        image_pinhole_list = sorted([
            name for name in roadside_sensor_modules[road_id][mode]
            if name.startswith("camera") and not name.endswith("_8") 
        ])
        # image_fisheye_list = ['camera_0_8','camera_1_8','camera_2_8','camera_3_8']
        image_fisheye_list = sorted([
            name for name in roadside_sensor_modules[road_id][mode]
            if name.startswith("camera") and name.endswith("_8") 
        ])
    else:
        assert pole in ["S0", "S1", "S2", "S3"]
        image_save_size = (800,800)
        color_bar = get_palette((800,350))
        pole_modules = single_pole_roadside_sensor_modules[road_id][mode][pole]
        image_pinhole_list = [
            cam_name for cam_name in pole_modules
            if cam_name.startswith("camera")
        ]
        print("image_pinhole_list:", image_pinhole_list)
        assert len(image_pinhole_list) == 1
        image_fisheye_list = None

    sub_origin_dir = os.path.join(origin_dir,sub_t)
    sub_label_dir = os.path.join(label_dir,sub_t)
    sub_tracked_dir = os.path.join(tracked_dir,sub_t)
    sub_bev_pro_dir = os.path.join(bev_pro_dir,sub_t)
    sub_merged_dir = os.path.join(merged_dir,sub_t)
    test_json_path = os.path.join(sub_bev_pro_dir,'scences','test.json')
    save_path = os.path.join(sub_label_dir,'selected')
    os.makedirs(save_path, exist_ok=True)
    save_path_big_objects = None
    if extra_show_big_objects:
        save_path_big_objects = os.path.join(sub_label_dir,'selected_big_objects')
        os.makedirs(save_path_big_objects, exist_ok=True)


    with open(test_json_path) as f:
        test_json = json.load(f)

    clip_list = os.listdir(sub_tracked_dir)
    datas = []
    for clip_name in tqdm(clip_list):
        clip_origin_dir = os.path.join(sub_origin_dir,clip_name)
        clip_tracked_dir = os.path.join(sub_tracked_dir,clip_name,'splited')
        clip_tracked_dir_filtered = os.path.join(sub_tracked_dir, clip_name, 'splited_nms_filtered')
        if os.path.exists(clip_tracked_dir_filtered):
            clip_tracked_dir = clip_tracked_dir_filtered
        clip_merged_dir = os.path.join(sub_merged_dir,clip_name)
        frame_list = os.listdir(clip_tracked_dir)
        for frame_name in tqdm(frame_list):
            datas.append(
                [
                    clip_tracked_dir,
                    clip_merged_dir,
                    frame_name,
                    save_path,
                    save_path_big_objects,
                    test_json,
                    image_pinhole_list,
                    clip_origin_dir,
                    image_save_size,
                    image_fisheye_list,
                    color_bar,
                    pcr,
                    road_id,
                    mode,
                    use_fisheye,
                    extra_show_big_objects,
                ]
            )

    poolprocess(datas, worker, pnum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--road', type=str, required=True)
    parser.add_argument("--origin_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--merged_dir", type=str, required=True)
    parser.add_argument("--bev_pro_dir", type=str, required=True)
    parser.add_argument("--tracked_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument('--pnum', type=int, default=30)
    parser.add_argument(
        '--mode',
        type=str,
        default="intersection",
        choices=["intersection", "roadsection"],
        help="collect only one mode lidar data.",
    )
    parser.add_argument('--use_fisheye', action='store_true')
    parser.add_argument(
        '--pole',
        type=str,
        default="None",
        choices=["S0", "S1", "S2", "S3", "None"],
        help="collect only given module data.",
    )
    parser.add_argument("--extra_show_big_objects", action='store_true')

    args = parser.parse_args()
    configs.load(args.config, recursive=True)

    cfg = Config(recursive_eval(configs), filename=args.config)
    pcr = cfg.pcr
    road_id = f"{args.location}_{args.road}"

    print(f"Start Visualize data ...")
    start_time = time.time()
    # # 单进程：耗时~145.8s
    # get_pick_data(
    #     args.origin_dir,
    #     args.label_dir,
    #     args.tracked_dir,
    #     args.bev_pro_dir,
    #     args.merged_dir,
    #     args.dataset_name,
    # )

    # 多进程：10个进程耗时~24.6s/第二次耗时16.6s, 20个进程32.5s/第二次还是32.3s，可能是模型在训练！！！貌似10个进程性能更好！！！

    # TODO: 当前路段模式不适用鱼眼
    get_pick_data_mp(
        args.origin_dir,
        args.label_dir,
        args.tracked_dir,
        args.bev_pro_dir,
        args.merged_dir,
        args.dataset_name,
        args.pnum,
        pcr,
        road_id,
        args.mode,
        args.use_fisheye,
        args.pole,
        args.extra_show_big_objects,
    )
    print(f"=============== Visualize data cost: {time.time() - start_time}s.")

    # get_pick_data('train_hy_4d_road_7_20250209_lx')
    # origin_dir = '/data1/turbo_data/4D_label_dataset/origin'
    # label_dir = '/data1/turbo_data/4D_label_dataset/labels'
    # merged_dir = '/data1/turbo_data/4D_label_dataset/models_res/merged'
    # bev_pro_dir = '/data1/turbo_data/4D_label_dataset/models_res/bevpro/' #train_hy_4d_road_7_20250208_lx/scences'
    # tracked_dir = '/data1/turbo_data/4D_label_dataset/models_res/offline_tracked'
