"""绘制检测结果和跟踪结果到BEV图上，并生成视频"""
import os
import json
import cv2
import glob
import argparse
import math

import pickle
import numpy as np
from tqdm import tqdm
import subprocess
from collections import defaultdict

import sys
sys.path.insert(0, "/data1/turbo_data/lishuaiyin/4D_label")
from visual_data.utils.nms import nms_angle
from visual_data.utils.visualize import _COLORS

# import open3d as o3d
import matplotlib.pyplot as plt

os.environ["DISPLAY"] = ":0"
os.environ["PYOPENGL_PLATFORM"] = "egl"

def draw_detection_box_on_pcd(params):
    """
    绘制检测框到点云图上
    params: [h, w, l, x, y, z, yaw_deg, score_id]
    """
    h, w, l, x, y, z, yaw_deg, score_id = params
    # 转换角度从度到弧度
    yaw = np.radians(yaw_deg)
    # 调整z坐标，因为输入的是z+(h/2.0)
    z = z - (h / 2.0)
    
    corners_3d_obj = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2,  -w / 2, -h / 2],
        [l / 2,   w / 2, -h / 2],
        [-l / 2,  w / 2, -h / 2],
        [-l / 2, -w / 2,  h / 2],
        [l / 2,  -w / 2,  h / 2],
        [l / 2,   w / 2,  h / 2],
        [-l / 2,  w / 2,  h / 2]
    ])
    corners_3d_obj_roation = rotate_points(corners_3d_obj.T, yaw)
    corners_3d = np.array([[x, y, z]]).T + corners_3d_obj_roation
    corners_3d = corners_3d.T[:5, :2]
    corners_3d[4, :] = corners_3d[0, :]

    return corners_3d

def read_detection_txt(file_path):
    """
    读取检测结果txt文件
    返回: [[h, w, l, x, y, z+(h/2.0), np.degrees(yaw), score_id], ...]
    """
    boxes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) >= 9:  # 确保有足够的值
                    # label h w l x y z+(h/2.0) math.degrees(yaw) score_id
                    label = vals[0]
                    h, w, l, x, y, z_offset, yaw_deg, score_id = map(float, vals[1:9])
                    boxes.append([h, w, l, x, y, z_offset, yaw_deg, score_id])
    except Exception as e:
        print(f"Error reading detection file {file_path}: {e}")
    return boxes



def images_to_video_ffmpeg(image_folder, output_video, fps=10):

        files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
        # files = files[:300]
        # files = files[4940:]
        with open("filelist.txt", "w") as f:
            for file in files:
                f.write(f"file '{os.path.join(image_folder, file)}'\n")
        cmd = [
           "/usr/bin/ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", "filelist.txt", 
            # "-vf", "setpts=2.0*PTS",  ##
            "-r", str(fps),
            "-c:v", "libx264",          # 
            "-preset", "slow",          # 
            "-crf", "23",               # 
            "-pix_fmt", "yuv420p",      # 
            "-movflags", "+faststart",  # 
            "-vf", "scale=iw:ih",       # 
            "-y",
            output_video                # 
        ]
        subprocess.run(cmd)
        os.remove("filelist.txt")  # 清理临时文件

def rotate_points(points, yaw):
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return rotation_matrix @ points

def draw_detection_box_on_pcd(params):
    """
    绘制检测框到点云图上
    params: [h, w, l, x, y, z, yaw_deg, score_id]
    """
    h, w, l, x, y, z, yaw_deg, score_id = params
    # 转换角度从度到弧度
    yaw = np.radians(yaw_deg)
    # 调整z坐标，因为输入的是z+(h/2.0)
    z = z - (h / 2.0)
    
    corners_3d_obj = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2,  -w / 2, -h / 2],
        [l / 2,   w / 2, -h / 2],
        [-l / 2,  w / 2, -h / 2],
        [-l / 2, -w / 2,  h / 2],
        [l / 2,  -w / 2,  h / 2],
        [l / 2,   w / 2,  h / 2],
        [-l / 2,  w / 2,  h / 2]
    ])
    corners_3d_obj_roation = rotate_points(corners_3d_obj.T, yaw)
    corners_3d = np.array([[x, y, z]]).T + corners_3d_obj_roation
    corners_3d = corners_3d.T[:5, :2]
    corners_3d[4, :] = corners_3d[0, :]

    return corners_3d
def draw_track_box_on_pcd(params):
    """
    绘制跟踪框到点云图上
    params: [x, y, z, l, w, h, yaw, _, _]
    """
    x, y, z, l, w, h, yaw, _, _ = params
    corners_3d_obj = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2,  -w / 2, -h / 2],
        [l / 2,   w / 2, -h / 2],
        [-l / 2,  w / 2, -h / 2],
        [-l / 2, -w / 2,  h / 2],
        [l / 2,  -w / 2,  h / 2],
        [l / 2,   w / 2,  h / 2],
        [-l / 2,  w / 2,  h / 2]
    ])
    corners_3d_obj_roation = rotate_points(corners_3d_obj.T, yaw)
    corners_3d = np.array([[x, y, z]]).T + corners_3d_obj_roation
    corners_3d = corners_3d.T[:5, :2]
    corners_3d[4, :] = corners_3d[0, :]

    return corners_3d

def draw_box_on_pcd(params):
    # x, y, z, l, w, h, yaw = params
    # # yaw = -1*(math.pi/2.0) - yaw_
    # yaw = (yaw + math.pi / 2) % (2 * math.pi)
    # breakpoint()
    x, y, z, l, w, h, yaw, _, _ = params
    corners_3d_obj = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2,  -w / 2, -h / 2],
        [l / 2,   w / 2, -h / 2],
        [-l / 2,  w / 2, -h / 2],
        [-l / 2, -w / 2,  h / 2],
        [l / 2,  -w / 2,  h / 2],
        [l / 2,   w / 2,  h / 2],
        [-l / 2,  w / 2,  h / 2]
    ])
    corners_3d_obj_roation = rotate_points(corners_3d_obj.T, yaw)
    corners_3d = np.array([[x, y, z]]).T + corners_3d_obj_roation
    corners_3d = corners_3d.T[:5, :2]
    corners_3d[4, :] = corners_3d[0, :]

    return corners_3d


def seq_res_on_bev(sequence, colored):

    xlim = ylim =[-80, 80]
    radius = 15
    linewidth = 10

    pcd_root = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track", sequence, "samples/lidar")
    pcd_list = [os.path.join(pcd_root, pcd_file) for pcd_file in os.listdir(pcd_root)]
    pcd_list = sorted(pcd_list)
    nums = len(pcd_list)

    seq_path = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track/offline_tracked", sequence)
    scene_name_list = sorted(os.listdir(seq_path))
    scene_list = [os.path.join(seq_path, scene) for scene in scene_name_list]
    obj_occ = defaultdict(list)
    count_num = 0
    for file_path in scene_list:
        res_file_root = os.path.join(file_path, "tracking")
        split_file_root = os.path.join(file_path, "splited")
        res_file = [os.path.join(res_file_root, file) for file in os.listdir(res_file_root) if file.startswith("tracking-test")][0]
        with open(res_file, "rb")as file:
            data = pickle.load(file)
            res_data = data[next(iter(data.keys()))] 

        for key, value in res_data.items():
            color = (_COLORS[key%len(_COLORS)] * 255).astype(np.uint8).tolist()
            color_normalized = (color[0]/255, color[1]/255, color[2]/255)
            for idx, params in zip(value['sample_idx'].tolist(), value['boxes_global']):
                
                idx = int(idx) + count_num
                pts = draw_box_on_pcd(params)
                obj_occ[idx].append((pts, color_normalized))
        count_num += len(os.listdir(split_file_root))
    
    for i in tqdm(range(nums)):
        lidar_pts = np.fromfile(pcd_list[i], dtype=np.float32).reshape(-1, 5)
        img_name = pcd_list[i].split('/')[-1].replace('pcd.bin', 'jpg')
        save_path = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all", sequence, "bev", img_name)
        fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
        ax = plt.gca()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.set_axis_off()
        colors = plt.get_cmap("rainbow")((lidar_pts[:, 3]-lidar_pts[:, 3].min())/(lidar_pts[:, 3].max()-lidar_pts[:, 3].min()))
        if lidar_pts is not None:
            plt.scatter(
                lidar_pts[:, 0],
                lidar_pts[:, 1],
                s=radius,
                c=colors[:, :3] if colored else 'w',
                # c='w',
            )
        for bbox, color_ in obj_occ[i]:
            ax.plot(bbox[:, 0], bbox[:, 1], color=color_, linewidth=linewidth, linestyle='-')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(
            save_path,
            dpi=10,
            facecolor="black",
            format="jpg",
            bbox_inches="tight",
            pad_inches=0,
        )
        ax.clear()
        plt.close(fig)
        
    image_folder = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all", sequence, "bev")
    save_video_path = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all/video_res", sequence + "_bev.mp4")
    images_to_video_ffmpeg(image_folder, save_video_path, 10)
    print("{} have finished!".format(sequence))

def detection_res_on_bev(sequence):
    """
    将检测结果投影到BEV视角并生成视频，不同类别使用不同颜色
    """
    # 定义类别颜色映射（参考data_show.py中的颜色定义）
    OBJECT_PALETTE = {
        "car":          (255, 0, 0),        # 红色
        "truck":        (0, 255, 0),        # 绿色
        "bus":          (0, 255, 255),      # 青色
        "rider":        (255, 255, 0),      # 黄色
        "bicycle":      (255, 0, 255),      # 洋红
        "person":       (0, 0, 255),        # 蓝色
        "motorcycle":   (255, 105, 180),    # 热粉色
        "NotUsed":      (160, 32, 240),     # 紫色
    }
    
    # 类别名称列表（与数据集中的类别对应）
    name_combile_list = [
        'car',        # 0
        'truck',      # 1   
        'bus',        # 2
        'rider',      # 3
        'bicycle',    # 4
        'person',     # 5
        'motorcycle', # 6
        'NotUsed'     # 7
    ]
    
    # 类别名称到索引的映射
    name_old_dict = {
        'car': 0,
        'truck': 1,
        'bus': 2,
        'bike': 4,
        'bicycle': 4,
        'person': 5,
        'motor': 6,
        'motorcycle': 6,
        'NotUsed': 7,
        'rider': 3,
        'pedestrain': 5,
        'pedestrian': 5
    }

    xlim = ylim = [-80, 80]
    radius = 15
    linewidth = 10

    # 点云路径
    pcd_root = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track", sequence, "samples/lidar")
    pcd_list = [os.path.join(pcd_root, pcd_file) for pcd_file in os.listdir(pcd_root)]
    pcd_list = sorted(pcd_list)
    nums = len(pcd_list)

    # 检测结果路径
    detection_root = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track", sequence, "model_pred")
    detection_files = [f for f in os.listdir(detection_root) if f.endswith('.txt')]
    detection_files = sorted(detection_files)
    
    # 存储每帧的检测框
    detection_occ = defaultdict(list)
    
    # 处理每一帧的检测结果
    for i, det_file in enumerate(tqdm(detection_files, desc="Processing detection files")):
        det_path = os.path.join(detection_root, det_file)
        detections = read_detection_txt_with_label(det_path)  # 使用新的读取函数
        
        # 为每个检测框根据类别分配颜色
        for params, label_name in detections:
            # 获取类别对应的颜色
            if label_name in OBJECT_PALETTE:
                color_bgr = OBJECT_PALETTE[label_name]
                # 转换BGR到RGB并归一化到0-1范围
                color_normalized = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)  # BGR转RGB并归一化
            else:
                # 如果类别未定义，使用默认颜色（灰色）
                color_normalized = (0.5, 0.5, 0.5)
            
            pts = draw_detection_box_on_pcd(params)
            detection_occ[i].append((pts, color_normalized, label_name))  # 添加类别名称用于图例
    
    # 生成BEV图像
    for i in tqdm(range(min(nums, len(detection_files))), desc="Generating BEV images"):
        # 加载点云数据
        lidar_pts = np.fromfile(pcd_list[i], dtype=np.float32).reshape(-1, 5)
        img_name = pcd_list[i].split('/')[-1].replace('pcd.bin', 'jpg')
        save_path = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all", sequence, "detection_bev", img_name)
        
        # 创建图像
        fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
        ax = plt.gca()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.set_axis_off()
        
        # 绘制点云
        colors = plt.get_cmap("rainbow")((lidar_pts[:, 3]-lidar_pts[:, 3].min())/(lidar_pts[:, 3].max()-lidar_pts[:, 3].min()))
        if lidar_pts is not None:
            plt.scatter(
                lidar_pts[:, 0],
                lidar_pts[:, 1],
                s=radius,
                c=colors[:, :3],
            )
        
        # 绘制检测框
        drawn_labels = set()  # 用于避免图例重复
        for bbox, color_, label_name in detection_occ[i]:
            ax.plot(bbox[:, 0], bbox[:, 1], color=color_, linewidth=linewidth, linestyle='-')
            drawn_labels.add(label_name)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = []
        for label_name in drawn_labels:
            if label_name in OBJECT_PALETTE:
                color_bgr = OBJECT_PALETTE[label_name]
                color_rgb_normalized = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)
                legend_elements.append(
                    Line2D([0], [0], color=color_rgb_normalized, linewidth=linewidth, linestyle='-', label=label_name)
                )
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(
            save_path,
            dpi=10,
            facecolor="black",
            format="jpg",
            bbox_inches="tight",
            pad_inches=0,
        )
        ax.clear()
        plt.close(fig)
        
    # 生成视频
    image_folder = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all", sequence, "detection_bev")
    save_video_path = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all/video_res", sequence + "_detection_bev.mp4")
    images_to_video_ffmpeg(image_folder, save_video_path, 10)
    print("{} detection BEV visualization have finished!".format(sequence))

def read_detection_txt_with_label(file_path):
    """
    读取检测结果txt文件，同时返回标签名称
    返回: [([h, w, l, x, y, z+(h/2.0), np.degrees(yaw), score_id], label_name), ...]
    """
    boxes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) >= 9:  # 确保有足够的值
                    # label h w l x y z+(h/2.0) math.degrees(yaw) score_id
                    label = vals[0]
                    h, w, l, x, y, z_offset, yaw_deg, score_id = map(float, vals[1:9])
                    boxes.append(([h, w, l, x, y, z_offset, yaw_deg, score_id], label))
    except Exception as e:
        print(f"Error reading detection file {file_path}: {e}")
    return boxes


def read_detection_txt_with_yaw(file_path):
    """
    读取检测结果txt文件，同时返回参数和yaw角度
    返回: [([h, w, l, x, y, z+(h/2.0), np.degrees(yaw), score_id], yaw_deg), ...]
    """
    boxes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) >= 9:  # 确保有足够的值
                    # label h w l x y z+(h/2.0) math.degrees(yaw) score_id
                    label = vals[0]
                    h, w, l, x, y, z_offset, yaw_deg, score_id = map(float, vals[1:9])
                    boxes.append(([h, w, l, x, y, z_offset, yaw_deg, score_id], yaw_deg))
    except Exception as e:
        print(f"Error reading detection file {file_path}: {e}")
    return boxes

def draw_detection_box_on_pcd_with_yaw(params, yaw_deg):
    """
    绘制检测框到点云图上
    params: [h, w, l, x, y, z, yaw_deg, score_id]
    """
    h, w, l, x, y, z, _, score_id = params  # yaw_deg已作为单独参数传入
    # 转换角度从度到弧度
    yaw = np.radians(yaw_deg)
    # 调整z坐标，因为输入的是z+(h/2.0)
    z = z - (h / 2.0)
    
    corners_3d_obj = np.array([
        [-l / 2, -w / 2, -h / 2],
        [l / 2,  -w / 2, -h / 2],
        [l / 2,   w / 2, -h / 2],
        [-l / 2,  w / 2, -h / 2],
        [-l / 2, -w / 2,  h / 2],
        [l / 2,  -w / 2,  h / 2],
        [l / 2,   w / 2,  h / 2],
        [-l / 2,  w / 2,  h / 2]
    ])
    corners_3d_obj_roation = rotate_points(corners_3d_obj.T, yaw)
    corners_3d = np.array([[x, y, z]]).T + corners_3d_obj_roation
    corners_3d = corners_3d.T[:5, :2]
    corners_3d[4, :] = corners_3d[0, :]

    return corners_3d


def combined_res_on_bev(sequence):
    """
    将检测结果和跟踪结果同时投影到BEV视角并生成视频
    检测结果用白色表示，跟踪结果用红色表示
    同时显示每个框的yaw角
    """
    xlim = ylim = [-80, 80]
    radius = 15
    detection_linewidth = 20.0   # 检测框线宽
    track_linewidth = 15.0       # 跟踪框线宽，更粗一些
    text_size = 100               # yaw角文字大小（修正为合理值）
    text_color_detection = 'white'
    text_color_tracking = 'yellow'  # 改变跟踪yaw角颜色以提高对比度
    
    # 点云路径
    pcd_root = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track", sequence, "samples/lidar")
    pcd_list = [os.path.join(pcd_root, pcd_file) for pcd_file in os.listdir(pcd_root)]
    pcd_list = sorted(pcd_list)
    nums = len(pcd_list)

    # 检测结果路径
    detection_root = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track", sequence, "model_pred")
    detection_files = [f for f in os.listdir(detection_root) if f.endswith('.txt')]
    detection_files = sorted(detection_files)
    
    # 跟踪结果路径
    seq_path = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track/offline_tracked", sequence)
    scene_name_list = sorted(os.listdir(seq_path))
    scene_list = [os.path.join(seq_path, scene) for scene in scene_name_list]
    
    # 存储每帧的检测框和跟踪框
    detection_occ = defaultdict(list)
    track_occ = defaultdict(list)
    
    # 处理每一帧的检测结果
    for i, det_file in enumerate(tqdm(detection_files, desc="Processing detection files")):
        det_path = os.path.join(detection_root, det_file)
        detections = read_detection_txt_with_yaw(det_path)  # 使用新的读取函数获取yaw角
        
        # 为每个检测框分配颜色 (使用白色表示检测结果)
        for params, yaw_deg in detections:
            color_normalized = (1, 1, 1)  # 白色 (RGB)
            pts = draw_detection_box_on_pcd_with_yaw(params, yaw_deg)
            detection_occ[i].append((pts, color_normalized, yaw_deg))  # 添加yaw角信息
    
    # 处理跟踪结果
    count_num = 0
    for file_path in scene_list:
        res_file_root = os.path.join(file_path, "tracking")
        split_file_root = os.path.join(file_path, "splited")
        res_file = [os.path.join(res_file_root, file) for file in os.listdir(res_file_root) if file.startswith("tracking-test")][0]
        with open(res_file, "rb") as file:
            data = pickle.load(file)
            res_data = data[next(iter(data.keys()))] 

        for key, value in res_data.items():
            # 为每个跟踪对象分配颜色 (使用红色表示跟踪结果)
            color_normalized = (1, 0, 0)  # 红色 (RGB)
            for idx, params in zip(value['sample_idx'].tolist(), value['boxes_global']):
                idx = int(idx) + count_num
                pts = draw_track_box_on_pcd(params)
                # 从参数中提取yaw角（第7个参数）
                yaw = params[6]  # params格式: [x, y, z, l, w, h, yaw, _, _]
                yaw_deg = np.degrees(yaw)  # 转换为度 yaw #
                track_occ[idx].append((pts, color_normalized, yaw_deg, key))  # 添加yaw角信息和目标ID
        count_num += len(os.listdir(split_file_root))
    
    # 生成BEV图像
    for i in tqdm(range(min(nums, len(detection_files))), desc="Generating combined BEV images"):
        # 加载点云数据
        lidar_pts = np.fromfile(pcd_list[i], dtype=np.float32).reshape(-1, 5)
        img_name = pcd_list[i].split('/')[-1].replace('pcd.bin', 'jpg')
        save_path = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all", sequence, "combined_bev", img_name)
        
        # 创建图像
        fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
        ax = plt.gca()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.set_axis_off()
        
        # 绘制点云
        colors = plt.get_cmap("rainbow")((lidar_pts[:, 3]-lidar_pts[:, 3].min())/(lidar_pts[:, 3].max()-lidar_pts[:, 3].min()))
        if lidar_pts is not None:
            plt.scatter(
                lidar_pts[:, 0],
                lidar_pts[:, 1],
                s=radius,
                c=colors[:, :3],
            )
        
        # 先绘制所有检测框和跟踪框
        box_plots = []  # 存储所有框的绘制对象
        for bbox, color_, yaw_deg in detection_occ[i]:
            line, = ax.plot(bbox[:, 0], bbox[:, 1], color=color_, linewidth=detection_linewidth, linestyle='--')
            box_plots.append(line)
            
        for bbox, color_, yaw_deg, obj_id in track_occ[i]:
            line, = ax.plot(bbox[:, 0], bbox[:, 1], color=color_, linewidth=track_linewidth, linestyle='-')
            box_plots.append(line)
        
        # 然后在所有框绘制完成后再绘制文本，避免遮挡
        for bbox, color_, yaw_deg in detection_occ[i]:
            center_x = np.mean(bbox[:-1, 0])  # 排除最后一个点（与第一个点重复）
            center_y = np.mean(bbox[:-1, 1])
            ax.text(center_x+10, center_y, f'{yaw_deg:.1f}°', 
                   color=text_color_detection, fontsize=text_size, 
                   ha='center', va='center',
                   weight='bold',
                   # 使用半透明背景以减少遮挡
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='black', edgecolor='white', alpha=0.6),
                   zorder=10)  # 设置较高的zorder确保文本在上层

        for bbox, color_, yaw_deg, obj_id in track_occ[i]:
            center_x = np.mean(bbox[:-1, 0])  # 排除最后一个点（与第一个点重复）
            center_y = np.mean(bbox[:-1, 1])
            
            # 在yaw角上方显示目标ID
            ax.text(center_x, center_y + 3, f'ID:{obj_id}', 
                   color='cyan', fontsize=text_size/2,  # ID文字大小为yaw角的一半
                   ha='center', va='center',
                   weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='black', edgecolor='cyan', alpha=0.7),
                   zorder=11)  # 更高的zorder确保ID在最上层
            
            # 显示yaw角
            ax.text(center_x, center_y, f'{yaw_deg:.1f}°', 
                   color=text_color_tracking, fontsize=text_size, 
                   ha='center', va='center',
                   weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='black', edgecolor='yellow', alpha=0.6),
                   zorder=10)  # 设置较高的zorder确保文本在上层
        
        # 添加图例说明
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='white', linewidth=detection_linewidth, linestyle='--', label='Detection'),
            Line2D([0], [0], color='red', linewidth=track_linewidth, linestyle='-', label='Tracking')
        ]
        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
        # 设置图例背景以确保一致性
        legend.get_frame().set_facecolor('black')
        legend.get_frame().set_alpha(0.8)
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(
            save_path,
            dpi=10,
            facecolor="black",
            format="jpg",
            bbox_inches=None,
            pad_inches=0.0,
        )
        ax.clear()
        plt.close(fig)
        
    # 生成视频
    image_folder = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all", sequence, "combined_bev")
    save_video_path = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all/video_res", sequence + "_combined_bev.mp4")
    images_to_video_ffmpeg_fixed_size(image_folder, save_video_path, 10)
    print("{} combined BEV visualization have finished!".format(sequence))
def images_to_video_ffmpeg_fixed_size(image_folder, output_video, fps=10):
    """
    修改后的视频生成函数，确保视频尺寸兼容
    """
    files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
    
    with open("filelist.txt", "w") as f:
        for file in files:
            f.write(f"file '{os.path.join(image_folder, file)}'\n")
    
    # 使用scale参数确保输出尺寸是偶数，避免编码错误
    cmd = [
        "/usr/bin/ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", "filelist.txt", 
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-vf", "scale=1280:1280:force_original_aspect_ratio=decrease,pad=1280:1280:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-y",
        output_video
    ]
    
    subprocess.run(cmd)
    os.remove("filelist.txt")  # 清理临时文件



def visualize_yaw_changes(sequence):
    """
    为特定目标创建yaw角随时间变化的图表，突出显示最大角度变化
    """
    # 跟踪结果路径
    seq_path = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track/offline_tracked", sequence)
    scene_name_list = sorted(os.listdir(seq_path))
    scene_list = [os.path.join(seq_path, scene) for scene in scene_name_list]
    
    # 点云路径，用于获取时间戳信息
    pcd_root = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track", sequence, "samples/lidar")
    pcd_files = sorted([f for f in os.listdir(pcd_root) if f.endswith('.pcd.bin')])
    
    # 创建帧号到时间戳的映射
    frame_to_timestamp = {}
    for i, pcd_file in enumerate(pcd_files):
        # 假设文件名格式为 xxxxxxxx.pcd.bin，其中xxxxxxxx是时间戳
        timestamp_str = pcd_file.replace('.pcd.bin', '')
        try:
            timestamp = float(timestamp_str)
        except ValueError:
            # 如果无法转换为浮点数，则使用索引作为时间戳
            timestamp = float(i)
        frame_to_timestamp[i] = timestamp
    
    # 存储每个目标的yaw角变化
    object_yaw_data = defaultdict(list)
    
    # 处理跟踪结果
    count_num = 0
    for file_path in scene_list:
        res_file_root = os.path.join(file_path, "tracking")
        split_file_root = os.path.join(file_path, "splited")
        res_file = [os.path.join(res_file_root, file) for file in os.listdir(res_file_root) if file.startswith("tracking-test")][0]
        with open(res_file, "rb") as file:
            data = pickle.load(file)
            res_data = data[next(iter(data.keys()))] 

        for key, value in res_data.items():
            for idx, params in zip(value['sample_idx'].tolist(), value['boxes_global']):
                idx = int(idx) + count_num
                # 从参数中提取yaw角（第7个参数）
                yaw = params[6]  # params格式: [x, y, z, l, w, h, yaw, _, _]
                yaw_deg = np.degrees(yaw)  # 转换为度
                object_yaw_data[key].append((idx, yaw_deg))
        count_num += len(os.listdir(split_file_root))
    
    # 为每个目标生成yaw角变化图
    for obj_id, yaw_data in object_yaw_data.items():
        if len(yaw_data) > 1:  # 只绘制存在多帧的目标
            # 按帧索引排序
            yaw_data.sort(key=lambda x: x[0])
            frames, yaws = zip(*yaw_data)
            
            # 计算连续帧之间的角度变化
            yaw_changes = []
            max_change = 0
            max_change_frames = None
            
            for i in range(1, len(yaws)):
                # 计算角度差值，并处理跨越360度的情况
                diff = yaws[i] - yaws[i-1]
                # 处理角度环绕问题（例如从359度到1度应该是2度的变化，而不是-358度）
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                abs_diff = abs(diff)
                yaw_changes.append(abs_diff)
                
                # 记录最大变化
                if abs_diff > max_change:
                    max_change = abs_diff
                    max_change_frames = (frames[i-1], frames[i])
            
            # 创建yaw角变化图
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # 绘制yaw角变化曲线
            line, = ax.plot(frames, yaws, marker='o', linewidth=2, markersize=6, color='blue', label='Yaw Angle')
            
            # 标记最大变化点
            if max_change_frames:
                idx1 = frames.index(max_change_frames[0])
                idx2 = frames.index(max_change_frames[1])
                ax.scatter([max_change_frames[0], max_change_frames[1]], 
                          [yaws[idx1], yaws[idx2]], 
                          color='red', s=100, zorder=5, label=f'Max Change Frames')
                
                # 获取时间戳
                ts1 = frame_to_timestamp.get(max_change_frames[0], float(max_change_frames[0]))
                ts2 = frame_to_timestamp.get(max_change_frames[1], float(max_change_frames[1]))
                
                # 在图上添加文本说明
                ax.annotate(f'Max Change: {max_change:.2f}°\nFrame {max_change_frames[0]} → {max_change_frames[1]}\nTime: {ts1:.6f} → {ts2:.6f}', 
                           xy=(max_change_frames[1], yaws[idx2]), 
                           xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.set_xlabel('Frame')
            ax.set_ylabel('Yaw Angle (degrees)')
            
            # 设置标题，包含时间信息
            if max_change_frames:
                ts1 = frame_to_timestamp.get(max_change_frames[0], float(max_change_frames[0]))
                ts2 = frame_to_timestamp.get(max_change_frames[1], float(max_change_frames[1]))
                title = f'Object {obj_id} Yaw Angle Changes\nMax Change: {max_change:.2f}° from Frame {max_change_frames[0]} to {max_change_frames[1]}\nTime: {ts1:.6f} to {ts2:.6f}'
            else:
                title = f'Object {obj_id} Yaw Angle Changes'
            ax.set_title(title)
            
            ax.grid(True, alpha=0.3)
            
            # 设置x轴刻度间隔为1
            ax.set_xticks(range(min(frames), max(frames) + 1, 1))
            ax.tick_params(axis='x', rotation=45)
            
            # 添加图例
            ax.legend()
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图像
            save_dir = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all", sequence, "yaw_analysis")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"object_{obj_id}_yaw_changes.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 打印统计信息
            print(f"Object {obj_id}:")
            print(f"  Total frames: {len(frames)}")
            print(f"  Frame range: {min(frames)} - {max(frames)}")
            print(f"  Yaw range: {min(yaws):.2f}° - {max(yaws):.2f}°")
            if max_change_frames:
                ts1 = frame_to_timestamp.get(max_change_frames[0], float(max_change_frames[0]))
                ts2 = frame_to_timestamp.get(max_change_frames[1], float(max_change_frames[1]))
                print(f"  Max change: {max_change:.2f}° from frame {max_change_frames[0]} to {max_change_frames[1]}")
                print(f"  Time: {ts1:.6f} to {ts2:.6f}")
            print(f"  Average change per frame: {np.mean(yaw_changes):.2f}°")
            print(f"  Std deviation: {np.std(yaws):.2f}°")
            print("-" * 50)


def visualize_yaw_changes_summary(sequence):
    """
    生成所有目标yaw角变化的汇总图
    """
    # 跟踪结果路径
    seq_path = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track/offline_tracked", sequence)
    scene_name_list = sorted(os.listdir(seq_path))
    scene_list = [os.path.join(seq_path, scene) for scene in scene_name_list]
    
    # 点云路径，用于获取时间戳信息
    pcd_root = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/dataset_track", sequence, "samples/lidar")
    pcd_files = sorted([f for f in os.listdir(pcd_root) if f.endswith('.pcd.bin')])
    
    # 创建帧号到时间戳的映射
    frame_to_timestamp = {}
    for i, pcd_file in enumerate(pcd_files):
        # 假设文件名格式为 xxxxxxxx.pcd.bin，其中xxxxxxxx是时间戳
        timestamp_str = pcd_file.replace('.pcd.bin', '')
        try:
            timestamp = float(timestamp_str)
        except ValueError:
            # 如果无法转换为浮点数，则使用索引作为时间戳
            timestamp = float(i)
        frame_to_timestamp[i] = timestamp
    
    # 存储每个目标的yaw角变化
    object_yaw_data = defaultdict(list)
    
    # 处理跟踪结果
    count_num = 0
    for file_path in scene_list:
        res_file_root = os.path.join(file_path, "tracking")
        split_file_root = os.path.join(file_path, "splited")
        res_file = [os.path.join(res_file_root, file) for file in os.listdir(res_file_root) if file.startswith("tracking-test")][0]
        with open(res_file, "rb") as file:
            data = pickle.load(file)
            res_data = data[next(iter(data.keys()))] 

        for key, value in res_data.items():
            for idx, params in zip(value['sample_idx'].tolist(), value['boxes_global']):
                idx = int(idx) + count_num
                # 从参数中提取yaw角（第7个参数）
                yaw = params[6]  # params格式: [x, y, z, l, w, h, yaw, _, _]
                yaw_deg = np.degrees(yaw)  # 转换为度
                object_yaw_data[key].append((idx, yaw_deg))
        count_num += len(os.listdir(split_file_root))
    
    # 计算每个目标的最大角度变化
    object_max_changes = {}
    for obj_id, yaw_data in object_yaw_data.items():
        if len(yaw_data) > 1:  # 只考虑存在多帧的目标
            # 按帧索引排序
            yaw_data.sort(key=lambda x: x[0])
            frames, yaws = zip(*yaw_data)
            
            # 计算连续帧之间的角度变化
            max_change = 0
            max_change_frames = None
            
            for i in range(1, len(yaws)):
                # 计算角度差值，并处理跨越360度的情况
                diff = yaws[i] - yaws[i-1]
                # 处理角度环绕问题
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                abs_diff = abs(diff)
                
                # 记录最大变化
                if abs_diff > max_change:
                    max_change = abs_diff
                    max_change_frames = (frames[i-1], frames[i])
            
            object_max_changes[obj_id] = {
                'max_change': max_change,
                'frames': max_change_frames,
                'total_frames': len(frames),
                'yaw_range': (min(yaws), max(yaws)),
                'timestamps': (
                    frame_to_timestamp.get(max_change_frames[0], float(max_change_frames[0])) if max_change_frames else None,
                    frame_to_timestamp.get(max_change_frames[1], float(max_change_frames[1])) if max_change_frames else None
                )
            }
    
    # 创建汇总图
    if object_max_changes:
        # 按最大变化排序
        sorted_objects = sorted(object_max_changes.items(), key=lambda x: x[1]['max_change'], reverse=True)
        
        obj_ids = [str(item[0]) for item in sorted_objects]
        max_changes = [item[1]['max_change'] for item in sorted_objects]
        
        fig, ax = plt.subplots(figsize=(15, 8))
        bars = ax.bar(range(len(obj_ids)), max_changes, color='skyblue')
        
        # 在柱状图上添加数值标签
        for i, (obj_id, change) in enumerate(zip(obj_ids, max_changes)):
            ax.text(i, change + 0.5, f'{change:.2f}°', ha='center', va='bottom')
        
        ax.set_xlabel('Object ID')
        ax.set_ylabel('Maximum Yaw Change (degrees)')
        ax.set_title('Maximum Yaw Angle Change per Object')
        ax.set_xticks(range(len(obj_ids)))
        ax.set_xticklabels(obj_ids, rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存汇总图
        save_dir = os.path.join("/data1/turbo_data/lishuaiyin/4D_label/vis_all", sequence, "yaw_analysis")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "yaw_changes_summary.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 打印汇总信息
        print("\n=== Yaw Change Summary ===")
        print(f"Total objects tracked: {len(object_max_changes)}")
        print("\nTop 5 objects with largest yaw changes:")
        for i, (obj_id, data) in enumerate(sorted_objects[:10]):
            frames_info = f"Frame {data['frames'][0]} → {data['frames'][1]}" if data['frames'] else "N/A"
            time_info = f"Time: {data['timestamps'][0]:.6f} → {data['timestamps'][1]:.6f}" if data['timestamps'][0] and data['timestamps'][1] else "N/A"
            print(f"  {i+1}. Object {obj_id}: {data['max_change']:.2f}° ({frames_info})")
            print(f"     {time_info}")
            print(f"     Total frames: {data['total_frames']}, Yaw range: {data['yaw_range'][0]:.1f}° - {data['yaw_range'][1]:.1f}°")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)

    args = parser.parse_args()
    
    # seq_res_on_bev(args.sequence, False)
    # detection_res_on_bev(args.sequence)
    combined_res_on_bev(args.sequence)
    visualize_yaw_changes(args.sequence)  # 生成每个目标的详细图表
    visualize_yaw_changes_summary(args.sequence)  # 生成汇总图表


    