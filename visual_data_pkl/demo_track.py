import os
import json
import cv2
import math
import pickle
import numpy as np
from tqdm import tqdm
import subprocess

import sys
sys.path.insert(0, "/data1/turbo_data/yuanqingwen/4D_label")
from visual_data.utils.nms import nms_angle
from visual_data.utils.visualize import _COLORS



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


def images_to_video_ffmpeg(image_folder, output_video, fps=10):

        files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
        with open("filelist.txt", "w") as f:
            for file in files:
                f.write(f"file '{os.path.join(image_folder, file)}'\n")
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", "filelist.txt",
            "-r", str(fps),
            "-c:v", "mpeg4",       # 改用MPEG-4编码
            "-q:v", "2",           # 质量参数(1-31)
            "-pix_fmt", "yuv420p",
            "-vf", "scale=iw:ih",
            "-y",
            output_video
        ]
        subprocess.run(cmd)
        os.remove("filelist.txt")  # 清理临时文件


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
                cv2.rectangle(canvas, (top_left_u, top_left_v), (bottom_right_u, bottom_right_v), color_bgr, -1)
                cv2.putText(canvas, name, (top_left_u + color_width + horizontal_gap, bottom_right_v), font, font_scale, font_color, thickness)
            else:
                cv2.putText(canvas, f"{name}: ", (top_left_u, bottom_right_v), font, font_scale, font_color, thickness)
            i += 1

    return canvas

def rotate_points(points, yaw):
    """
    将给定的3D点绕y轴旋转指定的角度（yaw）

    参数:
    points: 形状为 (n, 3) 的numpy数组，表示n个3D点坐标 (x, y, z)
    yaw: 绕y轴旋转的角度（弧度制）

    返回:
    rotated_points: 形状为 (n, 3) 的numpy数组，表示旋转后的3D点坐标
    """
    # yaw = math.pi
    # yaw = -1*yaw - math.pi/2.0
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return rotation_matrix @ points

def draw_3d_box(image, corners_2d, track_id, category, color=(0, 255, 0)):
    """
    在给定的图像上绘制3D框的2D投影

    参数:
    image: 要绘制的OpenCV图像（numpy数组形式）
    corners_2d: 形状为 (8, 2) 的numpy数组，表示3D框8个顶点投影后的2D坐标
    color: 绘制框的颜色，默认为绿色（BGR格式），可以根据需求修改

    返回:
    image: 绘制好3D框投影的图像
    """
    # 绘制底面四条边
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[(i + 1) % 4].astype(int)), color, 2)
    # 绘制顶面四条边
    for i in range(4, 8):
        cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[(i + 1) % 4 + 4].astype(int)), color, 2)
    # 绘制竖边
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[i + 4].astype(int)), color, 2)
    # ## 绘制头部
    # # color_head=(color[0]/2.0,color[1]/2.0,color[2]/2.0)
    # color_head = (255,255,255)
    # cv2.line(image, tuple(corners_2d[1].astype(int)), tuple(corners_2d[5].astype(int)), color_head, 6)# 底部
    # cv2.line(image, tuple(corners_2d[2].astype(int)), tuple(corners_2d[6].astype(int)), color_head, 6)# 顶部
    # cv2.line(image, tuple(corners_2d[1].astype(int)), tuple(corners_2d[2].astype(int)), color_head, 6)# 左边
    # cv2.line(image, tuple(corners_2d[5].astype(int)), tuple(corners_2d[6].astype(int)), color_head, 6)# 顶部
    text = '{}'.format(track_id)
    # text = '{} {}'.format(track_id, category)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
    cv2.putText(image, text, (corners_2d[4][0], corners_2d[4][1] + txt_size[1]), font, 0.4, (0, 0, 255), thickness=2)
    return image


def project_3d_to_pixel(points, image_shape, intrinsic_matrix, extrinsic_matrix, distortion_params, filter_z_camera=True, depth=None):
    # 将3D点坐标构建成齐次坐标形式
    x, y, z = points
    point_3d_homogeneous = np.array([[x], [y], [z], [1]])
    point_camera_coords = np.dot(extrinsic_matrix[:3, :], point_3d_homogeneous)
    
    if distortion_params is not None:
        k1, k2, k3, k4 = distortion_params
        # 通过外参矩阵将3D点从世界坐标系转换到相机坐标系
        # 计算归一化平面坐标（未考虑畸变）
        x_camera, y_camera, z_camera = point_camera_coords.flatten()[:3]
        if filter_z_camera and z_camera < 0:
            return None, None, False
        if depth is not None and z_camera < depth:
            return None, None, False
        normalized_x = x_camera / z_camera
        normalized_y = y_camera / z_camera
        # 考虑畸变，使用畸变模型进行坐标矫正
        r_squared = np.sqrt(normalized_x ** 2 + normalized_y ** 2)
        theta = np.arctan(r_squared)
        theda_d = theta * (1 + k1 * np.power(theta, 2) + k2 * np.power(theta, 4) + k3 * np.power(theta, 6) + k4 * np.power(theta, 8))
        distorted_x = (theda_d / r_squared) * normalized_x
        distorted_y = (theda_d / r_squared) * normalized_y
        # 通过内参矩阵将矫正后的坐标从归一化平面投影到图像平面
        # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix, np.array(distortion_params), (1024, 1024), np.eye(3), balance=1.0)
        projected_point = np.dot(intrinsic_matrix, np.array([[distorted_x], [distorted_y], [1]]))
        u = projected_point[0, 0]
        v = projected_point[1, 0]
    else:
        if point_camera_coords[2, 0] < 0:
            return None, None, False
        points_image_ = intrinsic_matrix @ point_camera_coords
        points_image = points_image_ / points_image_[2,:]
        u = points_image[0, 0]
        v = points_image[1, 0]
    
    if not (u > 0 and u < image_shape[1] and v > 0 and v < image_shape[1]):
        return u, v, False
    return u, v, True

def draw_box_on_img(params, image, cam2pixel, lidar2cam, track_id, category, distort=None, color=[255,0,0]):
    x, y, z, l, w, h, yaw, _, _ = params
    # z -= h/2
    # 构建3D框的8个顶点坐标（在物体坐标系下，原点在物体中心）
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
    corners_3d_rotation_points = np.array([[x, y, z]]).T + corners_3d_obj_roation
    corners_2d = []
    image_shape = image.shape
    valid_flag = []
    for points in corners_3d_rotation_points.T:
        u, v, vali = project_3d_to_pixel(points, image_shape, cam2pixel, lidar2cam, distort)
        if u is None and u is None:
            return image, None
        if vali:
            valid_flag.append(1)
        corners_2d.append([u, v])
    if len(valid_flag) < 1:
        return image, None
    corners_2d = np.array(corners_2d)
    image = draw_3d_box(image, corners_2d.astype(int), track_id, category, color=color)
    return image, corners_2d


if __name__ == '__main__':

    track_file = "/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/offline_tracked/train_hy_4d_road_7_20250218_lx/20250218070018/track.pkl"
    with open(track_file, "rb")as file:
        track_data = pickle.load(file)

    res_file = "/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/offline_tracked/train_hy_4d_road_7_20250218_lx/20250218070018/tracking/tracking-test-20250310-145203.pkl"
    with open(res_file, "rb")as file:
        data = pickle.load(file)
        res_data = data['train_hy_4d_road_7_20250218_lx_20250218070018']

    # drop_file = "/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/offline_tracked/train_hy_4d_road_7_20250218_lx/20250218070018/tracking/drop-test-20250310-145203.pkl"
    # with open(drop_file, "rb")as file:
    #     data = pickle.load(file)
    #     drop_data = data['train_hy_4d_road_7_20250218_lx_20250218070018']

    test_json = "/data1/turbo_data/yuanqingwen/data/test_4D_label/model_res/bevpro/train_hy_4d_road_7_20250218_lx/scences/test.json"
    with open(test_json, "r")as f:
        json_data = json.load(f)[0]
    cam2pixels = json_data['cam2img']
    cam2pixels.update(json_data['cam2img_fisheye']) 
    lidar2cams = json_data['lidar2cam']
    lidar2cams.update(json_data['lidar2cam_fisheye'])
    camera_distort_param = json_data['distort_fisheye']
    sequence, scene = res_file.split('/')[-4], res_file.split('/')[-3]
    img_root_path = os.path.join("/data1/turbo_data/4D_label_dataset/origin", sequence, scene)
    cam_list = [os.path.join(img_root_path, cam) for cam in os.listdir(img_root_path) if cam.startswith('camera')]
    cam_pinhole = [cam for cam in cam_list if cam.endswith('_0') or cam.endswith('_8')]

    for idx, (track_id, obj) in enumerate(res_data.items()):
        """
        No.track_id obj 
        obj.keys() = ['boxes_global', 'name', 'score', 'sample_idx', 'hit', 'num_points', 'obj_ids', 'pose', 'state']
            boxes_global-->(frames_num, 9), [center_x, center_y, center_z, length, witdh, height, yaw(radians), vx, vy]
                !!!! yaw in res_data != yaw in track.pkl
            name-->obj category label (frames_num, )
            sample_idx-->sample in which frames
            hit--> 0 / 1, coorespond sample_idx
            state--> static or dynamic
        """
        category_label = obj['name'][0]
        state = obj['state']
        imgs_list = sorted(os.listdir(cam_list[0]))
        for frame, params in tqdm(zip(obj['sample_idx'], obj['boxes_global'])):
            for cam in cam_pinhole:
                cam_name = cam.split('/')[-1]
                fisheye = True if cam.endswith('_8') else False
                distort = camera_distort_param[cam_name] if fisheye else None
                img_path = os.path.join(cam, imgs_list[int(frame)])
                img = cv2.imread(img_path)
                cam2pixel = np.array(cam2pixels[cam_name])[:3, :3]
                lidar2cam = np.array(lidar2cams[cam_name])
                color = (_COLORS[idx] * 255 * 0.7).astype(np.uint8).tolist()
                image, _ = draw_box_on_img(params, img.copy(), cam2pixel, lidar2cam, track_id, category_label, distort=distort, color=color)
                save_img_path = img_path.replace('4D_label_dataset/origin', 'yuanqingwen/data/test_4D_label/origin/model_res/vis').replace('/camera', '/'+str(track_id)+'/camera')
                # save_img_path = img_path.replace('4D_label_dataset/origin', 'yuanqingwen/data/test_4D_label/origin/model_res/vis').replace('/camera', '/multi/camera')
                os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
                cv2.imwrite(save_img_path, image)
            breakpoint()
        for cam in cam_pinhole:
            cam_name = cam.split('/')[-1]
            image_folder = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis/train_hy_4d_road_7_20250218_lx/20250218070018", str(track_id))
            # image_folder = "/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis/train_hy_4d_road_7_20250218_lx/20250218070018/multi"
            save_video_path = os.path.join(image_folder, cam_name+".mp4")
            images_to_video_ffmpeg(os.path.join(image_folder, cam_name), save_video_path)
        print("No.{} obj have finished!".format(track_id))
        breakpoint()

    for frame in track_data:
        """
        frame.keys() = ['name', 'boxes_lidar', 'frame_id', 'pose', 'score', 'sequence_name']
            name-->obj category label (objs_num, )
            boxes_lidar-->(objs_num, 9), [center_x, center_y, center_z, length, witdh, height, yaw, score, 0]
                yaw in track.pkl != yaw in splited.txt

            score-->(objs_num, )
        """
        pass
        breakpoint()

