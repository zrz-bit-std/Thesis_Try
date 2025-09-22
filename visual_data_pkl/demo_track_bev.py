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

def project_3d_to_pixel(points, image_shape, intrinsic_matrix, extrinsic_matrix, distortion_params, filter_z_camera=True, depth=None):
    x, y, z = points
    point_3d_homogeneous = np.array([[x], [y], [z], [1]])
    point_camera_coords = np.dot(extrinsic_matrix[:3, :], point_3d_homogeneous)
    
    if distortion_params is not None:
        k1, k2, k3, k4 = distortion_params
        x_camera, y_camera, z_camera = point_camera_coords.flatten()[:3]
        if filter_z_camera and z_camera < 0:
            return None, None, False
        if depth is not None and z_camera < depth:
            return None, None, False
        normalized_x = x_camera / z_camera
        normalized_y = y_camera / z_camera
        r_squared = np.sqrt(normalized_x ** 2 + normalized_y ** 2)
        theta = np.arctan(r_squared)
        theda_d = theta * (1 + k1 * np.power(theta, 2) + k2 * np.power(theta, 4) + k3 * np.power(theta, 6) + k4 * np.power(theta, 8))
        distorted_x = (theda_d / r_squared) * normalized_x
        distorted_y = (theda_d / r_squared) * normalized_y
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

def draw_box_on_img_per(params, image, cam2pixel, lidar2cam, distort=None):
    draw_pts = []
    colors = []
    for param, color in params:
        x, y, z, l, w, h, yaw = param
        # x, y, z, l, w, h, yaw, _, _ = param
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
                break
            if vali:
                valid_flag.append(1)
            corners_2d.append([u, v])
        if len(valid_flag) < 1:
            continue
        corners_2d = np.array(corners_2d)
        if corners_2d.shape[0] == 8:
            draw_pts.append(corners_2d)
            colors.append(color)

    if len(draw_pts) < 1:
        return image
    # breakpoint()
    image = draw_3d_box_multi(image, draw_pts, colors)
    return image

def draw_3d_box_multi(image, corners_2ds, colors):
    for corners_2d, color in zip(corners_2ds, colors):
        color_normalized = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        for i in range(4):
            cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[(i + 1) % 4].astype(int)), color_normalized, 2)
        for i in range(4, 8):
            cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[(i + 1) % 4 + 4].astype(int)), color_normalized, 2)
        for i in range(4):
            cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[i + 4].astype(int)), color_normalized, 2)
    return image

def images_to_video_ffmpeg(image_folder, output_video, fps=10):

        files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
        # files = files[:300]
        # files = files[4940:]
        with open("filelist.txt", "w") as f:
            for file in files:
                f.write(f"file '{os.path.join(image_folder, file)}'\n")
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", "filelist.txt", 
            "-vf", "setpts=3.0*PTS",  ##
            "-r", str(fps),
            "-c:v", "libx264",          # 
            "-preset", "slow",          # 
            "-crf", "23",               # 
            "-pix_fmt", "yuv420p",      # 
            "-movflags", "+faststart",  # 
            # "-vf", "scale=iw:ih",       # 
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

def draw_box_on_pcd(params):
    x, y, z, l, w, h, yaw = params
    # yaw = -1*(math.pi/2.0) - yaw_
    yaw = (yaw + math.pi / 2) % (2 * math.pi)
    # breakpoint()
    # x, y, z, l, w, h, yaw, _, _ = params
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

    pcd_root = os.path.join("/data1/turbo_data/liumengyuan/4D_label/dataset_track", sequence, "samples/lidar")
    pcd_list = [os.path.join(pcd_root, pcd_file) for pcd_file in os.listdir(pcd_root)]
    pcd_list = sorted(pcd_list)
    nums = len(pcd_list)

    seq_path = os.path.join("/data1/turbo_data/liumengyuan/4D_label/dataset_track/offline_tracked", sequence)
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
            # print(max(value['sample_idx']))
            for idx, params in zip(value['sample_idx'].tolist(), value['boxes_global']):
                idx = int(idx) + count_num
                pts = draw_box_on_pcd(params)
                obj_occ[idx].append((pts, color_normalized))
        count_num += len(os.listdir(split_file_root))
    # breakpoint()
    for i in tqdm(range(nums)):
        lidar_pts = np.fromfile(pcd_list[i], dtype=np.float32).reshape(-1, 5)
        img_name = pcd_list[i].split('/')[-1].replace('pcd.bin', 'jpg')
        save_path = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis_all", sequence, "bev", img_name)
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
        
    image_folder = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis_all", sequence, "bev")
    save_video_path = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis_all/video_res", sequence + "_bev.mp4")
    images_to_video_ffmpeg(image_folder, save_video_path, 10)
    print("{} have finished!".format(sequence))

def refine_res_on_bev(sequence, colored):

    xlim = ylim =[-80, 80]
    radius = 15
    linewidth = 10

    # refine_root_eval = "/data1/turbo_data/yuanqingwen/origin_version/DetZero/refining/output/ref_model_cfgs/vehicle_grm_model/test"
    refine_root_eval = "/data1/turbo_data/yuanqingwen/origin_version/DetZero/refining/output/ref_model_cfgs/vehicle_prm_model/test3"
    scene_names = os.listdir(refine_root_eval)

    for scene in scene_names:

        txt_root_path = os.path.join("/data1/turbo_data/yuanqingwen/data/4D_label/dataset_track/merged", sequence)
        txt_root = sorted(glob.glob(os.path.join(txt_root_path, scene, "*.txt")))

        pcd_file_lists = glob.glob(os.path.join("/data1/turbo_data/yuanqingwen/data/4D_label/dataset_track", sequence, "samples/lidar", "*.bin"))
        pcd_dict = {
            os.path.splitext(os.path.basename(file))[0]: file 
            for file in pcd_file_lists
        }
        
        nums = len(txt_root)

        obj_occ2 = defaultdict(list)
        obj_res = defaultdict(list)
        # refine_file = os.path.join(refine_root_eval, scene, "eval/epoch_20/test/Vehicle_geometry_test.pkl")
        refine_file = os.path.join(refine_root_eval, scene, "eval/epoch_50/test/Vehicle_position_test.pkl")
        with open(refine_file, "rb")as file:
            data = pickle.load(file)
            res_data = data[next(iter(data.keys()))]
        
        obj_nums = len(res_data.keys())
        objectIDs = list(res_data.keys())
        for res_idx, (key, value) in enumerate(res_data.items()):
            color = (_COLORS[res_idx%len(_COLORS)] * 255).astype(np.uint8).tolist()
            # color = (_COLORS[key%len(_COLORS)] * 255).astype(np.uint8).tolist()
            color_normalized = (color[0]/255, color[1]/255, color[2]/255)
            for idx, params, clss in zip(value['frame_id'], value['boxes_lidar'], value['name']):
                # breakpoint()
                # pts = draw_box_on_pcd(params[0])
                pts = draw_box_on_pcd(params)
                obj_occ2[idx].append(([pts, color_normalized]))
                arr_list = params.tolist()
                # arr_list = params[0].tolist()
                # tensor_val = value['name'][0].item()
                tensor_val = value['name'][0]
                combined = arr_list + [tensor_val]
                obj_res[idx].append(combined)
            # breakpoint()
        # breakpoint()
        for i in tqdm(range(nums)):
            save_txt_file = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/eval_res", sequence, scene, os.path.basename(txt_root[i]))
            mark = os.path.basename(txt_root[i]).replace("txt", "pcd")
            lidar_pts = np.fromfile(pcd_dict[mark], dtype=np.float32).reshape(-1, 5)
            img_name = pcd_dict[mark].split('/')[-1].replace('pcd.bin', 'jpg')
            # save_path = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/refine", sequence, "test", img_name)
            save_path = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/refine_prm", sequence, scene, img_name)

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
            obj_cur = obj_occ2[i]
            for bbox, color_ in obj_cur:
                ax.plot(bbox[:, 0], bbox[:, 1], color=color_, linewidth=linewidth, linestyle='-')
                # ax.plot(bbox[:, 0], bbox[:, 1], color=[0, 0, 1], linewidth=linewidth, linestyle='-')
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

            os.makedirs(os.path.dirname(save_txt_file), exist_ok=True)
            with open(save_txt_file, "w") as f:
                for item in obj_res[i]:
                    f.write(" ".join(map(str, item)) + "\n")
            # breakpoint()

        # breakpoint()
        image_folder = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/refine_prm", sequence, scene)
        save_video_path = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis_all", sequence + "_prm", scene+".mp4")
        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
        images_to_video_ffmpeg(image_folder, save_video_path, 10)
        print("{} have finished!".format(scene))
        # breakpoint()

def seq_res_on_multi_cams(sequence):

    root_path = "/data1/turbo_data/liumengyuan/4D_label/dataset_track"
    image_pinhole_list = ['camera_0_1', 'camera_1_0', 'camera_2_0', 'camera_3_1']
    image_fisheye_list = ['camera_0_8', 'camera_1_8', 'camera_2_8', 'camera_3_8']
    
    pcd_root = os.path.join(root_path, sequence, "samples/lidar")
    pcd_list = [os.path.join(pcd_root, pcd_file) for pcd_file in os.listdir(pcd_root)]
    pcd_list = sorted(pcd_list)
    nums = len(pcd_list)

    seq_path = os.path.join(root_path, "offline_tracked", sequence)
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
                obj_occ[idx].append((params, color_normalized))
        count_num += len(os.listdir(split_file_root))
    
    image_save_size = (800,800)

    test_json = os.path.join(root_path, sequence, "scences/test.json")
    with open(test_json, "r")as f:
        json_data = json.load(f)[0]
    cam2pixels = json_data['cam2img']
    cam2pixels.update(json_data['cam2img_fisheye']) 
    lidar2cams = json_data['lidar2cam']
    lidar2cams.update(json_data['lidar2cam_fisheye'])
    camera_distort_param = json_data['distort_fisheye']

    img_paths = os.path.join(root_path, sequence, "samples/camera_0_1")
    img_lists = [os.path.join(img_paths, img) for img in os.listdir(img_paths)]
    img_lists = sorted(img_lists)
    for i in tqdm(range(nums)):
        img_path = img_lists[i]
        obj_frame = obj_occ[i]
        image_res_list = []
        for cam in image_pinhole_list:
            img_path_cur = img_path.replace("camera_0_1", cam)
            img = cv2.imread(img_path_cur)
            cam2pixel = np.array(cam2pixels[cam])[:3, :3]
            lidar2cam = np.array(lidar2cams[cam])
            image = draw_box_on_img_per(obj_frame, img.copy(), cam2pixel, lidar2cam)
            image = cv2.resize(image, image_save_size)
            image_res_list.append(image)
        return_image_pinhole = np.concatenate((image_res_list[0], image_res_list[1], image_res_list[2], image_res_list[3]), axis=1)
        
        image_res_list = []
        for cam in image_fisheye_list:
            img_path_cur = img_path.replace("camera_0_1", cam)
            img = cv2.imread(img_path_cur)
            cam2pixel = np.array(cam2pixels[cam])[:3, :3]
            lidar2cam = np.array(lidar2cams[cam])
            distort = camera_distort_param[cam]
            image = draw_box_on_img_per(obj_frame, img.copy(), cam2pixel, lidar2cam, distort)
            image = cv2.resize(image, image_save_size)
            image_res_list.append(image)
        return_image_fisheye = np.concatenate((image_res_list[0], image_res_list[1], image_res_list[2], image_res_list[3]), axis=1)
            
        image_concated = np.concatenate(([return_image_pinhole, return_image_fisheye]),axis=0)

        img_name = pcd_list[i].split('/')[-1].replace('pcd.bin', 'jpg')
        save_path = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis_all", sequence, "camera", img_name)
        cv2.imwrite(save_path, image_concated)
        
    image_folder = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis_all", sequence, "camera")
    save_video_path = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis_all/video_res", sequence + "_camera.mp4")
    images_to_video_ffmpeg(image_folder, save_video_path, 10)
    print("{} have finished!".format(sequence))

def refine_res_on_multi_cams(sequence):

    root_path = "/data1/turbo_data/liumengyuan/4D_label/dataset_track"
    image_pinhole_list = ['camera_0_1', 'camera_1_0', 'camera_2_0', 'camera_3_1']
    image_fisheye_list = ['camera_0_8', 'camera_1_8', 'camera_2_8', 'camera_3_8']
    
    pcd_root = os.path.join(root_path, sequence, "samples/lidar")
    pcd_list = [os.path.join(pcd_root, pcd_file) for pcd_file in os.listdir(pcd_root)]
    pcd_list = sorted(pcd_list)
    nums = len(pcd_list)

    obj_occ2 = defaultdict(list)
    refine_file = "/data1/turbo_data/yuanqingwen/origin_version/DetZero/refining/output/ref_model_cfgs/vehicle_grm_model/vis/eval/epoch_30/test/Vehicle_geometry_test.pkl"
    with open(refine_file, "rb")as file:
        data = pickle.load(file)
        res_data = data[next(iter(data.keys()))]
    for key, value in res_data.items():
        color = (_COLORS[key%len(_COLORS)] * 255).astype(np.uint8).tolist()
        color_normalized = (color[0]/255, color[1]/255, color[2]/255)
        for idx, params in zip(value['frame_id'], value['boxes_lidar']):
            obj_occ2[idx].append((params[0], color_normalized))
    # breakpoint()
    image_save_size = (800, 800)

    test_json = os.path.join(root_path, sequence, "scences/test.json")
    with open(test_json, "r")as f:
        json_data = json.load(f)[0]
    cam2pixels = json_data['cam2img']
    cam2pixels.update(json_data['cam2img_fisheye']) 
    lidar2cams = json_data['lidar2cam']
    lidar2cams.update(json_data['lidar2cam_fisheye'])
    camera_distort_param = json_data['distort_fisheye']

    img_paths = os.path.join(root_path, sequence, "samples/camera_0_1")
    img_lists = [os.path.join(img_paths, img) for img in os.listdir(img_paths)]
    img_lists = sorted(img_lists)
    for i in tqdm(range(nums)):
    # for i in tqdm(range(21, nums)):
        img_path = img_lists[i]
        obj_frame = obj_occ2[i]
        image_res_list = []
        for cam in image_pinhole_list:
            img_path_cur = img_path.replace("camera_0_1", cam)
            img = cv2.imread(img_path_cur)
            cam2pixel = np.array(cam2pixels[cam])[:3, :3]
            lidar2cam = np.array(lidar2cams[cam])
            image = draw_box_on_img_per(obj_frame, img.copy(), cam2pixel, lidar2cam)
            image = cv2.resize(image, image_save_size)
            image_res_list.append(image)
        return_image_pinhole = np.concatenate((image_res_list[0], image_res_list[1], image_res_list[2], image_res_list[3]), axis=1)
        
        image_res_list = []
        for cam in image_fisheye_list:
            img_path_cur = img_path.replace("camera_0_1", cam)
            img = cv2.imread(img_path_cur)
            cam2pixel = np.array(cam2pixels[cam])[:3, :3]
            lidar2cam = np.array(lidar2cams[cam])
            distort = camera_distort_param[cam]
            image = draw_box_on_img_per(obj_frame, img.copy(), cam2pixel, lidar2cam, distort)
            image = cv2.resize(image, image_save_size)
            image_res_list.append(image)
        return_image_fisheye = np.concatenate((image_res_list[0], image_res_list[1], image_res_list[2], image_res_list[3]), axis=1)
            
        image_concated = np.concatenate(([return_image_pinhole, return_image_fisheye]),axis=0)

        img_name = pcd_list[i].split('/')[-1].replace('pcd.bin', 'jpg')
        save_path = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/refine", sequence, "camera", img_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image_concated)
        
    image_folder = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/refine", sequence, "camera")
    save_video_path = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis_all/video_res", sequence + "_camera_refine.mp4")
    images_to_video_ffmpeg(image_folder, save_video_path, 10)
    print("{} have finished!".format(sequence))

def vis_refine_origin(sequence):
    root_path = "/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res"
    refine_folder = os.path.join(root_path, "refine", sequence, "bev")
    origin_folder = os.path.join(root_path, "vis_all", sequence, "bev")
    com_folder = os.path.join(root_path, "refine", sequence, "combine")
    save_video_path = os.path.join(root_path, "vis_all/video_res", sequence + "_combine.mp4")
    refine_imgs = sorted([os.path.join(refine_folder, file) for file in os.listdir(refine_folder)])
    origin_imgs = sorted([os.path.join(origin_folder, file) for file in os.listdir(origin_folder)])
    assert len(refine_imgs) == len(origin_imgs), f"please check nums of two folder imgs"
    # for refine_img, origin_img in tqdm(zip(refine_imgs, origin_imgs)):
    #     assert refine_img.split('/')[-1] == origin_img.split('/')[-1], f"error img pairs!"
    #     image_lists = []
    #     image_lists.append(cv2.imread(origin_img))
    #     image_lists.append(cv2.imread(refine_img))
    #     return_image_com = np.concatenate((image_lists[0], image_lists[1]), axis=1)
    #     save_img_path = os.path.join(com_folder, refine_img.split('/')[-1])
    #     os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
    #     cv2.imwrite(save_img_path, return_image_com)
    images_to_video_ffmpeg(com_folder, save_video_path, 10)
    print("{} have finished!".format(sequence))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)

    args = parser.parse_args()
    
    # vis_refine_origin(args.sequence)
    # breakpoint()
    refine_res_on_bev(args.sequence, False)
    breakpoint()
    # refine_res_on_multi_cams(args.sequence)
    # breakpoint()
    # seq_res_on_bev(args.sequence, False)
    # seq_res_on_multi_cams(args.sequence)

    image_folder = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/refine", args.sequence, "bev")
    save_video_path = os.path.join("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/vis_all/video_res", args.sequence + "_bev_04.mp4")
    images_to_video_ffmpeg(image_folder, save_video_path, 10)
    print("{} have finished!".format(args.sequence))
    
    breakpoint()
    pkl_refine = "/data1/turbo_data/yuanqingwen/origin_version/DetZero/refining/output/ref_model_cfgs/vehicle_grm_model/debug/eval/epoch_1/test/Vehicle_geometry_test.pkl"
    with open(pkl_refine, "rb") as file:
        refine_data = pickle.load(file)
        refine_data = refine_data['train_sh_3d_road_5_20250524_5000_lx']
    