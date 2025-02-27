import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
from typing import Optional, List, Tuple
import mmcv
import pickle
import io

'''
{
        'car':0,
        'bus':1,
        'truck':2,
        'bike':3,
        'bicycle':4,
        'person':5,
        'NotUsed':6
    }

''' 

OBJECT_PALETTE = {
    "car": [255, 0, 0],
    "pedestrian": [0, 255, 0],
    "cyclist": [0, 0, 255],
}

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

def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    xlim: Tuple[float, float] = (-80, 80),
    ylim: Tuple[float, float] = (-80, 80),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 15,
    base_name: str
) -> None:
    """
    Visualize LiDAR points and bounding boxes in BEV.
    """
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )
    
    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes[:, [0, 1, 2, 3, 0],:][:,:,[0,1]]#bboxes.corners[:, [0, 1, 2, 3, 0], :2]
        for index in range(coords.shape[0]): # n个目标
            name = labels[index]#classes[labels[index]]
            plt.plot( 
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE.get(name, [255, 255, 255])) / 255,
            ) # 整体画框
        coords = bboxes[:, [1, 5], :][:,:,[0,1]] #
        for index in range(coords.shape[0]):
            name = labels[index]#classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness * 3,
                color=np.array(color or OBJECT_PALETTE.get(name, [255, 255, 255])) / 255,
            ) # 绘制头部

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        os.path.join(fpath,base_name+'_.jpg'),
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def plot_in_image(det_path, track_path):
    img_list = []
    objects_data = []
    with open(det_path, 'rb') as file:
        det_data = pickle.load(file)
    with open(track_path, 'rb') as file:
        track_data_pkl = pickle.load(file)
        first_key = list(track_data_pkl.keys())[0]
        print("track_data_pkl.keys():{}".format(first_key))
        track_data = track_data_pkl[first_key]
    frame_id = 0
    if len(det_data)!=len(track_data):
        print("len(det_data):{};len(track_data):{}".format(len(det_data), len(track_data)))
    
    
    for frame_idx in range(len(det_data)):
        corners_3d_world_rotation_track_list = []
        for track_id in track_data:
            sample_idx = track_data[track_id]['sample_idx']
            boxes_global = track_data[track_id]['boxes_global']
            boxes_pose = track_data[track_id]['pose']
            if str(frame_id) in sample_idx:
                track_frame_idx = sample_idx.tolist().index(str(frame_id))
                box_global_track_frame = boxes_global[track_frame_idx]
                box_pose_track_frame = boxes_pose[track_frame_idx]
                x,y,z,l,w,h,yaw,vx,vy = box_global_track_frame
                corners_3d_obj = np.array([
                    [-l / 2, -w / 2,-h / 2],
                    [l / 2, -w / 2, -h / 2],
                    [l / 2,  w / 2, -h / 2],
                    [-l / 2, w / 2, -h / 2],
                    [-l / 2, -w / 2, h / 2],
                    [l / 2, -w / 2, h / 2],
                    [l / 2, w / 2, h / 2],
                    [-l / 2, w / 2, h / 2]
                ])
                corners_3d_obj_roation = rotate_points(corners_3d_obj.T, yaw)
                corners_3d_world_rotation = np.array([[x, y, z]]).T + corners_3d_obj_roation
                corners_3d_world_rotation_hom = np.hstack((corners_3d_world_rotation.T, np.ones((corners_3d_world_rotation.T.shape[0], 1))))
                corners_3d_frame_hom = np.linalg.inv(box_pose_track_frame) @  corners_3d_world_rotation_hom.T
                corners_3d_world_rotation_track_list.append(corners_3d_frame_hom[:3,:])
        
        corners_3d_world_rotation_list = []
        lidar_boxes = det_data[frame_idx]['boxes_lidar']
        
        for box in lidar_boxes:
            x,y,z,l,w,h,yaw,vx,vy = box  
            corners_3d_obj = np.array([
                [-l / 2, -w / 2,-h / 2],
                [l / 2, -w / 2, -h / 2],
                [l / 2,  w / 2, -h / 2],
                [-l / 2, w / 2, -h / 2],
                [-l / 2, -w / 2, h / 2],
                [l / 2, -w / 2, h / 2],
                [l / 2, w / 2, h / 2],
                [-l / 2, w / 2, h / 2]
            ])
            corners_3d_obj_roation = rotate_points(corners_3d_obj.T, yaw)
            corners_3d_world_rotation = np.array([[x, y, z]]).T + corners_3d_obj_roation
            corners_3d_world_rotation_list.append(corners_3d_world_rotation)
        
        bboxes = np.array([i.T for i in corners_3d_world_rotation_list])
        bboxes_track = np.array([i.T for i in corners_3d_world_rotation_track_list])
        xlim = (-50, 50)
        ylim = (-50, 50)
        thickness=15
        fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
        ax = plt.gca()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(1)
        ax.set_axis_off()
        #########################################################################
        if bboxes is not None and len(bboxes) > 0:
            coords = bboxes[:, [0, 1, 2, 3, 0],:][:,:,[0,1]]#bboxes.corners[:, [0, 1, 2, 3, 0], :2]
            for index in range(coords.shape[0]): # n个目标
                # name = labels[index]#classes[labels[index]]
                plt.plot( 
                    coords[index, :, 0],
                    coords[index, :, 1],
                    linewidth=thickness,
                    color=[1,0,0]#np.array(color or OBJECT_PALETTE.get(name, [255, 255, 255])) / 255,
                ) # 整体画框
            coords = bboxes[:, [1, 2], :][:,:,[0,1]] #
            for index in range(coords.shape[0]):
                # name = labels[index]#classes[labels[index]]
                plt.plot(
                    coords[index, :, 0],
                    coords[index, :, 1],
                    linewidth=thickness * 2,
                    color=[1,1,1]#np.array(color or OBJECT_PALETTE.get(name, [255, 255, 255])) / 255,
                ) # 绘制头部
        #################################

        #########################################################################
        if bboxes_track is not None and len(bboxes_track) > 0:
            coords = bboxes_track[:, [0, 1, 2, 3, 0],:][:,:,[0,1]]#bboxes.corners[:, [0, 1, 2, 3, 0], :2]
            for index in range(coords.shape[0]): # n个目标
                # name = labels[index]#classes[labels[index]]
                plt.plot( 
                    coords[index, :, 0],
                    coords[index, :, 1],
                    linewidth=thickness,
                    color=[0,1,0]#np.array(color or OBJECT_PALETTE.get(name, [255, 255, 255])) / 255,
                ) # 整体画框
            coords = bboxes_track[:, [1, 2], :][:,:,[0,1]] #
            for index in range(coords.shape[0]):
                # name = labels[index]#classes[labels[index]]
                plt.plot(
                    coords[index, :, 0],
                    coords[index, :, 1],
                    linewidth=thickness * 2,
                    color=[1,1,0]#np.array(color or OBJECT_PALETTE.get(name, [255, 255, 255])) / 255,
                ) # 绘制头部
        #################################
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, dpi=10,
            facecolor="black",
            format="png",
            bbox_inches="tight",
            pad_inches=0,)
        # fig.savefig(
        #     os.path.join(fpath,str(frame_id).zfill(4)+'_.jpg'),
        #     dpi=10,
        #     facecolor="black",
        #     format="png",
        #     bbox_inches="tight",
        #     pad_inches=0,
        # )
        img_array = np.frombuffer(img_buffer.getvalue(), dtype=np.uint8)
        img_mat = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_list.append(img_mat)
        plt.close(fig)
        frame_id += 1
        print("frame_id:{}".format(frame_id))
        ######################
    return img_list

def generate_video(img_list, save_path):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    # h,w,_ = cv2.imread(os.path.join(img_list,img_list[0])).shape
    h,w,_ = img_list[0].shape
    videoWriter = cv2.VideoWriter(save_path,fourcc,5,(w,h))
    for img in img_list:
        videoWriter.write(img)
    videoWriter.release()

"""
功能：将视频转成图片(提取视频的每一帧图片)
     1.能够设置多少帧提取一帧图片
     2.可以设置输出图片的大小及灰度图
     3.手动设置输出图片的命名格式
"""
def ExtractVideoFrame(video_path, num_frame=600):
    # 输出文件夹不存在，则创建输出文件夹
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)
    times = 0               # 用来记录帧
    frame_frequency = 1     # 提取视频的频率，每frameFrequency帧提取一张图片，提取完整视频帧设置为1
    count = 0               # 计数用，分割的图片按照count来命名
    cap = cv2.VideoCapture(video_path)  # 读取视频文件

    img_list = []
    print('开始提取', video_path, '视频的图片')
    while True:
        times += 1
        print(times)
        det_data, image = cap.read()          # 读出图片。res表示是否读取到图片，image表示读取到的每一帧图片
        if not det_data or times > num_frame:
            print('图片提取结束')
            break
        if times % frame_frequency == 0:
            # picture_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图片转成灰度图
            # image_resize = cv2.resize(image, (368, 640))            # 修改图片的大小
            img_name = str(count).zfill(6)+'.jpg'
            # cv2.imwrite(output_path + os.sep + img_name, image)
            count += 1
            img_list.append(image)
    cap.release()
    return img_list

if __name__ == '__main__':
    det_path = '/home/mogo/data/dev/DetZero/tracking/track_input_20250120.pkl'
    track_path = '/home/mogo/data/dev/DetZero/tracking/results/tracking/tracking-test-20250120-121847.pkl' # update box
    # track_path = '/home/mogo/data/dev/DetZero/tracking/results/tracking/tracking-test-20250110-080917.pkl'
    img_list = plot_in_image(det_path, track_path)
    generate_video(img_list, save_path="demo.mp4")
