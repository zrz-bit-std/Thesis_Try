import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import sys
import pickle
sys.path.append('/rss/wangruihao/code/4D')
from utils.nms import nms_angle,bb_intersection_over_union_angle
from typing import Optional, List, Tuple
# import mmcv
import re

OBJECT_PALETTE = {
    "car":          [255, 0, 0],        #红色
    "truck":        [0, 255, 0],        #绿色
    "bus":          [0, 255, 255],      #青色
    "rider":        [255, 255, 0],      #黄色
    "bike":         [255, 255, 0],
    "bicycle":      [255, 0, 255],      #洋红
    "person":     [0, 0, 255],        #蓝色
    'motor':      [0,0,255],
    "NotUsed":      [160, 32, 240],     #紫色
}

'''
          'car': 0.44,
          'bus': 0.45,
          'person': 0.4,
          'truck': 0.5,
          'bike': 0.45,
          'bicycle': 0.45,
          'motor': 0.45,
          'NotUsed': 0.45,

'''
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

def project_3d_to_2d(points_3d, center_coord,camera_matrix, camera_extrinsic,image_shape):
    """
    将3D点投影到2D图像平面上

    参数:
    points_3d: 形状为 (n, 3) 的numpy数组，表示n个3D点，每个点的坐标为 (x, y, z)
    camera_matrix: 相机内参矩阵，形状为 (3, 3)
    camera_extrinsic: 相机外参（旋转和平移变换矩阵），形状为 (4, 4)

    返回:
    points_2d: 形状为 (n, 2) 的numpy数组，表示投影后的2D点坐标
    """
    # 将3D点坐标扩展为齐次坐标形式（增加一维，值为1）
    points_3d_homogeneous = np.hstack((points_3d.T, np.ones((points_3d.T.shape[0], 1))))
    # 先通过相机外参将世界坐标转换到相机坐标
    points_3d_camera = np.dot(camera_extrinsic[:3, :], points_3d_homogeneous.T).T
    # 再使用相机内参进行投影变换
    projected_points = np.dot(camera_matrix, points_3d_camera.T).T
    # 归一化（除以Z坐标，转换到图像平面坐标）
    points_2d = projected_points[:, :2] / projected_points[:, 2, np.newaxis]
    # 计算中心点的转化
    center_in_camera_ = np.dot(camera_extrinsic[:3, :], center_coord.T).T
    center_in_image_ = np.dot(camera_matrix, center_in_camera_.T).T[0]
    center_in_image = [center_in_image_[0]/center_in_image_[2],center_in_image_[1]/center_in_image_[2]]
    # 判断是否合理
    valid_point_idx = (points_2d[..., 0] >= 0) & \
                (points_2d[..., 0] <= image_shape[1]) & \
                (points_2d[..., 1] >= 0) & (points_2d[..., 1] <= image_shape[0])
    valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 1
    return points_2d,center_in_image,valid_bbox_idx

def draw_3d_box_image(image, corners_2d, color=(0, 255, 0)):
    """
    在给定的图像上绘制3D框的2D投影

    参数:
    image: 要绘制的OpenCV图像（numpy数组形式）
    corners_2d: 形状为 (8, 2) 的numpy数组，表示3D框8个顶点投影后的2D坐标
    color: 绘制框的颜色，默认为绿色（BGR格式），可以根据需求修改

    返回:
    image: 绘制好3D框投影的图像
    corners_3d_obj = np.array([
                [-l / 2, -h / 2, -w / 2],
                [l / 2, -h / 2, -w / 2],
                [l / 2, -h / 2, w / 2],
                [-l / 2, -h / 2, w / 2],
                [-l / 2, h / 2, -w / 2],
                [l / 2, h / 2, -w / 2],
                [l / 2, h / 2, w / 2],
                [-l / 2, h / 2, w / 2]
            ])
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
    ## 绘制头部
    color_head=(0, 0, 255)
    cv2.line(image, tuple(corners_2d[1].astype(int)), tuple(corners_2d[5].astype(int)), color_head, 2)# 底部
    cv2.line(image, tuple(corners_2d[2].astype(int)), tuple(corners_2d[6].astype(int)), color_head, 2)# 顶部
    cv2.line(image, tuple(corners_2d[1].astype(int)), tuple(corners_2d[2].astype(int)), color_head, 2)# 左边
    cv2.line(image, tuple(corners_2d[5].astype(int)), tuple(corners_2d[6].astype(int)), color_head, 2)# 顶部
    return image


def lidar_to_base(lidar_points, lidar_to_base_matrix):
    """
    Transform LiDAR points to Base frame using extrinsics.
    """
    lidar_homogeneous = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    base_points = (lidar_to_base_matrix @ lidar_homogeneous.T).T
    return base_points[:, :3]


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
    # yaw = yaw*-1
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return rotation_matrix @ points


def get_param():
    lidar2local_0 = [0.00420054419145802,0.987785850571192,0.155760934896587,-0.0312053215446107,
                0.0774827236144656,-0.155615531715935,0.984774204490586,0.242274772251415,
                0.99698484590779,0.00793219390395222,-0.0771900079679362,5.77751328830199,
                0,0,0,1]
    lidar2local_1 = [0.0163333136954781,0.994490606701314,-0.103545429868145,0.460278710495142,
                0.0616909727527799,0.102359608176502,0.9928326820239,0.0175039080310264,
                0.997961645928611,-0.0226040659352898,-0.0596792213308132,6.27453351412179,
                0,0,0,1]
    lidar2local_2 = [0.0344137333531827,0.999393976440711,-0.00523209429573199,0.0594080576246223,
                0.0545021363948473,0.00335069679056996,0.998508031995445,0.236678085896668,
                0.997920443765481,-0.0346475494810391,-0.0543537968302624,6.01354767881865,
                0,0,0,1]
    lidar2local_3 = [0.0123900486862991,0.988879390135607,0.148202693830371,0.153594296603299,
                0.041371168519722,-0.148594125523643,0.988032495556283,0.158255048883195,
                0.999067021329832,-0.00611045210205486,-0.0427521843447162,6.01868501997549,
                0,0,0,1]
    local2utm_0 = [-0.988178775064224,0.153305931106902,-2.16840434497101e-19,651915.880584317,
                    -0.153305931106902,-0.988178775064224,0,2973850.07431576,
                    0,1.38777878078145e-17,1,47.626,
                    0,0,0,1]
    local2utm_1 = [-0.00122638645008498,0.999999247987855,6.93889390390723e-18,651880.583636056,
                    -0.999999247987855,-0.00122638645008502,5.20417042793042e-18,2973819.18322367,
                    1.73472347597681e-18,-6.93889390390723e-18,1,47.406,
                    0,0,0,1]
    local2utm_2 = [0.992786700426065,-0.119893984240772,1.30104260698261e-17,651937.312934625,
                0.119893984240772,0.992786700426065,-6.93889390390723e-18,2973766.53320847,
                -4.39101879856629e-18,-6.93889390390723e-18,1,47.121,
                0,0,0,1]
    local2utm_3 = [0.161017419995756,0.986951564393061,0,651886.845536876,
            -0.986951564393061,0.161017419995756,7.58941520739853e-19,2973785.54986809,
            8.67361737988404e-19,-6.93889390390723e-18,1,46.83905,
            0,0,0,1]
    com2utm_0 = [ 1.3073711558850998e-01, 9.9141706995976042e-01, 0.,
            6.5192594590758975e+05, -9.9141706995976042e-01,
            1.3073711558850998e-01, 0., 2.9738054291297784e+06, 0., 0., 1.,
            4.7299999999999997e+01, 0., 0., 0., 1. ]
    com2utm_1 = [ 1.3073711558850998e-01, 9.9141706995976042e-01, 0.,
            6.5192594590758975e+05, -9.9141706995976042e-01,
            1.3073711558850998e-01, 0., 2.9738054291297784e+06, 0., 0., 1.,
            4.7299999999999997e+01, 0., 0., 0., 1. ]
    com2utm_2 =  [ 1.3073711558850998e-01, 9.9141706995976042e-01, 0.,
            6.5192594590758975e+05, -9.9141706995976042e-01,
            1.3073711558850998e-01, 0., 2.9738054291297784e+06, 0., 0., 1.,
            4.7299999999999997e+01, 0., 0., 0., 1. ]
    com2utm_3 = [ 1.3073711558850998e-01, 9.9141706995976042e-01, 0.,
            6.5192594590758975e+05, -9.9141706995976042e-01,
            1.3073711558850998e-01, 0., 2.9738054291297784e+06, 0., 0., 1.,
            4.7299999999999997e+01, 0., 0., 0., 1. ]
    return [local2utm_0,local2utm_1,local2utm_2,local2utm_3],[com2utm_0,com2utm_1,com2utm_2,com2utm_3],[lidar2local_0,lidar2local_1,lidar2local_2,lidar2local_3]



def angle_with_x_axis(point):
    """
    计算二维坐标系中给定点与原点连线和x轴正方向的夹角，范围在(0, 2*pi)
    """
    x, y = point
    # 计算该点到原点的向量
    vector = np.array([x, y])
    # 计算向量的模（长度）
    magnitude = np.linalg.norm(vector)
    # 计算向量与x轴正方向单位向量（[1, 0]）的点积
    unit_x_vector = np.array([1, 0])
    dot_product = np.dot(vector, unit_x_vector)
    # 通过点积公式计算夹角的余弦值
    cos_angle = dot_product / magnitude
    # 使用反余弦函数得到夹角（弧度制），初始范围是[0, pi]
    angle_rad = np.arccos(cos_angle)
    # 根据y值的正负来判断是否需要调整角度范围到(0, 2*pi)
    if y < 0:
        angle_rad = 2 * np.pi - angle_rad
    return np.rad2deg(angle_rad)

def smooth_with_outlier_replacement(sequence, window_size=5, threshold=3.0):
    """
    滑动窗口平滑序列，并用平滑后的结果补全离群点
    :param sequence: 输入的序列 (1D array-like)
    :param window_size: 滑动窗口大小，必须是正奇数
    :param threshold: 离群点判断阈值（以 MAD 为基准）
    :return: (平滑后的序列, 离群点索引)
    """
    if window_size % 2 == 0 or window_size < 1:
        raise ValueError("window_size 必须是正奇数")
    
    half_window = window_size // 2
    smoothed = np.zeros_like(sequence, dtype=float)
    outlier_indices = []

    for i in range(len(sequence)):
        # 获取窗口范围
        start = max(0, i - half_window)
        end = min(len(sequence), i + half_window + 1)
        window = sequence[start:end]
        
        # 计算中位数和 MAD（绝对中位差）
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        
        # 判断离群点：绝对差 > threshold * MAD
        if mad == 0:
            filtered_window = window  # 若 MAD 为零（窗口内无差异），不检测离群点
        else:
            filtered_window = window[np.abs(window - median) <= threshold * mad]
        
        # 如果过滤后窗口为空，用原始窗口计算均值
        if len(filtered_window) == 0:
            smoothed[i] = np.mean(window)
        else:
            smoothed[i] = np.mean(filtered_window)
        
        # 判断当前位置是否为离群点
        if np.abs(sequence[i] - median) > threshold * mad:
            outlier_indices.append(i)
            sequence[i] = smoothed[i]  # 替换离群点为平滑值
    
    return smoothed, outlier_indices
 

def smooth_boxes(boxes_global,score):
    # boxes_global = tracking['boxes_global'] # n*9
    # socre = tracking['score']
    x = smooth_with_outlier_replacement(boxes_global[:,0])[0]
    y = smooth_with_outlier_replacement(boxes_global[:,1])[0]
    z = smooth_with_outlier_replacement(boxes_global[:,2])[0]
    yaw = smooth_with_outlier_replacement(boxes_global[:,6])[0]
    wlh = boxes_global[score.argmax()][3:6]
    boxes_global[:,0] = x
    boxes_global[:,1] = y
    boxes_global[:,2] = z
    boxes_global[:,6] = yaw
    boxes_global[:,3:6] = wlh
    return boxes_global

# def nms

def combine_data_from_label():
    d_label = {0:'car',1:'truck',2:'bus',3:'bike',4:'bike',5:'person'}
    base_total_dir = '/rss/wangruihao/data/road_bev_bags_241112/20241112_064800/data/result'
    frame_name_list = sorted(os.listdir('/rss/wangruihao/data/road_bev_bags_241112/20241112_064800/data/result0'))[-200:]
    tracking_pickle_path = '/rss/wangruihao/code/4D/tracking-test-20241218-031310.pkl'
    online_lable_path = '/rss/wangruihao/data/road_bev_bags_241112/20241112_064800/label_from_online/label'
    lidar_path = '/rss/wangruihao/data/road_bev_bags_241112/20241112_064800/npy/'
    fpath = '/rss/wangruihao/code/4D/Vis_res_1217'
    local2utm_list, com2utm_list, lidar2local_list = get_param()
    local2utm_np = [np.array(i).reshape(4,4) for i in local2utm_list]
    utm2com_np = [np.linalg.inv(np.array(i).reshape(4,4)) for i in com2utm_list]
    local2com = [utm2com_np[i] @ local2utm_np[i] for i in range(len(local2utm_np))]
    lidar2local_np = [np.array(i).reshape(4,4) for i in lidar2local_list]
    with open(tracking_pickle_path, 'rb') as file:
        res_track_ = pickle.load(file)['20241216']
    # for track_id in res_track_:
    #     sample_idx = res_track_[track_id]['sample_idx']
    #     boxes_global = res_track_[track_id]['boxes_global']
    #     score = res_track_[track_id]['score']
    #     boxes_pose = res_track_[track_id]['pose']
    #     label_name = res_track_[0]['name']
    for frame_id_ in range(len(frame_name_list)):
        frame = frame_name_list[frame_id_]
        base_name = frame.split('.')[0]
        box_corner_list_lion = []
        nms_prepare_res = []
        corners_3d_world_rotation_track_list = []
        boxex_list_orgin = []
        for track_id in res_track_:
            sample_idx = res_track_[track_id]['sample_idx']
            boxes_global = res_track_[track_id]['boxes_global']
            score = res_track_[track_id]['score']
            boxes_pose = res_track_[track_id]['pose']
            label_name = res_track_[track_id]['name']
            boxes_global = smooth_boxes(boxes_global=boxes_global,score=score)
            if str(frame_id_) in sample_idx:
                track_frame_idx = sample_idx.tolist().index(str(frame_id_))
                box_global_track_frame = boxes_global[track_frame_idx]
                box_pose_track_frame = boxes_pose[track_frame_idx]
                x,y,z,l,w,h,yaw,s,label = box_global_track_frame
                label = label_name[track_frame_idx]
                print(label)
                boxex_list_orgin.append([label,x,y,z,l,w,h,yaw,s])
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
        ############################################
        # 
        ######################

        ############################################
        """
        Visualize LiDAR points and bounding boxes in BEV.
        """
        lidar_sub_path_list = [os.path.join(lidar_path,i,base_name+'.npy') for i in ['0','1','2','3']]
        lidar_points_1 = lidar_to_base(np.load(lidar_sub_path_list[0])[:,:3],lidar2local_np[0])
        lidar_points_2 = lidar_to_base(np.load(lidar_sub_path_list[1])[:,:3],lidar2local_np[1])
        lidar_points_3 = lidar_to_base(np.load(lidar_sub_path_list[2])[:,:3],lidar2local_np[2])
        lidar_points_4 = lidar_to_base(np.load(lidar_sub_path_list[3])[:,:3],lidar2local_np[3])
        # 转换到Base坐标系
        lidar_base_1 = lidar_to_base(lidar_points_1, local2com[0])
        lidar_base_2 = lidar_to_base(lidar_points_2, local2com[1])
        lidar_base_3 = lidar_to_base(lidar_points_3, local2com[2])
        lidar_base_4 = lidar_to_base(lidar_points_4, local2com[3])
        # # 合并点云
        lidar_base = np.vstack((lidar_base_1, lidar_base_2, lidar_base_3, lidar_base_4))
        bboxes = np.array([i.T for i in corners_3d_world_rotation_track_list])
        # bboxes_online = np.array([i.T for i in box_corner_list_online])
        # bboxes = bboxes_online
        xlim = (-80, 80)
        ylim = (-80, 80)
        color = [255,0,0]
        thickness=15
        fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
        ax = plt.gca()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(1)
        ax.set_axis_off()
        if lidar_base is not None:
            plt.scatter(
                lidar_base[:, 0],
                lidar_base[:, 1],
                s=30,
                c="white",
            )

        if bboxes is not None and len(bboxes) > 0:
            coords = bboxes[:, [0, 1, 2, 3, 0],:][:,:,[0,1]]#bboxes.corners[:, [0, 1, 2, 3, 0], :2]
            for index in range(coords.shape[0]): # n个目标
                # name = labels[index]#classes[labels[index]]
                # print(save_list[index][0])
                # print(np.array(OBJECT_PALETTE.get(save_list[index][0], [255, 255, 255])) / 255,) # 整体画框)
                # print(boxex_list_orgin[index][0])
                plt.plot( 
                    coords[index, :, 0],
                    coords[index, :, 1],
                    linewidth=thickness,
                    color=np.array(OBJECT_PALETTE.get(boxex_list_orgin[index][0], [255, 255, 255])) / 255,
                ) # 整体画框
            coords = bboxes[:, [1, 2], :][:,:,[0,1]] #
            for index in range(coords.shape[0]):
                # name = labels[index]#classes[labels[index]]
                plt.plot(
                    coords[index, :, 0],
                    coords[index, :, 1],
                    linewidth=thickness * 3,
                    color=[1,1,1]#np.array(color or OBJECT_PALETTE.get(name, [255, 255, 255])) / 255,
                ) # 绘制头部

        # mmcv.mkdir_or_exist(os.path.dirname(fpath))
        fig.savefig(
            os.path.join(fpath,base_name+'_1.jpg'),
            dpi=10,
            facecolor="black",
            format="png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
        print(base_name)
        ######################
        

if __name__ == '__main__':
    combine_data_from_label()