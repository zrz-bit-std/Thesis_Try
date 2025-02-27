import numpy as np
import cv2
from tqdm import tqdm

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

def project_3d_to_fisheye(points,image_shape,intrinsic_matrix, extrinsic_matrix, distortion_params):
    # 将3D点坐标构建成齐次坐标形式
    x,y,z = points
    point_3d_homogeneous = np.array([[x], [y], [z], [1]])
    # distortion_params = [1.0926628389307196e-01, -6.5713320780575097e-04, 8.4866561354316559e-03, -4.2045330300667406e-03]
    k1,k2,k3,k4 = distortion_params
    # 通过外参矩阵将3D点从世界坐标系转换到相机坐标系
    point_camera_coords = np.dot(extrinsic_matrix[:3, :], point_3d_homogeneous)

    # 计算归一化平面坐标（未考虑畸变）
    x_camera, y_camera, z_camera = point_camera_coords.flatten()[:3]
    if z_camera < 0:
        return None,None,False
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
    if not (u>0 and u<image_shape[1] and v>0 and v<image_shape[1]):
        return u,v,False
    return u, v,True

def project_3d_to_pinhole(points,image_shape,intrinsic_matrix, extrinsic_matrix, distortion_params = None):
    # 将3D点坐标构建成齐次坐标形式
    x,y,z = points
    point_3d_homogeneous = np.array([[x], [y], [z], [1]])

    # 通过外参矩阵将3D点从世界坐标系转换到相机坐标系
    point_camera_coords = np.dot(extrinsic_matrix[:3, :], point_3d_homogeneous)

    # 计算归一化平面坐标（未考虑畸变）
    # x_camera, y_camera, z_camera = point_camera_coords.flatten()[:3]
    if point_camera_coords[2,0] < 0:
        return None,None,False
    points_image_ = intrinsic_matrix @ point_camera_coords
    points_image = points_image_/points_image_[2,:]
    u = points_image[0, 0]
    v = points_image[1, 0]
    if not (u>0 and u<image_shape[1] and v>0 and v<image_shape[1]):
        return u,v,False
    return u, v,True

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
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
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

def draw_box_on_fisheye(shape_center,yaw,image,inter_param,exter_param,distort,color = [255,0,0]):
    w, l,h, x, y, z = shape_center
        # 构建3D框的8个顶点坐标（在物体坐标系下，原点在物体中心）
    corners_3d_obj = np.array([
        [-l / 2, -w / 2,-h / 2],
        [l / 2,-w / 2, -h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2]
    ])
    # center_coord = np.array([[x,y,z,1]])
    corners_3d_obj_roation = rotate_points(corners_3d_obj.T, yaw)
    corners_3d_rotation_points = np.array([[x, y, z]]).T + corners_3d_obj_roation
    corners_2d = []
    image_shape = image.shape
    valid_flag = []
    for points in corners_3d_rotation_points.T:
        u,v,vali = project_3d_to_fisheye(points,image_shape,inter_param, exter_param, distort)
        if u is None and v is None:
            return image
        if vali:
            valid_flag.append(1)
        corners_2d.append([u,v])
    if len(valid_flag) < 1:
        return image
    corners_2d = np.array(corners_2d)
    image = draw_3d_box(image, corners_2d.astype(int),color=color)
    return image

def draw_box_on_pinhole(shape_center,yaw,image,inter_param,exter_param,distort=None,color = [255,0,0]):
    w, l,h, x, y, z = shape_center
        # 构建3D框的8个顶点坐标（在物体坐标系下，原点在物体中心）
    corners_3d_obj = np.array([
        [-l / 2, -w / 2,-h / 2],
        [l / 2,-w / 2, -h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2]
    ])
    # center_coord = np.array([[x,y,z,1]])
    corners_3d_obj_roation = rotate_points(corners_3d_obj.T, yaw)
    corners_3d_rotation_points = np.array([[x, y, z]]).T + corners_3d_obj_roation
    corners_2d = []
    image_shape = image.shape
    valid_flag = []
    for points in corners_3d_rotation_points.T:
        u,v,vali = project_3d_to_pinhole(points,image_shape,inter_param, exter_param)
        if u is None and u is None:
            return image 
        if vali:
            valid_flag.append(1)
        corners_2d.append([u,v])
    if len(valid_flag) < 1:
        return image
    corners_2d = np.array(corners_2d)
    image = draw_3d_box(image, corners_2d.astype(int),color=color)
    return image

def draw_3d_box(image, corners_2d, color=(0, 255, 0)):
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
    ## 绘制头部
    # color_head=(color[0]/2.0,color[1]/2.0,color[2]/2.0)
    color_head = (255,255,255)
    cv2.line(image, tuple(corners_2d[1].astype(int)), tuple(corners_2d[5].astype(int)), color_head, 6)# 底部
    cv2.line(image, tuple(corners_2d[2].astype(int)), tuple(corners_2d[6].astype(int)), color_head, 6)# 顶部
    cv2.line(image, tuple(corners_2d[1].astype(int)), tuple(corners_2d[2].astype(int)), color_head, 6)# 左边
    cv2.line(image, tuple(corners_2d[5].astype(int)), tuple(corners_2d[6].astype(int)), color_head, 6)# 顶部
    return image

def image2common(image_view_points, camera_intrinsic, ego2camera_matrix, height):
    '''
    :param image_view_points: np.ndarray 3*n [[u,v,1],,,,]
    :param camera_intrinsic: np.ndarray 3*3
    :param ego2camera_matrix: nd.ndarray 4*4 ego -> camera
    :param height: int defalut = 0
    :return: ndarray 3*n [[x_ego,y_ego,hegiht],,,]
    '''
    camera_intrinsic_inv = np.linalg.inv(camera_intrinsic)
    R_inv = ego2camera_matrix[:3, :3].T
    T = ego2camera_matrix[:3, 3]
    mat1 = np.dot(np.dot(R_inv, camera_intrinsic_inv), image_view_points)
    mat2 = np.dot(R_inv, T)
    Zc = (height + mat2[2]) / mat1[2]
    points_ego = Zc * mat1 - np.expand_dims(mat2, 1)
    return points_ego

def fisheye2common(fisheye_points,K,exter_param,D,z_common):
    point = np.array([[[fisheye_points[0], fisheye_points[1]]]], dtype=np.float32)
    undistorted_point = cv2.fisheye.undistortPoints(point, K, D)
    x_norm, y_norm = undistorted_point[0][0]
    x_pixel = x_norm * K[0, 0] + K[0, 2]
    y_pixel = y_norm * K[1, 1] + K[1, 2]
    if x_pixel > 6000 or y_pixel > 6000 or x_pixel <-6000 or y_pixel < -6000:
        return None
    common_res = image2common(np.array([[x_pixel ,y_pixel,1]]).T,K, ego2camera_matrix=exter_param, height=z_common)
    return common_res.T[0]

if __name__ == '__main__':
    print('haha')