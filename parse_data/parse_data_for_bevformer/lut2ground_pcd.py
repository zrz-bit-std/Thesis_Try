import os
import json
import math
import numpy as np
import yaml
import pyproj
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time

# 将UTM坐标转换为相机坐标
def utm_to_local_coords(easting, northing, height, T_g_l):
    relative_coords = np.array([easting, northing, height]) 
    point_utm = np.array([relative_coords[0], relative_coords[1], relative_coords[2], 1.0])
    point_local = np.dot(T_g_l, point_utm)
    return point_local[:3]

def wgs84_to_cgcs2000(self, lat, lon):
        easting, northing = self.transformer.transform(lon, lat)
        #print("easting, northing",easting, northing)
        return easting, northing

def utm2xy(self, latitude, longitude):
        easting, northing = wgs84_to_cgcs2000(latitude, longitude)
        height = 0  # 假设地面高度为0
        local_coords = utm_to_local_coords(easting, northing, height, self.T_g_l)
        x, y = local_coords[0], local_coords[1]
        return x, y

# 注册 OpenCV Matrix 解析器
def opencv_matrix_constructor(loader, node):
    rows = int(node.value[0][1].value)
    cols = int(node.value[1][1].value)
    data = node.value[3][1].value
    data_values = [float(val.value) for val in data]
    return data_values

yaml.add_constructor('tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor)

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        cleaned_lines = lines[2:]
        cleaned_content = ''.join(cleaned_lines)
        data = yaml.load(cleaned_content, Loader=yaml.FullLoader)
        return data

wgs84 = pyproj.CRS("EPSG:4326")
utm_zone_51N = pyproj.CRS("EPSG:32651")

def save_pcd(points_xyz, filename, intensity_val=0, lidar_type=0):
    """
    保存点云为 PCD v0.7 ASCII 格式
    points_xyz: Nx3 numpy 数组 (x,y,z)
    """
    N = points_xyz.shape[0]
    timestamp = int(time.time())  # 当前时间戳，或者替换为你数据对应时间

    # 如果需要强度可以用随机或固定值
    intensity = np.full((N,), intensity_val, dtype=np.uint8)  # 0~255
    lidar_type_arr = np.full((N,), lidar_type, dtype=np.uint8)

    # 拼接最终 Nx6 数组
    data = np.column_stack([points_xyz, intensity, np.full(N, timestamp), lidar_type_arr])

    # 写头部
    header = f"""# .PCD v0.7 - Point Cloud Data file format
            VERSION 0.7
            FIELDS x y z intensity timestamp lidar_type
            SIZE 4 4 4 1 8 1
            TYPE F F F U F U
            COUNT 1 1 1 1 1 1
            WIDTH {N}
            HEIGHT 1
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS {N}
            DATA ascii
                """

    # 写文件
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.10f %.10f %.10f %d %d %d")

from multiprocessing import Pool, cpu_count

def find_height_star(args):
    """多进程需要传单个参数，用 *args 包装"""
    x, y, csv_mapping_list = args
    return find_height(x, y, csv_mapping_list)

def gen_pseudo_pcd(csv_mapping_list, T_utm_common_cv):
    from scipy.interpolate import RegularGridInterpolator

    bev_range = [200, 200] # 200m x 200m

    # 1. 生成粗分辨率 local 网格 (1m)
    step_coarse = 2.0
    x_coarse = np.arange(-int(bev_range[0]/2), int(bev_range[0]/2) + step_coarse, step_coarse)
    y_coarse = np.arange(-int(bev_range[1]/2), int(bev_range[1]/2) + step_coarse, step_coarse)
    Xc, Yc = np.meshgrid(x_coarse, y_coarse)
    points_local_coarse = np.stack([Xc.ravel(), Yc.ravel(), np.zeros_like(Xc.ravel())], axis=-1)

    # 2. 转换到 UTM
    points_utm_coarse = (T_utm_common_cv @ np.c_[points_local_coarse, np.ones(len(points_local_coarse))].T).T[:, :3]

    # 3. 用逐点查找高度
    heights_coarse = np.array([
        find_height(x, y, csv_mapping_list)
        for x, y in tqdm(zip(points_utm_coarse[:,0], points_utm_coarse[:,1]))
    ])

    points_utm_coarse[:,2] = heights_coarse
    ones_col = np.ones((points_utm_coarse.shape[0], 1))
    T_common_utm = np.linalg.inv(T_utm_common_cv)
    point_utm = np.hstack((points_utm_coarse, ones_col))
    local_points = np.dot(T_common_utm, point_utm.T).T
    height_grid_coarse = local_points[:, 2]

    # 4. 将粗网格高度 reshape 成二维
    height_grid_coarse = height_grid_coarse.reshape(len(y_coarse), len(x_coarse))

    # 5. 用双线性插值放大到 0.1m
    # 目标 finer grid
    step_fine = 0.5
    x_fine = np.arange(-int(bev_range[0]/2), int(bev_range[0]/2) + step_fine, step_fine)
    y_fine = np.arange(-int(bev_range[0]/2), int(bev_range[0]/2) + step_fine, step_fine)

    # 构建插值器
    interp_func = RegularGridInterpolator(
        (y_coarse, x_coarse),  # 注意顺序 (y, x)
        height_grid_coarse,
        bounds_error=False,
        # fill_value=12.7660
    )

    # 对 finer grid 插值
    Xf, Yf = np.meshgrid(x_fine, y_fine)
    points_fine = np.stack([Xf.ravel(), Yf.ravel()], axis=-1)
    heights_fine = interp_func(points_fine)

    # 得到高分辨率地面点
    points_local_fine = np.c_[points_fine, heights_fine]

    return points_local_fine

def find_height(utm_x, utm_y, csv_maping_list):
    def find_pos_z(mapping, x, y, tol=0.3):
        x_map = mapping[:, :, 0]
        y_map = mapping[:, :, 1]
        mask = (np.abs(x_map - x) < tol) & (np.abs(y_map - y) < tol)
        indices = np.argwhere(mask)
        if indices.size == 0: return None, None
        row, col = indices[0]
        return (row, col), mapping[row, col, 2]
    zs = [z for m in csv_maping_list if (p := find_pos_z(m, utm_x, utm_y))[1] is not None for z in [p[1]]]
    return np.mean(zs) if zs else 12.7660

#上海5号路口
# calib_path = '/adga/lushiyong/calibration_verify/BevCalib_lsy/20005博园路与墨玉南路交叉口20250521(F)/output/cam_ground_extrinsic_calib_result_30_common.yaml'
# csv_root = '/adga/lushiyong/bev_data/lut/sh_i5'
#上海2号路口
# calib_path = '/adga/lushiyong/calibration_verify/BevCalib_lsy/20002米泉南路和曹安公路0402.trans/output/cam_ground_extrinsic_calib_result_s0_0_30_common.yaml'
# csv_root = '/adga/lushiyong/bev_data/lut/sh_i2'
# 武汉wh_ccddxhl路口
# calib_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-26武汉4路口纯视觉BEV/车城大道与兴华路/output/cam_ground_extrinsic_calib_result_s1_0_common.yaml'
# csv_root = '/adga/lushiyong/bev_data/lut/wh_ccddxhl'
calib_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-26武汉4路口纯视觉BEV/车城大道与创业四路/output/cam_ground_extrinsic_calib_result_s1_0_common.yaml'
csv_root = '/adga/lushiyong/bev_data/lut/wh_i3'

save_pcd_name = 'wh_i3_pseudo_ground.pcd'

csv_maping_list = []


if __name__ == '__main__':
    csv_path_list = [
                        # csv_root + "/0/cam_ground_extrinsic_calib_result_0_caliblut.csv",
                        # csv_root + "/0/cam_ground_extrinsic_calib_result_8_caliblut.csv",
                        # csv_root + "/1/cam_ground_extrinsic_calib_result_0_caliblut.csv",
                        # csv_root + "/1/cam_ground_extrinsic_calib_result_8_caliblut.csv",
                        # csv_root + "/2/cam_ground_extrinsic_calib_result_0_caliblut.csv",
                        # csv_root + "/2/cam_ground_extrinsic_calib_result_8_caliblut.csv",
                        # csv_root + "/3/cam_ground_extrinsic_calib_result_0_caliblut.csv",
                        # csv_root + "/3/cam_ground_extrinsic_calib_result_8_caliblut.csv",
                        csv_root + "/0/s0_0.csv",
                        csv_root + "/0/s0_8.csv",
                        csv_root + "/1/s1_0.csv",
                        csv_root + "/1/s1_8.csv",
                        csv_root + "/2/s2_0.csv",
                        csv_root + "/2/s2_8.csv",
                        csv_root + "/3/s3_0.csv",
                        csv_root + "/3/s3_8.csv",
                    ]

    for i, csv_path in enumerate(csv_path_list):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, header=None, skiprows=1).values
            mapping = df.reshape((1024, 1024, 3)) if i % 2 else df.reshape((720, 1280, 3))
            # mapping = df.reshape((972, 1344, 3)) if i % 2 else df.reshape((720, 1280, 3))
            # mapping = df
            csv_maping_list.append(mapping)

    T_utm_common_cv = np.array(read_yaml_file(calib_path)['T_utm_common_cv']).reshape((-1, 4))

    points_local_fine = gen_pseudo_pcd(csv_maping_list, T_utm_common_cv)
    save_pcd(points_local_fine, save_pcd_name, intensity_val=40, lidar_type=0)

    print("伪点云已生成")
