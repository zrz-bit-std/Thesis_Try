# -*- coding: utf-8 -*-
# 导入所需的库和模块
import os                          # 操作系统接口，用于文件和目录操作
import json                        # JSON数据处理
import cv2                         # OpenCV库，用于图像处理
import glob                        # 文件路径匹配
import argparse                    # 命令行参数解析
import math                        # 数学计算
import pickle                      # Python对象序列化
import numpy as np                 # 数值计算库
from tqdm import tqdm              # 进度条显示
import subprocess                  # 子进程管理，用于执行外部命令
from collections import defaultdict # 默认字典数据结构

# 将项目路径添加到系统路径中，以便导入自定义模块
import sys
sys.path.insert(0, "/rss/zhanghui/4D_label")
# 导入可视化工具中的颜色定义
from visual_data.utils.visualize import _COLORS  # 若未使用，可保留以便扩展

# 设置可视化环境变量
# DISPLAY环境变量用于指定X Server显示设备
os.environ["DISPLAY"] = ":0"
# PYOPENGL_PLATFORM环境变量用于指定OpenGL平台
os.environ["PYOPENGL_PLATFORM"] = "egl"

# ============================= 基础工具 =============================

def images_to_video_ffmpeg(image_folder, output_video, fps=10, limit_first_n=None, slow_factor=1.0):
    """
    将图像文件夹中的图片序列转换为视频文件（使用 ffmpeg concat）
    """
    import os, subprocess

    # 1) 收集帧
    files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
    if limit_first_n is not None:
        files = files[:limit_first_n]
    if not files:
        print(f"[WARN] No images in {image_folder}, skip making video.")
        return

    # 2) 生成临时 filelist（写入绝对路径，配合 -safe 0 最稳）
    filelist_txt = os.path.join(image_folder, "__filelist.txt")
    with open(filelist_txt, "w") as f:
        for file in files:
            abs_path = os.path.join(image_folder, file)
            f.write(f"file '{abs_path}'\n")

    # 3) 组合滤镜：慢放(可选) + 你的 scale/pad/setsar
    vf_parts = []
    if slow_factor and slow_factor > 1.0:
        vf_parts.append(f"setpts={float(slow_factor):.3f}*PTS")
    vf_parts.append("scale=1280:1280:force_original_aspect_ratio=decrease")
    vf_parts.append("pad=1280:1280:(ow-iw)/2:(oh-ih)/2")
    vf_parts.append("setsar=1")
    vf_arg = ",".join(vf_parts)

    # 4) 正确传入 filelist 的“变量”，而不是字面量字符串
    cmd = [
        "/usr/bin/ffmpeg",
        "-v", "error",              # 仅打印错误，干净一点；调试可改成 info
        "-f", "concat",
        "-safe", "0",
        "-i", filelist_txt,         # ✅ 用变量（包含绝对路径）
        "-r", str(fps),             # 输出帧率
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-vf", vf_arg,              # ✅ 使用组合好的滤镜
        "-y",
        output_video
    ]

    print("[CMD]", " ".join(cmd))

    # 5) 执行并捕获错误，便于定位问题
    try:
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            # -v error 时通常只有错误；如果你想看详细过程，把 -v 改为 info
            print(completed.stderr)
    except subprocess.CalledProcessError as e:
        print("[ERROR] ffmpeg failed:")
        print("STDOUT:\n", e.stdout or "")
        print("STDERR:\n", e.stderr or "")
    finally:
        # 6) 清理临时文件
        try:
            os.remove(filelist_txt)
        except Exception:
            pass

def rotate_points(points, yaw):
    """
    根据偏航角(yaw)旋转3D点
    参数:
        points: 3D点坐标 (3xN矩阵)
        yaw: 偏航角(弧度)
    返回:
        旋转后的点坐标
    """
    # 构建绕Z轴旋转的旋转矩阵
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],   # X轴旋转分量
        [np.sin(yaw),  np.cos(yaw), 0],   # Y轴旋转分量
        [0,            0,           1]    # Z轴旋转分量
    ])
    # 应用旋转矩阵到点坐标
    return rotation_matrix @ points


def project_3d_to_pixel(points, image_shape, intrinsic_matrix, extrinsic_matrix,
                        distortion_params, filter_z_camera=True, depth=None):
    """
    将3D点投影到图像像素坐标（支持鱼眼畸变）
    参数:
        points: 3D点坐标 [x, y, z]
        image_shape: 图像尺寸 [height, width]
        intrinsic_matrix: 相机内参矩阵 (3x3)
        extrinsic_matrix: 相机外参矩阵 (4x4)
        distortion_params: 畸变参数 [k1, k2, k3, k4] 或 None
        filter_z_camera: 是否过滤Z坐标为负的点
        depth: 深度阈值
    返回:
        u, v: 投影后的像素坐标
        valid: 投影点是否在图像范围内
    """
    # 提取3D点坐标
    x, y, z = points
    # 构建齐次坐标 [x, y, z, 1]
    point_3d_homogeneous = np.array([[x], [y], [z], [1]])
    # 将3D点从世界坐标系转换到相机坐标系
    point_camera_coords = np.dot(extrinsic_matrix[:3, :], point_3d_homogeneous)

    # 处理鱼眼/畸变情况
    if distortion_params is not None:
        # 提取畸变参数
        k1, k2, k3, k4 = distortion_params
        # 提取相机坐标系中的点坐标
        x_camera, y_camera, z_camera = point_camera_coords.flatten()[:3]
        # 如果需要过滤且Z坐标为负，则返回无效点
        if filter_z_camera and z_camera < 0:
            return None, None, False
        # 如果设置了深度阈值且Z坐标小于阈值，则返回无效点
        if depth is not None and z_camera < depth:
            return None, None, False

        # 归一化图像坐标
        normalized_x = x_camera / max(z_camera, 1e-6)
        normalized_y = y_camera / max(z_camera, 1e-6)
        # 计算到图像中心的距离
        r = np.sqrt(normalized_x ** 2 + normalized_y ** 2)
        # 如果距离接近0，则直接投影原点
        if r < 1e-8:
            projected_point = np.dot(intrinsic_matrix, np.array([[0], [0], [1]]))
            u = projected_point[0, 0]
            v = projected_point[1, 0]
        else:
            # 计算畸变后的坐标
            theta = np.arctan(r)  # 计算角度
            # 应用鱼眼畸变模型
            theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
            # 计算缩放因子
            scale = (theta_d / max(r, 1e-6))
            # 计算畸变后的归一化坐标
            distorted_x = scale * normalized_x
            distorted_y = scale * normalized_y
            # 投影到像素坐标
            projected_point = np.dot(intrinsic_matrix, np.array([[distorted_x], [distorted_y], [1]]))
            u = projected_point[0, 0]
            v = projected_point[1, 0]
    else:
        # 处理传统针孔相机模型
        # 如果Z坐标为负，则返回无效点
        if point_camera_coords[2, 0] < 0:
            return None, None, False
        # 使用内参矩阵将相机坐标投影到图像坐标
        points_image_ = intrinsic_matrix @ point_camera_coords
        # 归一化齐次坐标
        points_image = points_image_ / max(points_image_[2, 0], 1e-6)
        # 提取像素坐标
        u = points_image[0, 0]
        v = points_image[1, 0]

    # 检查投影点是否在图像范围内
    if not (0 <= u < image_shape[1] and 0 <= v < image_shape[0]):
        return u, v, False
    # 返回有效的像素坐标
    return u, v, True

def _rgb_to_bgr_255(color):
    """
    输入 color: (r,g,b)
      - 若在 0~1 之间，按浮点归一化处理
      - 若已有 0~255，按整型处理
    返回: OpenCV 需要的 BGR(0~255)
    """
    r, g, b = float(color[0]), float(color[1]), float(color[2])
    if max(r, g, b) <= 1.0:
        return (int(b * 255), int(g * 255), int(r * 255))
    else:
        return (int(b), int(g), int(r))


def draw_3d_box_multi(image, corners_2ds, colors, thicknesses=None):
    """
    在图像上绘制多个3D包围框
    参数:
        image: 输入图像
        corners_2ds: 2D角点坐标列表
        colors: 颜色列表
    返回:
        绘制了3D框的图像
    """
    if thicknesses is None:
        thicknesses = [4] * len(corners_2ds)

    # 遍历每个3D框的角点和对应颜色
    for (corners_2d, color, thick) in zip(corners_2ds, colors, thicknesses):
        # 颜色从 [0,1] 归一化到 [0,255]
        # color_normalized = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        color_normalized= _rgb_to_bgr_255(color)
        # 绘制底面(4个点)
        for i in range(4):
            cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[(i + 1) % 4].astype(int)), color_normalized, thick)
        # 绘制顶面(4个点)
        for i in range(4, 8):
            cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[(i + 1) % 4 + 4].astype(int)), color_normalized, thick)
        # 绘制立柱(连接顶面和底面的4条线)
        for i in range(4):
            cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[i + 4].astype(int)), color_normalized, thick)
    # 返回绘制后的图像
    return image


def draw_box_on_img_per(params, image, cam2pixel, lidar2cam, distort=None):
    """
    将一个帧的所有框投影并绘制到图像上
    参数:
        params: 框参数列表，格式为 [([x,y,z,l,w,h,yaw], (r,g,b)), ...]
        image: 输入图像
        cam2pixel: 相机内参矩阵
        lidar2cam: LiDAR到相机的变换矩阵
        distort: 畸变参数，默认为None
    返回:
        绘制了3D框的图像
    """
    # 存储需要绘制的点和颜色
    draw_pts, colors, thicknesses = [], [], []
    # 遍历每个框参数
    for item in params:
        if len(item) == 2:
            param, color = item
            thickness = 4
        else:
            param, color, thickness = item
        # 提取框的参数
        x, y, z, l, w, h, yaw = param
        # 定义以目标中心为原点的8个顶点坐标
        corners_3d_obj = np.array([
            [-l/2, -w/2, -h/2],  # 顶点0: 左前下
            [ l/2, -w/2, -h/2],  # 顶点1: 右前下
            [ l/2,  w/2, -h/2],  # 顶点2: 右后下
            [-l/2,  w/2, -h/2],  # 顶点3: 左后下
            [-l/2, -w/2,  h/2],  # 顶点4: 左前上
            [ l/2, -w/2,  h/2],  # 顶点5: 右前上
            [ l/2,  w/2,  h/2],  # 顶点6: 右后上
            [-l/2,  w/2,  h/2],  # 顶点7: 左后上
        ])
        # 根据偏航角旋转顶点
        corners_3d_obj_roation = rotate_points(corners_3d_obj.T, yaw)
        # 将旋转后的顶点平移到世界坐标系中的实际位置
        corners_3d_rotation_points = np.array([[x, y, z]]).T + corners_3d_obj_roation

        # 存储投影后的2D点和有效标志
        corners_2d = []
        valid_flag = []
        # 获取图像尺寸
        image_shape = image.shape
        # 遍历每个3D顶点并投影到图像平面
        for pt in corners_3d_rotation_points.T:
            # 将3D点投影到像素坐标
            u, v, vali = project_3d_to_pixel(pt, image_shape, cam2pixel, lidar2cam, distort)
            # 如果投影失败则跳出循环
            if u is None and v is None:
                break
            # 如果点有效则添加标志
            if vali:
                valid_flag.append(1)
            # 添加投影后的点坐标
            corners_2d.append([u, v])

        # 如果所有8个点都有效且都成功投影，则添加到绘制列表
        if len(valid_flag) == 8 and len(corners_2d) == 8:
            draw_pts.append(np.array(corners_2d))
            colors.append(color)
            thicknesses.append(int(thickness))

    # 如果没有需要绘制的点，则直接返回原图
    if len(draw_pts) == 0:
        return image
    # 绘制所有3D框并返回图像
    return draw_3d_box_multi(image, draw_pts, colors, thicknesses)

# ============================= 解析与匹配 =============================

def _collect_all_files(root_dir, pattern="*.txt", track_mode=False):
    """
    递归收集一个目录下的所有匹配文件（仅一层/两层结构）
    如果是跟踪模式(track_mode=True)，则只收集splited目录下的文件
    参数:
        root_dir: 根目录路径
        pattern: 文件匹配模式，默认为"*.txt"
        track_mode: 是否为跟踪模式
    返回:
        匹配的文件路径列表
    """
    # 初始化文件列表
    files = []
    # 如果根目录不存在，则返回空列表
    if not os.path.isdir(root_dir):
        return files
    # 遍历根目录下的所有子目录
    for d in sorted(os.listdir(root_dir)):
        # 构建子目录路径
        p1 = os.path.join(root_dir, d)
        # 如果是目录
        if os.path.isdir(p1):
            # 如果是跟踪模式
            if track_mode:
                # 对于跟踪文件，只从splited目录收集
                splited_dir = os.path.join(p1, "splited")
                # 如果splited目录存在，则收集其中的文件
                if os.path.isdir(splited_dir):
                    files.extend(sorted(glob.glob(os.path.join(splited_dir, pattern))))
            else:
                # 对于检测文件，保持原有逻辑
                files.extend(sorted(glob.glob(os.path.join(p1, pattern))))
                # 第二层可能叫 splited
                p2 = os.path.join(p1, "splited")
                # 如果splited目录存在，则收集其中的文件
                if os.path.isdir(p2):
                    files.extend(sorted(glob.glob(os.path.join(p2, pattern))))
                # 对于merged目录，还需要检查子目录（如时间戳目录）
                for sub_d in sorted(os.listdir(p1)):
                    sub_p = os.path.join(p1, sub_d)
                    # 如果是目录，则收集其中的文件
                    if os.path.isdir(sub_p):
                        files.extend(sorted(glob.glob(os.path.join(sub_p, pattern))))
    # 返回文件列表
    return files


def _parse_result_files(result_files, color_rgb=(0,1,0), tag="DET", max_print_files=3, max_print_lines=5, line_thickness=4):
    """
    解析检测/跟踪结果文件，返回：
      - frame2objs: dict[str_frame_id] -> List[(params, color)]
      - frame_ids:  List[str_frame_id]
    同时打印若干文件的前五行用于快速核验。
    参数:
        result_files: 结果文件路径列表
        color_rgb: 颜色值(RGB)，默认为绿色(0,1,0)
        tag: 标签，用于日志输出，默认为"DET"
        max_print_files: 最大打印文件数，默认为3
        max_print_lines: 每个文件最大打印行数，默认为5
    返回:
        frame2objs: 帧ID到对象参数的映射字典
        frame_ids: 帧ID列表
    """
    # 初始化帧到对象的映射字典和帧ID列表
    frame2objs = defaultdict(list)
    frame_ids = []
    # 已打印文件计数器
    printed_files = 0

    # 遍历所有结果文件
    for res_file in result_files:
        # 对于merged目录结构，frame_id应该是文件名去掉扩展名
        # 例如：/path/to/20250531072723/1748647643.100004.txt -> 1748647643.100004
        # frame_id = os.path.splitext(os.path.basename(res_file))[0]
        # 提取帧ID(保留小数部分)
        frame_id = os.path.basename(res_file).replace(".txt", "")  # 保留小数

        # 将帧ID添加到列表中
        frame_ids.append(frame_id)

        # 尝试读取文件内容
        try:
            with open(res_file, "r") as f:
                lines = f.readlines()
        except Exception as e:
            # 如果读取失败，打印警告信息并继续处理下一个文件
            print(f"[WARN] {tag}: read fail {res_file}: {e}")
            continue

        # # 如果已打印文件数小于最大打印文件数
        # if printed_files < max_print_files:
        #     # 打印示例文件的前几行用于调试
        #     print(f"[DEBUG-{tag}] 示例文件: {res_file}（前{max_print_lines}行）")
        #     for line in lines[:max_print_lines]:
        #         print("     ", line.rstrip("\n"))
        #     # 增加已打印文件计数
        #     printed_files += 1

        # # 解析每行： type h w l x y z yaw score
        # for line in lines:
        #     # 分割行内容
        #     parts = line.strip().split()
        #     # 如果分割后的部分少于9个，则跳过该行
        #     if len(parts) < 9:
        #         continue
        #     # obj_type = parts[0]   # 对象类型，暂时不用
        #     # 提取对象参数 h, w, l, x, y, z
        #     # h, w, l, x, y, z, yaw = map(float, parts[1:8])
        #     h, w, l, x, y, z = map(float, parts[1:7])
        #     # 将yaw角度转换为弧度
        #     yaw = math.radians(float(parts[7]))   # 角度转弧度
        #     # score = float(parts[8])  # 置信度分数，暂时不用
        #     # 构造参数数组
        #     params = np.array([x, y, z, l, w, h, yaw], dtype=np.float32)
        #     # 将参数和颜色添加到对应帧ID的列表中
        #     frame2objs[frame_id].append((params, color_rgb, line_thickness))
        # 兼容两种格式:
        # 老格式:  type h w l x y z yaw score        (9 列)
        # # 新格式:  uid  type h w l x y z yaw score   (10 列) —— 需跳过首列 uid
        for line in lines:
            parts = line.strip().split()
            # 最少也得有 9 列
            if len(parts) < 9:
                continue

            # 如果有 >=10 列，默认认为首列是 uid，需要跳过
            base = 1 if len(parts) >= 10 else 0

            # 防御式判断：确保剩余字段足够
            # 需要: type (base+0), [h,w,l,x,y,z] (base+1 ~ base+6), yaw (base+7), score (base+8)
            if len(parts) < base + 9:
                continue

            # obj_type = parts[base + 0]    # 如需类别可启用
            try:
                h, w, l, x, y, z = map(float, parts[base + 1 : base + 7])
                yaw_deg = float(parts[base + 7])
                # score = float(parts[base + 8])  # 目前可视化不需要
            except ValueError:
                # 行内出现非法数值，跳过
                continue

            yaw = math.radians(yaw_deg)  # 角度→弧度
            params = np.array([x, y, z, l, w, h, yaw], dtype=np.float32)
            frame2objs[frame_id].append((params, color_rgb))


    # 打印解析完成信息
    print(f"[INFO-{tag}] 解析完成：{len(frame2objs)} 帧")
    # 返回帧到对象的映射字典和帧ID列表
    return frame2objs, frame_ids


def _load_test_json(test_json_path):
    """
    读取 test.json，返回各矩阵与可能的时间戳集合
    参数:
        test_json_path: test.json文件路径
    返回:
        cam2pixels: 相机内参字典
        lidar2cams: LiDAR到相机变换矩阵字典
        distort_params: 畸变参数字典
        ts_from_json: 时间戳集合
    """
    # 读取JSON文件
    with open(test_json_path, "r") as f:
        jd = json.load(f)
    # 如果是列表，则取第一个元素
    if isinstance(jd, list):
        jd = jd[0]
    # 合并普通相机和鱼眼相机的内参
    cam2pixels = jd['cam2img']
    cam2pixels.update(jd['cam2img_fisheye'])
    # 合并普通相机和鱼眼相机的外参
    lidar2cams = jd['lidar2cam']
    lidar2cams.update(jd['lidar2cam_fisheye'])
    # 获取鱼眼相机畸变参数
    distort_params = jd['distort_fisheye']

    # 可选：尝试取时间戳字段，用于匹配核对（不同项目结构字段名可能不同）
    ts_from_json = set()
    # 遍历可能包含时间戳的字段
    for key in ['frames', 'samples', 'timestamps']:
        # 如果字段存在且为列表
        if key in jd and isinstance(jd[key], list):
            # 遍历列表中的每个元素
            for it in jd[key]:
                # 允许元素是 dict 或 直接是字符串
                if isinstance(it, dict):
                    # 常见字段
                    for k in ['timestamp', 'ts', 'time', 'name']:
                        # 如果字段存在，则添加到时间戳集合中
                        if k in it:
                            ts_from_json.add(str(it[k]))
                else:
                    # 如果是字符串，则直接添加到时间戳集合中
                    ts_from_json.add(str(it))

    # 返回解析结果
    return cam2pixels, lidar2cams, distort_params, ts_from_json

# ============================= 可视化主流程 =============================

def _render_sequence(sequence, mode="overlay", make_video=True):
    """
    可视化序列的主函数
    参数:
        sequence: 序列名称
        mode: 可视化模式，可选"detection"(仅检测)、"tracking"(仅跟踪)、"overlay"(叠加)
        make_video: 是否生成视频
    mode:
      - "detection": 仅画检测
      - "tracking" : 仅画跟踪
      - "overlay"  : 同图叠加检测(绿) + 跟踪(红)
    """
    # 设置数据根路径
    root_path = "/rss/zhanghui/4D_label/dataset_track"
    # 定义针孔相机列表
    image_pinhole_list = ['camera_0_0', 'camera_1_0', 'camera_2_0', 'camera_3_0']
    # 定义鱼眼相机列表
    image_fisheye_list = ['camera_0_8', 'camera_1_8', 'camera_2_8', 'camera_3_8']
    # 设置图像保存尺寸
    image_save_size = (800, 800)

    # 构建test.json文件路径
    test_json = os.path.join(root_path, sequence, "scences/test.json")
    # 打印加载信息
    print(f"[INFO] Loading test.json: {test_json}")
    # 尝试加载test.json文件
    try:
        cam2pixels, lidar2cams, camera_distort_param, ts_from_json = _load_test_json(test_json)
    except Exception as e:
        # 如果加载失败，打印错误信息并返回
        print(f"[ERROR] load test.json fail: {e}")
        return

    # 收集所有检测/跟踪结果
    # 检测结果根目录
    det_root = os.path.join(root_path, "merged", sequence)  # 修改为merged目录
    # 跟踪结果根目录
    trk_root = os.path.join(root_path, "offline_tracked", sequence)

    # 收集检测文件和跟踪文件
    det_files = _collect_all_files(det_root, "*.txt", track_mode=False)
    trk_files = _collect_all_files(trk_root, "*.txt", track_mode=True)

    # 打印文件统计信息
    print(f"[INFO] DET files: {len(det_files)} | TRK files: {len(trk_files)}")

    # 初始化检测和跟踪的映射字典及ID列表
    det_map, det_ids = defaultdict(list), []
    trk_map, trk_ids = defaultdict(list), []

    if mode in ("detection", "overlay"):
        det_map, det_ids = _parse_result_files(det_files,
                                           color_rgb=(0, 1, 0),  # 绿色(归一化)
                                           tag="DET",
                                           line_thickness=8)     # 粗
    if mode in ("tracking", "overlay"):
        trk_map, trk_ids = _parse_result_files(trk_files,
                                           color_rgb=(1, 0, 0),  # 红色(归一化)
                                           tag="TRK",
                                           line_thickness=2)     # 细


    # 打印与 test.json 时间戳的匹配统计（若 test.json 含时间戳）
    if len(ts_from_json) > 0:
        # 计算检测和跟踪结果与时间戳的匹配数量
        det_hit = len([x for x in det_ids if x in ts_from_json])
        trk_hit = len([x for x in trk_ids if x in ts_from_json])
    #     # 打印匹配统计信息
    #     print(f"[INFO] 与 test.json 时间戳匹配：DET 命中 {det_hit}/{len(det_ids)}，TRK 命中 {trk_hit}/{len(trk_ids)}")
    # else:
    #     # 如果test.json中没有时间戳信息，则打印提示信息
    #     print("[INFO] test.json 中未找到时间戳信息")

    # # 打印一些det_map和trk_map中的键用于调试
    # print(f"[DEBUG] det_map中的部分键: {list(det_map.keys())[:10]}")
    # print(f"[DEBUG] trk_map中的部分键: {list(trk_map.keys())[:10]}")

    # 图像路径（以 camera_0_0 作为基准搜集所有帧）
    img_root_path = os.path.join(root_path, sequence, "samples", "camera_0_0")
    # 检查图像根路径是否存在
    if not os.path.isdir(img_root_path):
        # 如果不存在，打印错误信息并返回
        print(f"[ERROR] Image root not found: {img_root_path}")
        return
    # 获取所有.jpg图像文件并排序
    img_files = sorted(glob.glob(os.path.join(img_root_path, "*.jpg")))
    # 打印找到的图像文件数量
    print(f"[INFO] Found {len(img_files)} images under {img_root_path}")

    # # 输出一些图像文件名用于调试
    # print(f"[DEBUG] 部分图像文件名: {[os.path.basename(f) for f in img_files[:10]]}")
    
    # 根据模式设置输出目录
    out_root = {
        "detection":      os.path.join("/rss/zhanghui/4D_label/visual_combine/detection_results", sequence),
        "tracking":       os.path.join("/rss/zhanghui/4D_label/visual_combine/tracking_results", sequence),
        "overlay":        os.path.join("/rss/zhanghui/4D_label/visual_combine/det_trk_results", sequence),
    }[mode]
    # 如果目录已存在，先删除再创建，确保重新运行时会覆盖原来生成的文件
    if os.path.exists(out_root):
        import shutil
        shutil.rmtree(out_root)
    # 创建输出目录
    os.makedirs(out_root, exist_ok=True)

    # 初始化处理帧数和保存图像数计数器
    processed_frames = 0
    saved_images = 0

    # 创建一个映射，将文件名映射到完整路径，便于匹配
    det_filename_map = {os.path.splitext(os.path.basename(f))[0]: f for f in det_files}
    trk_filename_map = {os.path.splitext(os.path.basename(f))[0]: f for f in trk_files}
    
    # # 打印调试信息
    # print(f"[DEBUG] det_filename_map中的部分键: {list(det_filename_map.keys())[:10]}")
    # print(f"[DEBUG] trk_filename_map中的部分键: {list(trk_filename_map.keys())[:10]}")

    # 遍历所有图像文件
    for img0 in tqdm(img_files):
        # 从图像文件名中提取frame_id，需要去掉文件扩展名
        # frame_id = os.path.splitext(os.path.basename(img0))[0]
        # 提取帧ID(保留.jpg扩展名前的部分)
        frame_id = os.path.basename(img0).replace(".jpg", "")  
        # # 添加调试信息
        # if processed_frames < 10:  # 增加调试信息数量
        #     print(f"[DEBUG] 处理帧 {frame_id}: det_map包含该帧={frame_id in det_map}, trk_map包含该帧={frame_id in trk_map}")
        #     print(f"[DEBUG] det_filename_map包含该帧={frame_id in det_filename_map}, trk_filename_map包含该帧={frame_id in trk_filename_map}")

        # 聚合该帧要画的框
        obj_frame = []
        # 根据模式添加检测框
        if mode in ("detection", "overlay") and frame_id in det_map:
            obj_frame.extend(det_map[frame_id])
        # 根据模式添加跟踪框
        if mode in ("tracking", "overlay") and frame_id in trk_map:
            obj_frame.extend(trk_map[frame_id])

        # 如果没有任何框，跳过
        if not obj_frame:
            processed_frames += 1
            continue

        # 增加处理帧数计数
        processed_frames += 1
        # 获取图像文件名
        mark = os.path.basename(img0)
        # 获取图像文件所在目录
        base_dir = os.path.dirname(img0)  # .../samples/camera_0_0

        # 先画针孔四视角
        image_res_list = []
        # 遍历所有针孔相机
        for cam in image_pinhole_list:
            # 构建当前相机的图像路径
            img_path_cur = img0.replace("camera_0_0", cam)
            # 读取图像
            img = cv2.imread(img_path_cur)
            # 如果图像读取失败，打印警告信息并继续
            if img is None:
                print(f"[WARN] missing image: {img_path_cur}")
                continue
            # 提取当前相机的内参矩阵
            cam2pixel = np.array(cam2pixels[cam])[:3, :3]
            # 提取当前相机的外参矩阵
            lidar2cam = np.array(lidar2cams[cam])
            # 在图像上绘制3D框
            img = draw_box_on_img_per(obj_frame, img.copy(), cam2pixel, lidar2cam)
            # 调整图像尺寸
            img = cv2.resize(img, image_save_size)
            # 添加到结果列表
            image_res_list.append(img)

        # 如果针孔图像数量不足4个，则跳过该帧
        if len(image_res_list) < 4:
            # 针孔视角不齐，尽量跳过避免拼接错误
            print(f"[WARN] Not enough pinhole images for frame {frame_id}, skip.")
            continue
        # 水平拼接针孔图像
        pinhole_concat = np.concatenate((image_res_list[0], image_res_list[1], image_res_list[2], image_res_list[3]), axis=1)

        # 再画鱼眼四视角
        image_res_list = []
        # 遍历所有鱼眼相机
        for cam in image_fisheye_list:
            # 构建当前相机的图像路径
            img_path_cur = img0.replace("camera_0_0", cam)
            # 读取图像
            img = cv2.imread(img_path_cur)
            # 如果图像读取失败，打印警告信息并继续
            if img is None:
                print(f"[WARN] missing image: {img_path_cur}")
                continue
            # 提取当前相机的内参矩阵
            cam2pixel = np.array(cam2pixels[cam])[:3, :3]
            # 提取当前相机的外参矩阵
            lidar2cam = np.array(lidar2cams[cam])
            # 提取当前相机的畸变参数
            distort = camera_distort_param[cam]
            # 在图像上绘制3D框(考虑畸变)
            img = draw_box_on_img_per(obj_frame, img.copy(), cam2pixel, lidar2cam, distort)
            # 调整图像尺寸
            img = cv2.resize(img, image_save_size)
            # 添加到结果列表
            image_res_list.append(img)

        # 如果鱼眼图像数量不足4个，则跳过该帧
        if len(image_res_list) < 4:
            print(f"[WARN] Not enough fisheye images for frame {frame_id}, skip.")
            continue
        # 水平拼接鱼眼图像
        fisheye_concat = np.concatenate((image_res_list[0], image_res_list[1], image_res_list[2], image_res_list[3]), axis=1)

        # 垂直拼接针孔和鱼眼图像
        final_img = np.concatenate((pinhole_concat, fisheye_concat), axis=0)

        # 构建保存路径
        save_path = os.path.join(out_root, mark)
        # 保存图像
        cv2.imwrite(save_path, final_img)
        # 增加保存图像计数
        saved_images += 1

    # 打印处理完成信息
    print(f"[OK] Processed {processed_frames} frames, saved {saved_images} images to {out_root}")

    # 生成视频
    if make_video and saved_images > 0:
        # 根据模式设置视频输出根目录
        videos_root = {
            "detection": "/rss/zhanghui/4D_label/visual_combine/detection_videos",
            "tracking":  "/rss/zhanghui/4D_label/visual_combine/tracking_videos",
            "overlay":   "/rss/zhanghui/4D_label/visual_combine/overlay_videos",
        }[mode]
        # 构建视频序列路径
        video_sequence_path = os.path.join(videos_root, sequence)
        # 如果视频目录已存在，先删除再创建，确保重新运行时会覆盖原来生成的文件
        if os.path.exists(video_sequence_path):
            import shutil
            shutil.rmtree(video_sequence_path)
        # 创建视频输出目录
        os.makedirs(video_sequence_path, exist_ok=True)
        # 构建输出视频文件路径
        out_video = os.path.join(videos_root, sequence, f"{mode}.mp4")
        # 将图像序列转换为视频
        images_to_video_ffmpeg(out_root, out_video, fps=10, limit_first_n=None, slow_factor=1.0)
        # 打印视频生成完成信息
        print(f"[OK] {mode} video saved: {out_video}")
    else:
        # 如果没有生成视频，则打印提示信息
        print("[INFO] No video generated (either disabled or no images).")

# ============================= 兼容旧接口 =============================

def detection_results_on_multi_cams(sequence):
    """
    在多相机上显示检测结果的兼容接口
    参数:
        sequence: 序列名称
    """
    _render_sequence(sequence, mode="detection", make_video=True)

def tracking_results_on_multi_cams(sequence):
    """
    在多相机上显示跟踪结果的兼容接口
    参数:
        sequence: 序列名称
    """
    _render_sequence(sequence, mode="tracking", make_video=True)

def detection_and_tracking_results_on_multi_cams(sequence):
    """
    在多相机上叠加显示检测和跟踪结果的兼容接口
    参数:
        sequence: 序列名称
    """
    _render_sequence(sequence, mode="overlay", make_video=True)

def _has_encoder(name="libx264"):
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", f"encoder={name}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

def combine_detection_and_tracking_videos(sequence):
    """
    将 detection.mp4 与 tracking.mp4 横向拼接为 combined.mp4
    - 自动探测编码器：优先 libx264（CRF），否则降级 libopenh264（CBR）
    - 统一两路视频的帧率与高度，避免尺寸/时基不一致
    """
    import os, subprocess, shutil

    detection_videos_root = os.path.join("/rss/zhanghui/4D_label/visual_combine/detection_videos", sequence)
    tracking_videos_root  = os.path.join("/rss/zhanghui/4D_label/visual_combine/tracking_videos",  sequence)
    combined_videos_root  = os.path.join("/rss/zhanghui/4D_label/visual_combine/combined_videos", sequence)

    if os.path.exists(combined_videos_root):
        shutil.rmtree(combined_videos_root)
    os.makedirs(combined_videos_root, exist_ok=True)

    detection_video = os.path.join(detection_videos_root, "detection.mp4")
    tracking_video  = os.path.join(tracking_videos_root,  "tracking.mp4")
    if not os.path.exists(detection_video):
        print(f"[ERROR] Detection video not found: {detection_video}"); return
    if not os.path.exists(tracking_video):
        print(f"[ERROR] Tracking video not found: {tracking_video}"); return

    combined_video = os.path.join(combined_videos_root, "combined.mp4")

    # 统一帧率(按你生成图片时的 fps，比如 10) + 统一高度(1080，可按需改)
    filter_complex = (
        "[0:v]fps=10,scale=-2:1080:flags=bicubic[v0];"
        "[1:v]fps=10,scale=-2:1080:flags=bicubic[v1];"
        "[v0][v1]hstack=inputs=2[v]"
    )

    use_x264 = _has_encoder("libx264")

    if use_x264:
        # x264: CRF 模式（质量优先，文件体积可控）
        cmd = [
            "/usr/bin/ffmpeg", "-y", "-v", "error",
            "-i", detection_video, "-i", tracking_video,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-c:v", "libx264", "-crf", "23", "-preset", "slow",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            combined_video
        ]
    else:
        # openh264: 码率模式（这台环境会走这里）
        cmd = [
            "/usr/bin/ffmpeg", "-y", "-v", "error",
            "-i", detection_video, "-i", tracking_video,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-c:v", "libopenh264",
            "-profile:v", "main", "-level", "3.1", "-bf", "0",
            "-b:v", "6M", "-maxrate", "6M", "-bufsize", "12M",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            combined_video
        ]

    print("[CMD]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] Combined video saved: {combined_video}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] combine videos failed: {e}")

# ============================= CLI =============================

# 如果作为主程序运行
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加序列参数，必需
    parser.add_argument("--sequence", type=str, required=True, help="例如 train_sh_3d_road_2_20250531_5000_lx")
    # 添加根路径参数，可选，默认为'/rss/zhanghui/4D_label/origin'
    parser.add_argument('--root-path', type=str, default='/rss/zhanghui/4D_label/origin', help='保留以兼容旧参数')
    # 添加可视化类型参数，可选，默认为'detection'
    parser.add_argument('--vis-type', type=str, default='detection',
                        choices=['detection', 'tracking', 'both', 'overlay', 'combine'],
                        help="detection=只画检测; tracking=只画跟踪; both=先各自出视频再拼接; overlay=同图叠加检测+跟踪")
    # 解析命令行参数
    args = parser.parse_args()

    # 根据可视化类型调用相应函数
    if args.vis_type == 'detection':
        detection_results_on_multi_cams(args.sequence)
    elif args.vis_type == 'tracking':
        tracking_results_on_multi_cams(args.sequence)
    elif args.vis_type == 'overlay':
        detection_and_tracking_results_on_multi_cams(args.sequence)
    elif args.vis_type == 'both':
        # 生成各自视频 + 合并视频（左右对比）
        detection_results_on_multi_cams(args.sequence)
        tracking_results_on_multi_cams(args.sequence)
        combine_detection_and_tracking_videos(args.sequence)
    elif args.vis_type == 'combine':
        combine_detection_and_tracking_videos(args.sequence)