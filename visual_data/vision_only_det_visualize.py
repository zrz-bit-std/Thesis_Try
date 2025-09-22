import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------- 工具函数 -----------

def gun_img_undistort(img, K, D):
    h, w = img.shape[:2]
    K = K[:3,:3]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)

    # initUndistortRectifyMap + remap 更高效，适合实时处理
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # roi 是有效区域，可以裁剪掉黑边
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    return undistorted, new_K

def invert_transform(Tr):
    """
    输入: Tr (4x4 齐次矩阵, Velodyne->Camera)
    输出: Tr_inv (4x4 齐次矩阵, Camera->Velodyne)
    """
    R = Tr[:3, :3]
    t = Tr[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    Tr_inv = np.eye(4)
    Tr_inv[:3, :3] = R_inv
    Tr_inv[:3, 3] = t_inv
    return Tr_inv

def load_calib(calib_file):
    """读取标定文件"""
    data = {}
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            key, value = line.strip().split(":", 1)
            data[key] = np.array([float(x) for x in value.strip().split()])

    calib = {}
    for i in range(8):
        P = data[f"P{i}"].reshape(3, 4)
        Tr = data[f"Tr_velo_to_cam_{i}"].reshape(3, 4)
        Tr = np.vstack([Tr, [0, 0, 0, 1]])
        calib[f"cam{i}"] = {"P": P, "Tr": Tr}
    return calib


def parse_label(label_file):
    """读取 KITTI label，返回 3D box"""
    boxes = []
    with open(label_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(" ")
            name = parts[0]
            h, w, l = map(float, parts[8:11])   # 尺寸
            x, y, z = map(float, parts[11:14]) # 中心点 (velo系)
            ry = float(parts[14])              # yaw
            boxes.append((name, h, w, l, x, y, z, ry))
    return boxes


def get_3d_box(h, w, l, x, y, z, ry):
    """生成 3D box 8角点 (velo系)"""
    # 物体局部坐标系下的8角点
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2,-w/2,-w/2,w/2, w/2,-w/2,-w/2,w/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]

    corners = np.array([x_corners, y_corners, z_corners])
    R = np.array([[np.cos(ry), -np.sin(ry), 0],
                  [np.sin(ry), np.cos(ry), 0],
                  [0, 0, 1]])
    corners_3d = R @ corners
    corners_3d = corners_3d + np.array([[x],[y],[z]])
    return corners_3d.T  # (8,3)


def project_points(pts_3d, P, Tr):
    """将 3D 点投影到图像"""
    pts_3d_h = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
    pts_cam = (Tr @ pts_3d_h.T).T
    pts_img = (P @ pts_cam.T).T

    depth = pts_img[:, 2]
    pts_img[:, 0] /= pts_img[:, 2]
    pts_img[:, 1] /= pts_img[:, 2]
    mask = depth > 0.5

    return pts_img[mask, :2]


def draw_3d_box(img, corners_2d):
    """在图像上绘制3D box"""
    corners_2d = corners_2d.astype(int)
    # 定义12条边
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), (0,255,0), 1)


def undistort_fisheye(img, K, D):
    """鱼眼图像去畸变"""
    h, w = img.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1)
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted, new_K

# ----------- 主函数 -----------

def visualize_scene(root_dir, frame_id=None):
    if frame_id == None:
        frame_id = os.path.splitext(os.listdir(os.path.join(root_dir, "calib"))[0])[0]
    img_dir = os.path.join(root_dir, "images")
    calib_file = os.path.join(root_dir, "calib", f"{frame_id}.txt")
    # label_file = os.path.join(root_dir, "label", f"{frame_id}.txt")
    label_file = os.path.join(root_dir, "infer_result_nms", f"{frame_id}.txt")

    calib = load_calib(calib_file)
    boxes = parse_label(label_file)

    # TODO: to read param by road_id!!!
    D_fisheye = np.array([-1.4700240669729036e-02, 
                          -2.1507305011098320e-03, 
                           2.5849931851393244e-04, 
                          -3.5575608063581203e-04])
    
    D_gun = np.array([-3.9833230828724620e-01, 1.6017667605991995e-01, 3.5443612047006859e-03, 7.8701121820389864e-04, 0.])

    results = []
    for i in range(8):
        if i < 4:  # 标准相机
            cam_name = f"camera_{i}"
        else:      # 鱼眼相机
            cam_name = f"camera_{i-4}_fisheye"

        img_path = os.path.join(img_dir, cam_name, f"{frame_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)

        P = calib[f"cam{i}"]["P"]
        Tr = invert_transform(calib[f"cam{i}"]["Tr"])

        if i >= 4:  # 鱼眼相机需要去畸变
            K = P[:, :3]
            img = cv2.resize(img[:,280:1000], (1024, 1024))
            img, P_new = undistort_fisheye(img, K, D_fisheye)
            P = np.hstack([P_new, np.zeros((3,1))])
        else:
            K = P[:, :3]
            h, w = img.shape[:2]
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D_gun, (w, h), 0, (w, h))
            img = cv2.undistort(img, K, D_gun, None, new_K)
            # img, new_K = gun_img_undistort(img, K, D_gun)

        for name,h,w,l,x,y,z,ry in boxes:
            corners_3d = get_3d_box(h,w,l,x,y,z,ry)
            # corners_3d = np.array([
            #     [ -9.7620347 ,  32.31879826,   1.668     ],
            #     [ -7.76456488,  32.05358266,   1.668     ],
            #     [ -8.3639653 ,  27.53920174,   1.668     ],
            #     [-10.36143512,  27.80441734,   1.668     ],
            #     [ -9.7620347 ,  32.31879826,   0.18      ],
            #     [ -7.76456488,  32.05358266,   0.18      ],
            #     [ -8.3639653 ,  27.53920174,   0.18      ],
            #     [-10.36143512,  27.80441734,   0.18      ]
            # ])
            pts_2d = project_points(corners_3d, P, Tr)
            if pts_2d.shape[0] >= 8:
                draw_3d_box(img, pts_2d)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results.append(img_rgb)

    # 拼图 2x4
    fig, axes = plt.subplots(2, 4, figsize=(40, 20), dpi=200) 
    for i, ax in enumerate(axes.flat):
        ax.imshow(results[i])
        ax.set_title(f"Cam{i}" if i<4 else f"Fish{i-4}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig('tmp.jpg')


if __name__ == "__main__":
    root_dir = "/rss/4D_label_dataset_vision_only/Intersection/train_bj_3d_road_7_20250919_lx-debug/kitti_result"   # 修改为你的数据集路径
    # frame_id = "1756955766059_wh3"                 # 修改为要可视化的帧id
    visualize_scene(root_dir, frame_id=None)