import glob
import os
import pickle
import numpy as np
from sklearn.cluster import KMeans



def get_3d_anchors(dimensions, num_anchors=3, random_state=42):
    """
    通过 k-means 聚类从 3D 尺寸数据中得到 anchor 尺寸
    Args:
        dimensions (ndarray): Nx3 数组，每一行是 [l, w, h]
        num_anchors (int): anchor 数量
        random_state (int): 随机种子，保证可复现
    Returns:
        anchors (ndarray): num_anchors x 3，代表 anchor 尺寸
    """
    # 检查输入
    if not isinstance(dimensions, np.ndarray):
        dimensions = np.array(dimensions)
    assert dimensions.shape[1] == 3, "输入必须是 Nx3 的 [l, w, h] 数组"

    # KMeans 聚类
    kmeans = KMeans(n_clusters=num_anchors, random_state=random_state, n_init=10)
    kmeans.fit(dimensions)
    
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0])]  # 按长度排序，方便阅读
    return anchors

def extract_lwh_from_pkls(pkl_dir):
    """遍历多个目录，提取所有 object 的 lwh，并输出每个文件的框数量"""
    all_lwh = []
    file_counts = {}

    all_pkl_files = []
    for pkl_dir in pkl_dirs:
        pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
        all_pkl_files.extend(pkl_files)

    print(f"总共找到 {len(all_pkl_files)} 个 pkl 文件\n")

    for pkl_file in all_pkl_files:
        with open(pkl_file, "rb") as f:
            pkl_data = pickle.load(f)

        file_lwh = []

        for obj_id, obj_data in pkl_data.items():
            if "gt_boxes_global" not in obj_data:
                continue

            boxes = np.array(obj_data["gt_boxes_global"])  # Nx7
            if boxes.ndim != 2 or boxes.shape[1] < 6:
                continue  # 数据不符合预期，跳过

            lwh = boxes[:, 3:6]  # 取 l, w, h
            file_lwh.append(lwh)

        if file_lwh:
            file_lwh = np.vstack(file_lwh)
            all_lwh.append(file_lwh)
            file_counts[os.path.basename(pkl_file)] = file_lwh.shape[0]
        else:
            file_counts[os.path.basename(pkl_file)] = 0

    if not all_lwh:
        raise ValueError("没有提取到任何 lwh 数据，请检查 pkl 文件内容")

    all_lwh = np.vstack(all_lwh)
    return all_lwh, file_counts


if __name__ == '__main__':

    # ===== 设置你的 pkl 文件路径 =====
    pkl_dirs = ["/data1/turbo_data/yuanqingwen/origin_version/DetZero/data/align/train/Truck", 
                "/data1/turbo_data/yuanqingwen/origin_version/DetZero/data/align/test/Truck"]

    # 1. 提取所有 lwh，并统计每个文件的数量
    all_lwh, file_counts = extract_lwh_from_pkls(pkl_dirs)

    print("每个文件的框数量：")
    for fname, count in file_counts.items():
        print(f"{fname}: {count} 个框")
    print(f"总共提取到 {all_lwh.shape[0]} 个{os.path.basename(pkl_dirs[0])}尺寸样本")

    # 2. 计算 anchors
    anchors = get_3d_anchors(all_lwh, num_anchors=3)
    print("\n推荐的 Anchor 尺寸 (长, 宽, 高):")
    for a in anchors:
        print(f"{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}")