import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 创建 1280x720 的黑色图像 (全为0)
img = np.zeros((720, 1280, 3), dtype=np.uint8)

# files_dir = '/adga/lushiyong/bev_data/dataset/sh_i6_0325/images/camera_0'
txt_path = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/wh_i4_0830_5s.txt'

# files_dir = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/images/camera_1_0'

target_dirs = [
    # '/adga/lushiyong/bev_data/dataset/sh_i6_0501/images/camera_2',
    # '/adga/lushiyong/bev_data/dataset/sh_i6_0501/images/camera_2_fisheye',
    # '/adga/lushiyong/bev_data/adp_data/sh_i4_9013/kitti_result/camera_2',
    # '/adga/lushiyong/bev_data/adp_data/sh_i4_9013/kitti_result/camera_2_fisheye',
    '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/images/camera_0',
    '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/images/camera_0_fisheye',
               ]

# 读取文件名列表
with open(txt_path, 'r') as f:
    # i10_sift_filename = [line.strip() for line in f]
    files_list = [line.strip() for line in f]

# files_list = os.listdir(files_dir)

def split_list(lst, n):
    # 将列表 lst 均匀切分为 n 块
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def gen_blank_images(files_list):
    for target_dir in target_dirs:
        os.makedirs(target_dir, exist_ok=True)

        for file in files_list:
            file = os.path.splitext(file)[0]+'.jpg'
            # file = file+'.jpg'
            dst_path = os.path.join(target_dir, file)
            print(f'Writing {dst_path}')
            cv2.imwrite(dst_path, img)

if __name__ == '__main__':
    thread_num = 16
    chunks = split_list(files_list, thread_num)
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        executor.map(gen_blank_images, chunks)