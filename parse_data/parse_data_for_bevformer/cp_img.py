#多线程
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


txt_path = '/adga/lushiyong/bev_data/anno_data/road_mix/ImageSet/hy_i7_i10_sh_i2_i3_i4_i5_i6_tx_i1_mix_77187_valset.txt'
src_data_root = '/adga/lushiyong/bev_data/anno_data/road_mix'
dst_data_root = '/adga/lushiyong/vis_anno/dataset/sh_i2/4d_val_set/send/1'

# 读取文件名列表
with open(txt_path, 'r') as f:
    i7_sift_filename = [line.strip() for line in f]

# 定义路径
src_img_dir = os.path.join(src_data_root, 'images')
dst_img_dir = dst_data_root

# 确保目标目录存在
os.makedirs(dst_img_dir, exist_ok=True)

img_dir = [
    'camera_0_0',
    'camera_1_0',
    'camera_2_0',
    'camera_3_0',
    'camera_0_8',
    'camera_1_8',
    'camera_2_8',
    'camera_3_8',
]

for i in img_dir:
    os.makedirs(os.path.join(dst_img_dir, i), exist_ok=True)

# 单个文件的复制逻辑
def copy_file(file):
    for i in range(4):
        # 普通相机图像
        img_name = file + '.jpg'
        img_name_dst = file +'.jpg'

        src_gun_img_path = os.path.join(src_img_dir, f'camera_{i}', img_name)
        dst_gun_img_path = os.path.join(dst_img_dir, img_dir[i], img_name_dst)
        print(f"copying {src_gun_img_path} to {dst_gun_img_path}")
        shutil.copy(src_gun_img_path, dst_gun_img_path)

        # 鱼眼相机图像
        src_fish_img_path = os.path.join(src_img_dir, f'camera_{i}_fisheye', img_name)
        dst_fish_img_path = os.path.join(dst_img_dir, img_dir[i+4], img_name_dst)
        print(f"copying {src_fish_img_path} to {dst_fish_img_path}")
        shutil.copy(src_fish_img_path, dst_fish_img_path)



# 使用线程池并行复制
with ThreadPoolExecutor(max_workers=30) as executor:
    list(tqdm(executor.map(copy_file, i7_sift_filename), total=len(i7_sift_filename)))