
#多线程
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# txt_path = '/adga/lushiyong/bev_data/adp_data/tx_6000/kitti_result/tx1_6057.txt'
# src_data_root = '/adga/lushiyong/bev_data/adp_data/tx_6000/kitti_result'
# dst_data_root = '/adga/lushiyong/bev_data/anno_data/road_mix'

txt_path = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/kitti_result/hm_i7_0830_5s.txt'
src_data_root = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/kitti_result'
dst_data_root = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830_1876/send/1/'

# 读取文件名列表
with open(txt_path, 'r') as f:
    # i10_sift_filename = [line.strip() for line in f]
    i7_sift_filename = [line.strip() for line in f]

# 定义路径
src_img_dir = os.path.join(src_data_root, 'images')

dst_img_dir = dst_data_root


# 确保目标目录存在
os.makedirs(dst_img_dir, exist_ok=True)

for i in range(4):
    os.makedirs(os.path.join(dst_img_dir, f'camera_{i}_0'), exist_ok=True)
    os.makedirs(os.path.join(dst_img_dir, f'camera_{i}_8'), exist_ok=True)


# 单个文件的复制逻辑
def copy_file(file):
    for i in range(4):
        # 普通相机图像
        img_name = file + '.jpg'
        img_name_dst = file +'.jpg'

        src_gun_img_path = os.path.join(src_img_dir, f'camera_{i}_0', img_name)
        dst_gun_img_path = os.path.join(dst_img_dir, f'camera_{i}_0', img_name_dst)
        print(f"copying {src_gun_img_path} to {dst_gun_img_path}")
        shutil.copy(src_gun_img_path, dst_gun_img_path)

        # 鱼眼相机图像
        src_fish_img_path = os.path.join(src_img_dir, f'camera_{i}_8', img_name)
        dst_fish_img_path = os.path.join(dst_img_dir, f'camera_{i}_8', img_name_dst)
        print(f"copying {src_fish_img_path} to {dst_fish_img_path}")
        shutil.copy(src_fish_img_path, dst_fish_img_path)


# 使用线程池并行复制
with ThreadPoolExecutor(max_workers=30) as executor:
    list(tqdm(executor.map(copy_file, i7_sift_filename), total=len(i7_sift_filename)))