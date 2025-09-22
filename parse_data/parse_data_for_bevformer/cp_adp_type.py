

import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# data_root = '/adp/zhujiawei/data/origin_data/shanghai/3d_road/train_sh_3d_road_5_20250528_lx'
# data_root = '/adp/zhujiawei/data/origin_data/xiongan/beijing/train_bj_3d_road_6_20250821_lx'
# data_root = '/adp/zhujiawei/data/origin_data/xiongan/beijing/train_bj_3d_road_7_20250821_lx'
# data_root = '/adp/zhujiawei/data/origin_data/beijing/3d_road/train_bj_3d_road_6_20250830_lx'
# data_root = '/adp/zhujiawei/data/origin_data/xiongan/3d_road/train_bj_3d_road_7_20250830_lx'
data_root = '/adp/zhujiawei/data/origin_data/wuhan/3d_road/train_wh_3d_road_4_20250903_lx/'


# dst_dir = '/adga/lushiyong/bev_data/dataset/sh_i5_0528_5w/images/'
# dst_dir = '/adga/lushiyong/vis_anno/dataset/hm_i6_0830/images/'
# dst_dir = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/images/'
dst_dir = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/images/'
# dst_dir = '/adga/lushiyong/vis_anno/dataset/hm_i7/images/'

road_info = '_wh4'

data_dirs = os.listdir(data_root)

img_dirs = [
    # 'camera_0_0',
    'camera_1_0',
    'camera_2_0',
    'camera_3_0',
    # 'camera_0_8',
    'camera_1_8',
    'camera_2_8',
    'camera_3_8',
]

def copy_image(data_dir, img_dir):
    try:
        img_dir_path = os.path.join(data_root, data_dir, img_dir)
        tmp_dst_dir = os.path.join(dst_dir, img_dir)
        os.makedirs(tmp_dst_dir, exist_ok=True)
        # img_name = os.listdir(img_dir_path)[0]
        for img_name in os.listdir(img_dir_path):
            img_path = os.path.join(img_dir_path, img_name)

            dst_img_name = img_name[:10] + img_name[11:14] + road_info + '.jpg'
            dst_img_path = os.path.join(tmp_dst_dir, dst_img_name)

            shutil.copy(img_path, dst_img_path)
    except Exception as e:
        print(f'Error copying {img_dir} in {data_dir}: {e}')

# 准备任务列表
tasks = []
for data_dir in data_dirs:
    if not os.path.isdir(os.path.join(data_root, data_dir)):
        continue
    for img_dir in img_dirs:
        tasks.append((data_dir, img_dir))

# 开多线程执行任务
with ThreadPoolExecutor(max_workers=30) as executor:
    futures = [executor.submit(copy_image, data_dir, img_dir) for data_dir, img_dir in tasks]
    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass

print('✅ All images copied.')