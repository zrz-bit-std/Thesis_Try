#多线程
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

source_image_dir = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830_1876/send/1/camera_0_0'
src_data_root = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830_1876'
dst_data_root = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830_1876_sample'


i7_sift_filename = []
print('preparing file names...')
for f in tqdm(os.listdir(source_image_dir)):
    if os.path.isfile(os.path.join(source_image_dir, f)):
         i7_sift_filename.append(f)
i7_sift_filename = i7_sift_filename[:20]

dst_img_dir = dst_data_root

# 确保目标目录存在
os.makedirs(dst_img_dir, exist_ok=True)

file_dir = [
    'result_json',
    'send/1/3d_url',
    'send/1/camera_0_0',
    'send/1/camera_1_0',
    'send/1/camera_2_0',
    'send/1/camera_3_0',
    'send/1/camera_0_8',
    'send/1/camera_1_8',
    'send/1/camera_2_8',
    'send/1/camera_3_8',
]

for i in file_dir:
    os.makedirs(os.path.join(dst_img_dir, i), exist_ok=True)

# 单个文件的复制逻辑
def copy_file(file):
    for i in file_dir:
        # 普通相机图像
        file_base_name = os.path.splitext(file)[0]
        if i == 'result_json':
            file_name = file_base_name + '.json'
            file_name_dst = file_base_name + '.json'
        elif i == 'send/1/3d_url':
            file_name = file_base_name + '.pcd'
            file_name_dst = file_base_name + '.pcd'
        else:
            file_name = file_base_name + '.jpg'
            file_name_dst = file_base_name + '.jpg'
        src_file_path = os.path.join(os.path.join(src_data_root, i), file_name)
        dst_file_path = os.path.join(os.path.join(dst_img_dir, i), file_name_dst)
        print(f"copying {src_file_path} to {dst_file_path}")
        shutil.copy(src_file_path, dst_file_path)


# 使用线程池并行复制
with ThreadPoolExecutor(max_workers=30) as executor:
    list(tqdm(executor.map(copy_file, i7_sift_filename), total=len(i7_sift_filename)))