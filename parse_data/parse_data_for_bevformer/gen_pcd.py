import os
import shutil
from tqdm import tqdm

# 定义源目录和目标目录
# ============================================================================================================================
# root_path = '/adga/lushiyong/vis_anno'
# source_image_dir = '/adga/lushiyong/vis_anno/dataset/sh_i2/4d_val_set/result_json'
# pcd_source_file = os.path.join(root_path, 'sh_i2_pseudo_ground.pcd')
# target_pcd_dir = '/adga/lushiyong/vis_anno/dataset/sh_i2/4d_val_set/send/1/3d_url/'

root_path = '/adga/lushiyong/vis_anno'
source_image_dir = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830_1876/result_json'
pcd_source_file = os.path.join(root_path, 'hm_i7_ground.pcd')
target_pcd_dir = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830_1876/send/1/3d_url/'
# ============================================================================================================================

# 确保目标目录存在，如果不存在则创建
if not os.path.exists(target_pcd_dir):
    os.makedirs(target_pcd_dir)

# 读取calib.txt文件内容
with open(pcd_source_file, 'r') as f:
    calib_content = f.read()

# 获取源目录中的所有文件名（假设都是图像文件）
file_names = []
print('preparing file names...')
for f in tqdm(os.listdir(source_image_dir)):
    if os.path.isfile(os.path.join(source_image_dir, f)):
         file_names.append(f)

# 对于每一个文件名，创建对应的calib文件并写入内容
print('generating calib file...')
for file_name in tqdm(file_names):
    # 假设你想使用文件名（不包括扩展名）作为新calib文件的名称
    base_name, _ = os.path.splitext(file_name)
    # base_name = file_name
    target_calib_file = os.path.join(target_pcd_dir, f'{base_name}.pcd')

    with open(target_calib_file, 'w') as f:
        f.write(calib_content)
    
    print(f'Created {target_calib_file}')

print('Calibration files generation completed.')