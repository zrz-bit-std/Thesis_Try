import os
import shutil
from tqdm import tqdm

# 定义源目录和目标目录
# ============================================================================================================================
root_path = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/kitti_result'
# source_image_dir = os.path.join(root_path, 'images/camera_0_0')
txt_path = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/kitti_result/hm_i7_0830_5s.txt'
# root_path = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830'
# txt_path = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/wh_i4_0830_5s.txt'


calib_source_file = os.path.join(root_path, 'calib_bev.txt')
target_calib_dir = os.path.join(root_path, 'calib')
# ============================================================================================================================

# 确保目标目录存在，如果不存在则创建
if not os.path.exists(target_calib_dir):
    os.makedirs(target_calib_dir)

# 读取calib.txt文件内容
with open(calib_source_file, 'r') as f:
    calib_content = f.read()

# 获取源目录中的所有文件名（假设都是图像文件）
file_names = []
print('preparing file names...')
# for f in tqdm(os.listdir(source_image_dir)):
#     if os.path.isfile(os.path.join(source_image_dir, f)):
#          file_names.append(f)
with open(txt_path, 'r') as f:
    # i10_sift_filename = [line.strip() for line in f]
    file_names = [line.strip() for line in f]

# 对于每一个文件名，创建对应的calib文件并写入内容
print('generating calib file...')
for file_name in tqdm(file_names):
    # 假设你想使用文件名（不包括扩展名）作为新calib文件的名称
    # base_name, _ = os.path.splitext(file_name)
    base_name = file_name
    target_calib_file = os.path.join(target_calib_dir, f'{base_name}.txt')

    with open(target_calib_file, 'w') as f:
        f.write(calib_content)
    
    # print(f'Created {target_calib_file}')

print('Calibration files generation completed.')