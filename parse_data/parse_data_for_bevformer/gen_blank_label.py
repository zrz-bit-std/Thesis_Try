import os

files_dir = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/calib'
target_dir = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/label'
files_dir = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/calib'
target_dir = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/label'

os.makedirs(target_dir, exist_ok=True)

files_list = os.listdir(files_dir)

for file in files_list:
    file = os.path.splitext(file)[0]+'.txt'
    dst_path = os.path.join(target_dir, file)
    print(f'Writing {dst_path}')
    # os.remove(dst_path)
    with open(dst_path, 'w') as f:
        pass