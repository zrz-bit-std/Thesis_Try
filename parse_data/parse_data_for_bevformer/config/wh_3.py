# DAN 数据目录
data_root = '/adp/zhujiawei/data/origin_data/wuhan/3d_road/train_wh_3d_road_3_20250904_lx'

# 目标保存文件夹
dst_root_dir = '/adga/lushiyong/vis_anno/dataset/wh_i3_0904'

# 伪点云路径
pcd_source_file = '/adga/lushiyong/vis_anno/pseudo_pcd/wh_i3.pcd'

# 标定文件目录
intrin_yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-26武汉4路口纯视觉BEV/车城大道与创业四路/input/cam'
yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-8-26武汉4路口纯视觉BEV/车城大道与创业四路/output'

# 路口名
road_info = '_wh3'

# DAN 图像数据子目录
img_dirs = [
    'camera_0_0',
    'camera_1_0',
    'camera_2_0',
    'camera_3_0',
    'camera_0_8',
    'camera_1_8',
    'camera_2_8',
    'camera_3_8',
]

# 相机对应标定文件名称
calib_dict = {
        'camera_0_0': 's0_0',
        'camera_1_0': 's1_0',
        'camera_2_0': 's2_0',
        'camera_3_0': 's3_0',
        'camera_0_8': 's0_8',
        'camera_1_8': 's1_8',
        'camera_2_8': 's2_8',
        'camera_3_8': 's3_8',
        }