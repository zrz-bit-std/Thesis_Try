# DAN 数据目录
data_root = '/rss/RALG/data/3.0_pro/Inter+Road/label/shanghai_road_6/p2_lx/train_sh_3d_road_6_20250517_5000_lx/original_data'

# 目标保存文件夹
dst_root_dir = '/adga/lushiyong/vis_anno/dataset/4d/train_sh_3d_road_6_20250517_5000_lx'

# 伪点云路径, 先用环贸7凑凑数
pcd_source_file = '/adga/lushiyong/vis_anno/pseudo_pcd/hm_i7_ground.pcd'

# 标定文件目录
intrin_yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-6-5上海20006博园路与于田南路交叉口（E）/input/cam'
yaml_path = '/adga/lushiyong/calibration_verify/BevCalib/2025-6-5上海20006博园路与于田南路交叉口（E）/output'

# 路口名
road_info = '_sh6'

# DAN 图像数据子目录
img_dirs = [
    'camera_0_0',
    'camera_1_0',
    'camera_3_0',
    'camera_0_8',
    'camera_1_8',
    'camera_3_8',
]

# 相机对应标定文件名称
calib_dict = {
        'camera_0_0': 's0_0_30',
        'camera_1_0': 's1_0_60',
        'camera_2_0': None,
        'camera_3_0': 's3_0_120',
        'camera_0_8': 's0_8_38',
        'camera_1_8': 's1_8_68',
        'camera_2_8': None,
        'camera_3_8': 's3_8_128',
        }