# # DAN 数据目录
data_root = "/rss/4D_label_dataset_vision_only/origin/train_bj_3d_road_7_20250919_lx-debug"

# 目标保存文件夹
dst_root_dir = '/rss/4D_label_dataset_vision_only/Intersection/train_bj_3d_road_7_20250919_lx-debug'

# 伪点云路径
pcd_source_file = '/rss/4D_label_dataset_vision_only/pseudo_pcd/hm_i7_ground.pcd'

# 标定文件目录
yaml_path = '/rss/huben/projects/4D_label/BevCalib/2025-8-25北京环贸纯视觉bev路口/7号路口/output'

# 路口名
road_info = '_hm7'

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