import os
import shutil
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='重命名pkl文件')
parser.add_argument('--src_dir', type=str, 
                    default="/rss/lishuaiyin/4D_label/dataset_track/train_sh_3d_road_2_20250531_5000_lx/model_pred_pkl",
                    help='源pkl文件目录')
parser.add_argument('--origin_data_dir', type=str,
                    default="/rss/lishuaiyin/4D_label/origin/train_sh_3d_road_2_20250531_5000_lx/original_data",
                    help='原始数据目录')
parser.add_argument('--output_dir', type=str,
                    default="/rss/lishuaiyin/4D_label/dataset_track/train_sh_3d_road_2_20250531_5000_lx/model_pred_pkl",
                    help='输出目录')
parser.add_argument('--dry-run', action='store_true',
                    help='仅模拟运行，不实际移动文件')

args = parser.parse_args()

# 目录路径
src_dir = args.src_dir
origin_data_dir = args.origin_data_dir
output_dir = args.output_dir

# 检查目录是否存在
if not os.path.exists(src_dir):
    raise FileNotFoundError(f"源pkl目录不存在: {src_dir}")
if not os.path.exists(origin_data_dir):
    raise FileNotFoundError(f"原始数据目录不存在: {origin_data_dir}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取源目录中的所有pkl文件名
pkl_files = [f for f in os.listdir(src_dir) if f.endswith('.pkl')]
print(f"找到 {len(pkl_files)} 个pkl文件")

# 创建一个字典，将时间戳映射到对应的jpg文件名（转换为pkl）
timestamp_to_jpg = {}

# 遍历所有子目录
for subdir in os.listdir(origin_data_dir):
    subdir_path = os.path.join(origin_data_dir, subdir)
    if os.path.isdir(subdir_path):
        # 查找以 'camera' 开头的子目录
        camera_dirs = [d for d in os.listdir(subdir_path) if d.startswith('camera') and os.path.isdir(os.path.join(subdir_path, d))]
        if camera_dirs:
            # 按字母顺序排序并取第一个camera目录
            first_camera_dir = sorted(camera_dirs)[0]
            camera_path = os.path.join(subdir_path, first_camera_dir)
            
            # 获取该目录下的所有jpg文件
            jpg_files = [f for f in os.listdir(camera_path) if f.endswith('.jpg')]
            for jpg_file in jpg_files:
                # 从文件名中提取时间戳部分，例如从 "1748647643.100004.jpg" 提取 "1748647643100"
                if '.' in jpg_file:
                    parts = jpg_file.split('.')
                    timestamp_str = parts[0] + parts[1][:3]  # 秒 + 毫秒前3位
                    # 将jpg文件名映射为pkl文件名
                    timestamp_to_jpg[timestamp_str] = jpg_file.replace('.jpg', '.pkl')

print(f"映射了 {len(timestamp_to_jpg)} 个时间戳到jpg文件名")

# 重命名pkl文件
count = 0
for pkl_file in pkl_files:
    # 从pkl文件名提取时间戳，如 1748647643100_sh2.pkl -> 1748647643100
    base_name = pkl_file.split('.')[0]  # 先按.分割
    if '_' in base_name:
        timestamp = base_name.split('_')[0]  # 再按_分割取时间戳部分
    else:
        timestamp = base_name  # 如果没有下划线，整个就是时间戳

    # 查找对应的jpg文件名
    if timestamp in timestamp_to_jpg:
        new_name = timestamp_to_jpg[timestamp]
        old_path = os.path.join(src_dir, pkl_file)
        new_path = os.path.join(output_dir, new_name)
        
        # 执行重命名（移动）
        if args.dry_run:
            print(f"[模拟] 将会重命名: {pkl_file} -> {new_name}")
        else:
            shutil.move(old_path, new_path)
            count += 1