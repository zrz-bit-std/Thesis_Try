import os
import yaml

# 修改后的自定义构造器，返回字典
def opencv_matrix_constructor(loader, node):
    return loader.construct_mapping(node, deep=True)

# 注册自定义标签处理
yaml.add_constructor('tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor, Loader=yaml.SafeLoader)



def read_param_from_txt(file_path):
    result_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(':')
            key = parts[0].strip()
            values = parts[1].strip().split(' ')
            # 将字符串类型的数值转换为对应的数值类型（float），如果不需要转换可以注释掉下一行
            values = [float(val) for val in values]
            result_dict[key] = values
    intri_param = [np.array(result_dict[k]).reshape(3,4)[:,:3] for k in ["P0","P1","P2","P3"]]
    extri_param = [np.array(result_dict[k] + [0,0,0,1]).reshape(4,4) for k in ['Tr_velo_to_cam_0','Tr_velo_to_cam_1','Tr_velo_to_cam_2','Tr_velo_to_cam_3']]
    return intri_param,extri_param

def load_yaml(yaml_file):
    """
    从 YAML 文件中提取 T_common_cam 数据的前 12 个元素。
    """
    with open(yaml_file, 'r') as f:
        content = f.readlines()[2:]  # 跳过前两行 %YAML:1.0 和 ---
        content = ''.join(content)
        data = yaml.safe_load(content)
    return data

def extract_T_common_cam_data(yaml_file):
    """
    从 YAML 文件中提取 T_common_cam 数据的前 12 个元素。
    """
    with open(yaml_file, 'r') as f:
        content = f.readlines()[2:]  # 跳过前两行 %YAML:1.0 和 ---
        content = ''.join(content)
        data = yaml.safe_load(content)
    T_common_cam = data.get('T_common_cam', {})
    matrix_data = T_common_cam.get('data', [])
    if len(matrix_data) < 12:
        raise ValueError(f"文件 {yaml_file} 中的 T_common_cam 数据不足 12 个元素。")
    # 提取前 12 个元素
    return matrix_data[:12]

def get_label_from_filename(filename):
    """
    根据文件名生成标签，例如：
    文件名 'cam_ground_extrinsic_calib_result_0_s0_common.yaml' -> 标签 'Tr_velo_to_cam_0_s0'
    """
    parts = filename.split('_')
    if len(parts) < 7:
        raise ValueError(f"文件名格式不符合预期: {filename}")
    cam_index = parts[4]  # 获取相机索引，例如 '0' 或 '8'
    s_index = parts[5]     # 获取传感器索引，例如 's0'
    label = f"Tr_velo_to_cam_{cam_index}_{s_index}"
    return label

def main():
    import argparse

    parser = argparse.ArgumentParser(description='提取 YAML 文件中的 T_common_cam 数据并保存为 KITTI 格式的 TXT 文件。')
    parser.add_argument('--input_dir', type=str, default='/data1/turbo_data/liangyaolin/road_dataset/8cam_bev_dataset/scripts/batch_scripts_tongxiang_1/calib/result', help='包含 YAML 文件的目录路径。')
    parser.add_argument('--output_file', type=str, default='calib_kitti_tongxiang30001.txt', help='输出 TXT 文件的名称。')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file

    # 列出符合模式的 YAML 文件
    yaml_files = [f for f in os.listdir(input_dir) if f.startswith('cam_ground_extrinsic_calib_result_') and f.endswith('_common.yaml')]

    if not yaml_files:
        print("未找到符合条件的 YAML 文件。")
        return

    with open(output_file, 'w') as out_f:
        for yaml_file in sorted(yaml_files):
            file_path = os.path.join(input_dir, yaml_file)
            try:
                matrix = extract_T_common_cam_data(file_path)
                label = get_label_from_filename(yaml_file)
                # 将数值格式化为字符串，保留足够的小数位
                matrix_str = ' '.join([f"{num:.17g}" for num in matrix])
                out_f.write(f"{label}: {matrix_str}\n")
                print(f"已处理文件: {yaml_file}")
            except Exception as e:
                print(f"处理文件 {yaml_file} 时出错: {e}")

    print(f"校准数据已保存到 {output_file}")

if __name__ == "__main__":
    main()
