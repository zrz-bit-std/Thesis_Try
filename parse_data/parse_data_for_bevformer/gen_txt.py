import os
from natsort import natsorted
from tqdm import tqdm

data_cnt = {}

road_info_register = {
    '_10': '10',
    '_20': '20',
    '7th': '7',
    '_sh2': 'sh2',
    '_sh3': 'sh3',
    '_sh4': 'sh4',
    '_sh5': 'sh5',
    '_sh6': 'sh6',
    '_hy20': 'hy20',
    '_hm6': 'hm6',
}


def generate_txt_from_filenames(folder_path, output_txt):
    """
    根据文件夹中文件名生成txt文件，去掉文件后缀。
    
    Args:
        folder_path (str): 文件夹路径。
        output_txt (str): 输出txt文件路径。
    """
    try:
        # 获取文件夹中的所有文件
        filenames = os.listdir(folder_path)
        
        # 去掉文件名后缀
        names_without_extension = [os.path.splitext(filename)[0] for filename in filenames]
        names_without_extension = natsorted(names_without_extension)

        for filename in names_without_extension:
            for road in road_info_register.keys():
                if road in filename:
                    if road_info_register[road] not in data_cnt.keys():
                        data_cnt[road_info_register[road]] = 1
                    else:
                        data_cnt[road_info_register[road]]+=1

        for key, val in data_cnt.items():
            print(f'{key}: {val}')
        
        print(f'total: {len(names_without_extension)}')
        
        # 写入到txt文件
        with open(output_txt, 'w') as f:
            for name in names_without_extension:
                f.write(name + '\n')
        
        print(f"已成功生成 {output_txt}")
    except Exception as e:
        print(f"发生错误: {e}")

def generate_txt_from_common_filenames(root_dir, output_txt):
    # 你的5个目录
    dirs = [
        'images/camera_0_0', 
        'images/camera_1_0', 
        'images/camera_2_0', 
        'images/camera_3_0', 
        'images/camera_0_8', 
        'images/camera_1_8', 
        'images/camera_2_8', 
        'images/camera_3_8', 
        # 'label'
        ]

    # 获取每个目录下的文件名集合
    file_sets = [
        set(os.path.splitext(f)[0] for f in tqdm(os.listdir(os.path.join(root_dir, d))) if os.path.isfile(os.path.join(root_dir, d, f)))
        for d in dirs
    ]

    # 取交集
    common_files = set.intersection(*file_sets)

    # 写入 txt
    with open(output_txt, 'w') as f:
        for filename in sorted(common_files):
            f.write(filename + '\n')

    print(f"找到 {len(common_files)} 个共同文件，已写入 {output_txt}")

# 示例使用
print('请注意文件名中是否已添加了标明路口的信息')
# folder_path = '/adga/lushiyong/vis_anno/dataset/hm_i6_0830/images/camera_0_0'
# folder_path = '/adga/lushiyong/vis_anno/dataset/hm_i6_0830/'
# output_txt = '/adga/lushiyong/vis_anno/dataset/hm_i6_0830/hm_i6_0830.txt'
# folder_path = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/images/camera_0_0'
# output_txt = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830.txt'
folder_path = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830_1876_sample/result_json'
output_txt = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830_1876_sample.txt'


generate_txt_from_filenames(folder_path, output_txt)
# generate_txt_from_common_filenames(folder_path, output_txt)