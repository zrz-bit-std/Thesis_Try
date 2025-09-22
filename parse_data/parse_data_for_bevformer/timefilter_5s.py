# 满足现在4D单帧5s间隔的需求
import os
from tqdm import tqdm

def filter_timestamps(timestamps, min_interval=5):
    """
    保留间隔 >= min_interval 的时间戳
    :param timestamps: 递增的时间戳列表
    :param min_interval: 最小时间间隔（秒）
    :return: 过滤后的时间戳列表
    """
    if not timestamps:
        return []

    filtered = [timestamps[0]]  # 保留第一个
    last = timestamps[0]

    for t in tqdm(timestamps[1:]):
        t_ts = int(t[:10])
        last_ts = int(last[:10])
        if t_ts - last_ts >= min_interval:
            filtered.append(t)
            last = t

    return filtered

if __name__ == '__main__':
    # txt_path = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830.txt'
    ref_file_dir = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/images/camera_1_0'
    output_txt = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/wh_i4_0830_5s.txt'
    
    # with open(txt_path, 'r') as f:
    #     file_names = [line.strip() for line in f]

    file_names = sorted(os.path.splitext(i)[0] for i in os.listdir(ref_file_dir))

    filtered = filter_timestamps(file_names)
    with open(output_txt, 'w') as f:
        for name in filtered:
            f.write(name + '\n')
