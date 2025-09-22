import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 类别映射
class_map = {
    'person': "1000",
    'motor': "4000",
    'car': "5000",
    'truck': "6000",
    'bus': "7000",
}

# 转换后的目标数据结构
def create_output_data(file_wo_ext):
    return {
        "data": {
            "3d_url": f"{file_wo_ext}.pcd"
        },
        "nCloud": 173031,
        "result": {
            "data": []
        }
    }

# 处理每个文件的函数
def process_file(file, txt_dir, save_dir):
    file_wo_ext = os.path.splitext(file)[0]
    txt_path = os.path.join(txt_dir, file)
    output_data = create_output_data(file_wo_ext)

    with open(txt_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            items = line.strip().split()
            if len(items) < 15:  # 忽略不完整的数据行
                continue
            anno_cls = items[0]
            _, _, _, _, _, _, _, h, w, l, x, y, z, theta = map(float, items[1:15])

            result_item = {
                "3Dcenter": {
                    "x": x,
                    "y": y,
                    "z": z
                },
                "3Dsize": {
                    "length": l,
                    "width": w,
                    "height": h,
                    "alpha": theta,
                    "rx": 0,
                    "ry": 0,
                    "rz": theta
                },
                "group": "0",  # 固定值
                "ObjectID": i,  # 转换 obj_id 为整数
                "label": class_map.get(anno_cls, "unknown"), 
                "sublabel": class_map.get(anno_cls, "unknown"),  # 固定值
                "pointnum": 6,  # 固定值
                "Pseudo_3D_larger_than_60cm": 0  # 固定值
            }
            output_data["result"]["data"].append(result_item)

    # 保存文件
    output_json_path = os.path.join(save_dir, f"{file_wo_ext}.json")
    with open(output_json_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

# 多线程操作
def process_files_in_parallel(txt_dir, save_dir, files_list, max_workers=4):
    os.makedirs(save_dir, exist_ok=True)

    # 使用 ThreadPoolExecutor 执行多线程操作
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file in files_list:
            futures.append(executor.submit(process_file, file, txt_dir, save_dir))

        # 等待所有任务完成
        for future in tqdm(as_completed(futures), desc="Processing files"):
            future.result()  # 触发异常如果有的话

# 主执行函数
def main():
    # txt_dir = '/adga/lushiyong/bev_8cam_compare_result/sh_i2_val/train_on_hy_i7_i10_sh_i2_i3_i4_i5_i6_tx_i1_mix_77187_trainset_ep50/conf0.3/3d_url'
    # save_dir = '/adga/lushiyong/vis_anno/dataset/sh_i2/4d_val_set/result_json/'
    txt_dir = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/kitti_result/3d_url_nms'
    save_dir = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830_1876/result_json/'
    files_list = os.listdir(txt_dir)

    process_files_in_parallel(txt_dir, save_dir, files_list)

if __name__ == "__main__":
    main()