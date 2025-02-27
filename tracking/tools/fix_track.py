import pickle
import json
import numpy as np
import os

def ndarray_to_list(obj):
    """递归地将 ndarray 转换为列表"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 将 ndarray 转换为列表
    elif isinstance(obj, dict):
        return {key: ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ndarray_to_list(item) for item in obj]
    else:
        return obj

def split_pkl(track_path, json_dir):
    with open(track_path, 'rb') as file:
        track_data_pkl = pickle.load(file)
        first_key = list(track_data_pkl.keys())[0]
        print("track_data_pkl.keys():{}".format(first_key))
        track_data = track_data_pkl[first_key]
        print(f"len(track_data):{len(track_data)}")
        for track_id in track_data:
            sample_idx = track_data[track_id]['sample_idx']
            boxes_global = track_data[track_id]['boxes_global']
            boxes_pose = track_data[track_id]['pose']
            json_name = str(track_id)+".json"
            json_file_path = os.path.join(json_dir, json_name)
            data = ndarray_to_list(boxes_global)
            try:
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(data, json_file, ensure_ascii=False)
                print(f"Successfully save to {json_file_path}")
            except (TypeError, ValueError) as e:
                print(f"Error saving JSON: {e}")

def vis_track(ori_dir, fix_dir=None):
    
    if fix_dir!=None:

if __name__ == "__main__":
    root_dir = '/home/mogo/data/dev/DetZero/tracking/results/tracking/'
    pkl_file_path = root_dir+'tracking-test-20250122-103530.pkl'
    json_dir = root_dir+'tracklet_ori'
    # split_pkl(pkl_file_path, json_dir)


