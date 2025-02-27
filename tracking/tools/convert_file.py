import pickle
import json
import numpy as np

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

def pkl_to_json(pkl_file_path, json_file_path):
    # 加载 .pkl 文件
    with open(pkl_file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        print(f"data:{data.keys()}")
    
    # 将数据中的 ndarray 转换为列表
    data = ndarray_to_list(data)
    
    # 将数据转换为 JSON 格式
    try:
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False)
        print(f"Successfully converted {pkl_file_path} to {json_file_path}")
    except (TypeError, ValueError) as e:
        print(f"Error converting to JSON: {e}")



def json_to_pkl(json_file_path, pkl_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    with open(pkl_file_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
    print(f"Successfully converted {json_file_path} to {pkl_file_path}")

if __name__ == "__main__":
    root_dir = '/home/mogo/data/dev/DetZero/tracking/results/tracking/'
    pkl_file_name = 'tracking-test-20250122-103530.pkl'
    json_file_name = pkl_file_name.replace('pkl','json')
    # json_to_pkl('data.json', 'data.pkl')
    pkl_to_json(root_dir+pkl_file_name, root_dir+json_file_name)