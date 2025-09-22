#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
from collections import defaultdict
import json
import numpy as np
from datetime import datetime

def calculate_distance(box1, box2):
    """
    计算两个边界框中心点之间的欧氏距离
    box: [x, y, z, l, w, h, yaw, vx, vy]
    """
    dx = box1[0] - box2[0]
    dy = box1[1] - box2[1]
    dz = box1[2] - box2[2]
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def assign_object_ids(boxes_data, distance_threshold=2.0):
    """
    根据距离阈值为对象分配ID
    如果前后帧中对象的距离小于阈值，则认为是同一个对象
    """
    if not boxes_data:
        return [], 0
    
    # 按帧分组数据
    frames_data = defaultdict(list)
    for data in boxes_data:
        frame_idx = int(data['frame_idx'])
        frames_data[frame_idx].append(data)
    
    # 按帧排序
    sorted_frames = sorted(frames_data.keys())
    
    # 存储每个检测对应的对象ID
    object_ids = [None] * len(boxes_data)  # 初始化为None
    
    # 对象ID计数器
    next_object_id = 0
    assigned_ids = {}  # frame_idx -> [ids] 映射
    
    # 处理第一帧
    if sorted_frames:
        frame_idx = sorted_frames[0]
        start_idx = 0
        # 找到第一帧的数据在boxes_data中的起始位置
        for i, data in enumerate(boxes_data):
            if int(data['frame_idx']) == frame_idx:
                object_ids[i] = next_object_id
                next_object_id += 1
            elif int(data['frame_idx']) > frame_idx:
                break
    
    # 处理后续帧
    for i in range(1, len(sorted_frames)):
        prev_frame_idx = sorted_frames[i-1]
        curr_frame_idx = sorted_frames[i]
        
        # 找到当前帧和前一帧的数据索引范围
        prev_indices = []
        curr_indices = []
        
        for idx, data in enumerate(boxes_data):
            if int(data['frame_idx']) == prev_frame_idx:
                prev_indices.append(idx)
            elif int(data['frame_idx']) == curr_frame_idx:
                curr_indices.append(idx)
        
        # 为当前帧的对象分配ID
        for curr_idx_in_list, curr_data_idx in enumerate(curr_indices):
            curr_box = boxes_data[curr_data_idx]['box']
            assigned_id = None
            
            # 查找前一帧中最近的对象
            min_distance = float('inf')
            best_match_prev_idx = -1
            
            for prev_idx_in_list, prev_data_idx in enumerate(prev_indices):
                prev_box = boxes_data[prev_data_idx]['box']
                distance = calculate_distance(curr_box, prev_box)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match_prev_idx = prev_idx_in_list
            
            # 如果找到了距离足够近的对象，则认为是同一个对象
            if min_distance < distance_threshold and best_match_prev_idx != -1:
                # 使用前一帧中匹配对象的ID
                prev_data_idx = prev_indices[best_match_prev_idx]
                assigned_id = object_ids[prev_data_idx]
            else:
                # 分配新的ID
                assigned_id = next_object_id
                next_object_id += 1
            
            object_ids[curr_data_idx] = assigned_id
    
    # 检查是否有未分配的ID并分配
    for i in range(len(object_ids)):
        if object_ids[i] is None:
            object_ids[i] = next_object_id
            next_object_id += 1
    
    return object_ids, next_object_id

def merge_drop_and_track(droped_dir, track_dir, merged_dir):
    """
    将drop结果和跟踪结果按时间戳合并生成最终结果
    
    Args:
        droped_dir: drop结果目录
        track_dir: 跟踪结果目录
        merged_dir: 合并结果输出目录
    """
    droped_path = Path(droped_dir)
    track_path = Path(track_dir)
    merged_path = Path(merged_dir)
    
    # 创建输出目录
    merged_path.mkdir(parents=True, exist_ok=True)
    
    # 收集所有跟踪文件（包括子目录结构）
    track_files = defaultdict(list)
    if track_path.is_dir():
        for txt_file in track_path.rglob("*.txt"):
            # 获取时间戳（文件名，不含扩展名）
            timestamp = txt_file.stem
            track_files[timestamp].append(txt_file)
    
    print(f"Found {len(track_files)} unique timestamps in tracking files")
    
    # 收集所有drop文件（包括子目录结构）
    drop_files = defaultdict(list)
    if droped_path.is_dir():
        for txt_file in droped_path.rglob("*.txt"):
            # 获取时间戳（文件名，不含扩展名）
            timestamp = txt_file.stem
            drop_files[timestamp].append(txt_file)
    
    print(f"Found {len(drop_files)} unique timestamps in drop files")
    
    # 获取所有时间戳
    all_timestamps = set(track_files.keys()) | set(drop_files.keys())
    print(f"Total unique timestamps: {len(all_timestamps)}")
    
    # 处理每个时间戳
    for timestamp in tqdm(all_timestamps, desc="Processing timestamps"):
        # 创建对应时间戳的合并目录
        # 我们需要确定文件应该放在哪个子目录中
        merged_timestamp_dir = merged_path
        
        # 查找跟踪文件
        track_file_list = track_files.get(timestamp, [])
        # 查找drop文件
        drop_file_list = drop_files.get(timestamp, [])
        
        # 确定输出文件路径
        merged_file = None
        
        # 如果存在跟踪文件，使用第一个跟踪文件的相对路径结构
        if track_file_list:
            track_file = track_file_list[0]  # 使用第一个文件作为参考
            relative_path = track_file.relative_to(track_path)
            merged_file = merged_path / relative_path
        # 如果只有drop文件，使用drop文件的相对路径结构
        elif drop_file_list:
            drop_file = drop_file_list[0]  # 使用第一个文件作为参考
            relative_path = drop_file.relative_to(droped_path)
            merged_file = merged_path / relative_path
        
        # 确保输出目录存在
        if merged_file:
            merged_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 合并文件内容
            with open(merged_file, 'w') as mf:
                # 先写入跟踪文件内容
                for track_file in track_file_list:
                    if track_file.exists():
                        with open(track_file, 'r') as tf:
                            content = tf.read()
                            if content.strip():
                                mf.write(content)
                                # 如果文件不以换行符结尾，添加一个
                                if not content.endswith('\n'):
                                    mf.write('\n')
                
                # 再追加drop文件内容
                for drop_file in drop_file_list:
                    if drop_file.exists():
                        with open(drop_file, 'r') as df:
                            content = df.read()
                            if content.strip():
                                # 如果合并文件已经有内容且不以换行符结尾，先添加换行符
                                # if mf.tell() > 0 and not content.startswith('\n'):
                                #     mf.write('\n')
                                mf.write(content)
    
    print(f"Merge completed. Results saved to {merged_dir}")

def generate_tracking_pkl(merged_dir):
    """
    根据合并后的txt结果文件生成pkl文件
    
    Args:
        merged_dir: 合并结果目录
    """
    merged_path = Path(merged_dir)
    
    # 收集所有合并后的txt文件，按目录分组
    txt_files_by_dir = defaultdict(list)
    for txt_file in merged_path.rglob("*.txt"):
        # 获取父目录作为分组键
        parent_dir = txt_file.parent
        txt_files_by_dir[parent_dir].append(txt_file)
    
    print(f"Found {len(txt_files_by_dir)} directories with txt files for pkl generation")
    
    # 为每个目录生成一个pkl文件
    for dir_path, txt_files in txt_files_by_dir.items():
        print(f"Processing directory: {dir_path}")
        print(f"Found {len(txt_files)} txt files in this directory")
        
        # 按文件名排序
        txt_files.sort(key=lambda x: x.stem)
        
        # 构建数据结构
        tracking_data = {}
        
        # 从目录结构中提取数据集名称和时间戳
        try:
            # 假设目录结构类似于 .../dataset_name/timestamp/splited/
            # 我们需要创建 .../dataset_name/timestamp/tracking/
            parts = dir_path.parts
            dataset_name = parts[-3]  # 假设数据集名称在倒数第三层
            timestamp = parts[-2]     # 假设时间戳在倒数第二层
            
            # 构建输出目录路径，将splited替换为tracking
            relative_dir_path = dir_path.relative_to(merged_path)
            output_dir_path = merged_path / relative_dir_path
            if output_dir_path.name == "splited":
                # 将最后的"splited"替换为"tracking"
                output_dir_path = output_dir_path.parent / "tracking"
            
            output_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error processing directory path: {e}")
            # 使用默认路径
            output_dir_path = output_path
            output_dir_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            dataset_name = "unknown"
        
        # 收集所有帧的数据
        all_boxes_data = []
        
        # 处理每个txt文件
        for frame_idx, txt_file in enumerate(tqdm(txt_files, desc=f"Processing txt files in {dir_path.name}", leave=False)):
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                values = line.strip().split()
                if len(values) < 9:
                    continue
                    
                label = values[0]
                h, w, l, x, y, z, yaw, confidence = map(float, values[1:9])
                
                # boxes_global: [x, y, z, l, w, h, yaw, vx, vy]
                # 简化处理，将vx, vy设为0
                box = [x, y, z, l, w, h, yaw, 0.0, 0.0]
                
                # 保存数据用于后续处理
                box_data = {
                    'frame_idx': frame_idx,
                    'box': box,
                    'label': label,
                    'confidence': confidence
                }
                all_boxes_data.append(box_data)
        
        if not all_boxes_data:  # 如果没有数据，跳过
            print(f"No data found in directory {dir_path}, skipping...")
            continue
            
        # 根据距离为对象分配ID
        object_ids, total_objects = assign_object_ids(all_boxes_data)
        
        # 构建按对象ID分组的数据结构
        object_data = defaultdict(list)
        for i, data in enumerate(all_boxes_data):
            obj_id = object_ids[i]
            object_data[obj_id].append(data)
        
        # 构建最终的数据结构
        tracking_data = {}
        tracking_key = f"{dataset_name}_{timestamp}"
        
        # 为每个对象构建数据
        for obj_id, frames_data in object_data.items():
            boxes_global = []
            names = []
            scores = []
            sample_idxs = []
            hits = []
            num_points = []
            obj_ids = []
            poses = []
            states = "dynamic"  # 这里改为列表形式
            
            # 按帧索引排序
            frames_data.sort(key=lambda x: x['frame_idx'])
            
            for data in frames_data:
                boxes_global.append(data['box'])
                names.append(data['label'])
                scores.append(data['confidence'])
                sample_idxs.append(str(data['frame_idx']))
                hits.append(1)
                num_points.append(0.0)
                obj_ids.append(obj_id)
                
                # pose: 按照当前形式填写
                pose = [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
                poses.append(pose)
                
                # state: 通过速度判断状态
                # 简化处理，根据速度大小判断状态
                vx, vy = data['box'][7], data['box'][8]
                speed = np.sqrt(vx**2 + vy**2)
                if speed < 0.1:
                    states.append("static")
                else:
                    states.append("dynamic")
            
            # 添加到跟踪数据中，以对象ID为键
            tracking_data[str(obj_id)] = {
                "boxes_global": boxes_global,
                "name": names,
                "score": scores,
                "sample_idx": sample_idxs,
                "hit": hits,
                "num_points": num_points,
                "obj_ids": obj_ids,
                "pose": poses,
                "state": states  # 保持列表形式
            }
        
        # 包装在数据集键下
        final_data = {tracking_key: tracking_data}
        
        # 保存为JSON文件（模拟pkl）
        timestamp_str = datetime.now().strftime("tracking-test-%Y%m%d-%H%M%S")
        output_file = output_dir_path / f"{timestamp_str}.json"
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=4)
        
        print(f"Tracking pkl file generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Merge drop results with tracking results by timestamp')
    parser.add_argument('--droped', type=str, required=True, help='Drop结果目录')
    parser.add_argument('--track', type=str, required=True, help='跟踪结果目录')
    parser.add_argument('--merged', type=str, required=True, help='合并结果输出目录')
    
    args = parser.parse_args()
    
    # 合并drop和track结果
    merge_drop_and_track(args.droped, args.track, args.merged)
    
    # 生成pkl文件
    # generate_tracking_pkl(args.merged)

if __name__ == "__main__":
    main()