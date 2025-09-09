#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
from collections import defaultdict

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

def main():
    parser = argparse.ArgumentParser(description='Merge drop results with tracking results by timestamp')
    parser.add_argument('--droped', type=str, required=True, help='Drop结果目录')
    parser.add_argument('--track', type=str, required=True, help='跟踪结果目录')
    parser.add_argument('--merged', type=str, required=True, help='合并结果输出目录')
    
    args = parser.parse_args()
    
    merge_drop_and_track(args.droped, args.track, args.merged)

if __name__ == "__main__":
    main()