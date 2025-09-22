#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math

def convert_radians_to_degrees_in_file(file_path):
    """
    将文件中每一行的倒数第二列(假设是yaw角，单位为弧度)转换为角度
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    converted_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            converted_lines.append(line)
            continue
            
        parts = stripped_line.split()
        if len(parts) < 2:
            converted_lines.append(line)
            continue
            
        # 假设倒数第二列是yaw角(弧度)
        try:
            yaw_radians = float(parts[-2])
            yaw_degrees = yaw_radians * 180.0 / math.pi
            parts[-2] = f"{yaw_degrees:.6f}"  # 保留6位小数
            converted_lines.append(" ".join(parts) + "\n")
        except ValueError:
            # 如果转换失败，保持原样
            converted_lines.append(line)
    
    # 写回文件
    with open(file_path, 'w') as f:
        f.writelines(converted_lines)

def main():
    folder_path = "/rss/lishuaiyin/4D_label/dataset_track/train_sh_3d_road_2_20250531_5000_lx/model_pred"
    
    if not os.path.exists(folder_path):
        print(f"目录 {folder_path} 不存在")
        return
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    print(f"找到 {len(files)} 个txt文件")
    
    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)
        convert_radians_to_degrees_in_file(file_path)
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1} 个文件")
    
    print("所有文件处理完成")

if __name__ == "__main__":
    main()