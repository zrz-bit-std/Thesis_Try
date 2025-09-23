#!/usr/bin/env python3
import pickle
import numpy as np
import os
import argparse

def process_pkl_files(pkl_dir, output_dir):
    # 定义目录路径（已通过参数传入）
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有pkl文件
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
    
    print(f"找到 {len(pkl_files)} 个pkl文件")
    
    # 处理每个pkl文件
    for i, pkl_file in enumerate(pkl_files):
        pkl_path = os.path.join(pkl_dir, pkl_file)
        
        # 加载pkl文件
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 检查数据维度
        if data.ndim != 2:
            print(f"警告: 文件 {pkl_file} 数据维度不是2D，跳过")
            continue
            
        rows, cols = data.shape
        print(f"文件 {pkl_file}: {rows} 行, {cols} 列")
        
        # 提取前9列数据
        first_9_cols = data[:, :9] if cols >= 9 else data
        
        # 创建第一列数据，从0000开始递增
        first_col = np.array([f"{j:04d}" for j in range(rows)]).reshape(-1, 1)
        
        # 将第一列与前9列数据合并
        combined_data = np.hstack([first_col, first_9_cols])
        
        # 生成输出文件名（保持与输入文件相同的名称，但扩展名改为.txt）
        txt_file = pkl_file.replace('.pkl', '.txt')
        txt_path = os.path.join(output_dir, txt_file)
        
        # 保存为txt文件
        np.savetxt(txt_path, combined_data, delimiter=' ', fmt='%s')
        
        if i % 500 == 0:
            print(f"已处理 {i+1}/{len(pkl_files)} 个文件")
    
    print("所有文件处理完成")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将pkl文件转换为txt文件')
    parser.add_argument('--pkl_dir', type=str, 
                        default="/rss/lishuaiyin/4D_label/dataset_track/train_sh_3d_road_2_20250531_5000_lx/model_pred_pkl",
                        help='pkl文件目录')
    parser.add_argument('--output_dir', type=str,
                        default="/rss/lishuaiyin/4D_label/dataset_track/train_sh_3d_road_2_20250531_5000_lx/model_pred",
                        help='输出txt文件目录')
    
    args = parser.parse_args()
    
    process_pkl_files(args.pkl_dir, args.output_dir)