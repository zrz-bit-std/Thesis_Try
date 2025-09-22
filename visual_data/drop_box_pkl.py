import os
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import json
import shutil
from class_mapping import class_mapping_7, class_name2refine_name, classname2id_7, classname2id_refine

def rotate_corners(w, l, heading):
    """
    返回 BEV 框的四个角点，相对于中心 (0,0)
    """
    x_corners = np.array([ w/2,  w/2, -w/2, -w/2])
    y_corners = np.array([ l/2, -l/2, -l/2,  l/2])
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s],[s, c]])
    corners = np.stack([x_corners, y_corners], axis=0)  # 2 x 4
    rotated = R @ corners
    return rotated.T  # 4 x 2

def polygon_area(corners):
    """
    计算多边形面积，corners: N x 2
    """
    x = corners[:,0]
    y = corners[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1)))

def intersect_poly(p1, p2):
    """
    计算两个矩形多边形的交集面积 (简单矩形裁剪方法)
    """
    # 分别取 min-max xy
    x1_min, x1_max = p1[:,0].min(), p1[:,0].max()
    y1_min, y1_max = p1[:,1].min(), p1[:,1].max()
    x2_min, x2_max = p2[:,0].min(), p2[:,0].max()
    y2_min, y2_max = p2[:,1].min(), p2[:,1].max()
    
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    return x_overlap * y_overlap

def bev_iou(box1, box2):
    """
    计算BEV视角下的IOU
    box: [x, y, w, l, heading]
    """
    corners1 = rotate_corners(box1[2], box1[3], box1[4]) + np.array([box1[0], box1[1]])
    corners2 = rotate_corners(box2[2], box2[3], box2[4]) + np.array([box2[0], box2[1]])
    
    inter_area = intersect_poly(corners1, corners2)
    area1 = box1[2]*box1[3]
    area2 = box2[2]*box2[3]
    union_area = area1 + area2 - inter_area + 1e-6
    return inter_area / union_area

def read_pred_txt(file_path):
    """
    读取预测文件
    返回 list: [x,y,z,w,l,h,yaw,confidence,class_id]
    """
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 9:
                continue
                
            # 检查标签是否在映射中
            if vals[1] not in class_name2refine_name:
                print(f"Warning: Unknown label '{vals[1]}' in file {file_path}")
                continue
                
            class_name = class_name2refine_name[vals[1]]  # 获取映射后的类别名，如'person'->'Pedestrian'
            vals = list(map(float, vals[2:10]))
            h, w, l, x, y, z, yaw, confidence = vals
            yaw = np.radians(yaw)
            # 构造bbox: [x, y, z, w, l, h, yaw, confidence, class_name]
            bbox = [x, y, z, w, l, h, yaw, confidence, class_name,vals[0]]
            boxes.append(bbox)  
    return boxes

def convert_box_format(box):
    """
    将box格式从 [label, h, w, l, x, y, z, yaw, score] 转换为 [x, y, w, l, yaw] 用于IOU计算
    """
    label, h, w, l, x, y, z, yaw, score = box
    # 返回 [x, y, w, l, yaw]
    return [x, y, w, l, yaw]

def read_detection_txt(file_path):
    """
    通用函数，用于读取检测文件（包括track和drop文件）
    返回 list: [x,y,z,w,l,h,yaw,score,class_id]
    文件格式: label h w l x y z math.degrees(yaw) score
    """
    boxes = []
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                vals = line.strip().split()
                if len(vals) < 9:
                    print(f"Warning: Line {line_num} has insufficient values: {vals}")
                    continue
                    
                try:
                    # 解析字段: label h w l x y z math.degrees(yaw) score
                    label = vals[1]
                    h, w, l, x, y, z, yaw_deg, score = map(float, vals[2:10])
                    
                    # 转换角度从度到弧度
                    yaw = np.radians(yaw_deg)
                    
                    # 处理标签 - 可能是字符串也可能是数字
                    final_class_name = None
                    if label in class_name2refine_name:
                        # 标签是字符串，如 'car', 'pedestrian' 等
                        final_class_name = class_name2refine_name[label]
                    elif label.isdigit():
                        # 标签是数字，需要映射回类别名
                        label_int = int(label)
                        # 通过classname2id_refine反向查找
                        for k, v in classname2id_refine.items():
                            if v == label_int:
                                final_class_name = k
                                break
                        # 如果没找到，直接使用数字
                        if final_class_name is None:
                            final_class_name = label_int
                    else:
                        print(f"Warning: Unknown label '{label}' in line {line_num}")
                        continue
                    
                    # 构造bbox: [x, y, z, w, l, h, yaw, score, class_name]
                    bbox = [x, y, z, w, l, h, yaw, score, final_class_name]
                    boxes.append(bbox)
                except Exception as e:
                    print(f"Warning: Error parsing line {line_num} in {file_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        
    return boxes

def boxes_equal(box1, box2, threshold=0.01):
    """
    检查两个框是否相等（位置、尺寸、角度差异在阈值内）
    """
    # 检查中心点、宽、长、高、角度
    for i in [0, 1, 2, 3, 4, 5, 6]:  # x, y, z, w, l, h, yaw
        if abs(box1[i] - box2[i]) > threshold:
            return False
    # 检查类别
    if box1[8] != box2[8]:  # class_id
        return False
    return True

def box_distance(box1, box2):
    """
    计算两个框中心点之间的欧氏距离
    box: [x, y, z, w, l, h, yaw, confidence, class_name]
    """
    # 计算中心点距离
    dx = box1[0] - box2[0]
    dy = box1[1] - box2[1]
    dz = box1[2] - box2[2]
    distance = np.sqrt(dx*dx + dy*dy + dz*dz)
    return distance

def find_unmatched_detections(pred_boxes, track_boxes, distance_threshold=1.0):
    """
    找到在pred中存在但在track中没有匹配的检测框
    使用中心点距离来判断是否匹配
    """
    unmatched = []
    
    for pred_box in pred_boxes:
        matched = False
        for track_box in track_boxes:
            # 首先检查类别是否相同
            if pred_box[8] != track_box[8]:  # class_name
                continue
                
            # 计算中心点距离
            distance = box_distance(pred_box, track_box)
            # print(f"Distance between pred {pred_box} and track {track_box}: {distance}")
            
            # 如果距离小于阈值，则认为是匹配的
            if distance < distance_threshold:
                matched = True
                break
                
        # 只有当预测框没有匹配的跟踪框时，才将其添加到未匹配列表中
        if not matched:
            unmatched.append(pred_box)
            
    return unmatched

def clear_droped_files(drop_root):
    """
    清除droped文件夹下的所有txt文件
    """
    if not drop_root.exists():
        return
        
    # 遍历drop_root目录及其子目录，查找所有droped文件夹
    for droped_dir in drop_root.rglob("droped"):
        if droped_dir.is_dir():
            # 删除droped目录下的所有txt文件
            for txt_file in droped_dir.glob("*.txt"):
                txt_file.unlink()
            print(f"Cleared txt files in {droped_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='预测文件目录')
    parser.add_argument('--track', type=str, required=True, help='跟踪结果目录')
    parser.add_argument('--droped', type=str, required=True, help='输出的droped目录')
    parser.add_argument('--clear_droped', action='store_true', help='在生成新文件前清空droped目录下的txt文件')
    args = parser.parse_args()
    
    pred_root = Path(args.pred)
    track_root = Path(args.track)
    drop_root = Path(args.droped)
    
    # 如果指定--clear_droped参数，则清空droped目录下的txt文件
    if args.clear_droped:
        clear_droped_files(drop_root)
    
    # drop_root.mkdir(parents=True, exist_ok=True)
    
    # 收集所有预测文件，包括子目录中的文件，并保持目录结构信息
    pred_files = []
    if pred_root.is_dir():
        # 遍历pred_root及其所有子目录查找txt文件
        for txt_file in pred_root.rglob("*.txt"):
            pred_files.append(txt_file)
    
    print(f"Found {len(pred_files)} prediction files")
    
    # 收集所有跟踪文件
    track_files = {}
    if track_root.is_dir():
        # 遍历track_root及其所有子目录查找txt文件
        for txt_file in track_root.rglob("*.txt"):
            track_files[txt_file.stem] = txt_file
    
    print(f"Found {len(track_files)} tracking files")
    
    # 处理每个预测文件
    for pred_file in tqdm(pred_files, desc="Processing files"):
        stem = pred_file.stem
        
        # 查找对应的track文件
        track_file = track_files.get(stem)
        
        # 如果没有找到track文件，跳过
        if track_file is None:
            print(f"Warning: track file not found for {stem}")
            continue
            
        # 确定输出文件路径，保持与pred文件相同的目录结构
        # 获取pred文件相对于pred_root的路径
        relative_path = pred_file.relative_to(pred_root)
        
        # 在路径的第一个目录名后插入"droped"
        path_parts = list(relative_path.parts)
        if len(path_parts) > 0:
            path_parts.insert(1, "droped")
        else:
            path_parts = ["droped"]
            
        drop_relative_path = Path(*path_parts)
        
        drop_file = drop_root / drop_relative_path
        
        # 确保输出目录存在
        drop_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果预测文件不存在，跳过
        if not pred_file.exists():
            print(f"Warning: Prediction file {pred_file} does not exist")
            continue
            
        # 如果跟踪文件不存在，跳过
        if not track_file.exists():
            print(f"Warning: Track file {track_file} does not exist")
            continue
        
        # 读取预测框和跟踪框
        pred_boxes = read_pred_txt(pred_file)
        track_boxes = read_detection_txt(track_file)
        
        # print(f"File {stem}: pred_boxes={len(pred_boxes)}, track_boxes={len(track_boxes)}")
        
        # 找到未匹配的检测框
        unmatched_boxes = find_unmatched_detections(pred_boxes, track_boxes)
        
        # print(f"File {stem}: unmatched_boxes={len(unmatched_boxes)}")
        
        # 写入drop文件
        if len(unmatched_boxes) > 0:
            with open(drop_file, 'w') as f:
                for box in unmatched_boxes:
                    # 转换回原始格式: label h w l x y z yaw_deg score
                    # 查找原始标签
                    class_name = box[8]
                    
                    # 尝试找到原始标签
                    original_label = None
                    # 通过class_name2refine_name反向查找
                    for k, v in class_name2refine_name.items():
                        if v == class_name:
                            original_label = k
                            break
                    
                    # 如果无法找到原始标签，使用类别名作为标签
                    if original_label is None:
                        original_label = str(class_name)
                    
                    yaw_deg = np.degrees(box[6])
                    f.write(f"{box[9]} {original_label} {box[5]} {box[3]} {box[4]} {box[0]} {box[1]} {box[2]} {yaw_deg} {box[7]}\n")
            # print(f"File {stem}: pred_boxes={len(pred_boxes)}, track_boxes={len(track_boxes)}, unmatched_boxes={len(unmatched_boxes)}")
        elif drop_file.exists():
            # 如果没有未匹配的框，但drop文件已存在，则删除它
            drop_file.unlink()
if __name__ == "__main__":
    main()

