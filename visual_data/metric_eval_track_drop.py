import argparse
from pathlib import Path
from tqdm import tqdm
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from class_mapping import class_mapping_7, class_map_refine, classname2id_7
import sys

DEFAULT_JSON_FILENAME = "train_sh_23d_road_6_20250517_5000_lx_4_final.json"

def create_json_map():
    pred_root = Path("/rss/lishuaiyin/4D_label/dataset_track/train_sh_23d_road_6_20250517_5000_lx_4/model_pred")
    track_root = Path("/rss/lishuaiyin/4D_label/dataset_track/offline_tracked/train_sh_23d_road_6_20250517_5000_lx_4/20250517123903/splited_final")
    drop_root  = Path("/rss/lishuaiyin/4D_label/dataset_track/offline_tracked/train_sh_23d_road_6_20250517_5000_lx_4/20250517123903/droped_final")
    result_base = Path("/rss/lishuaiyin/4D_label/results/train_sh_23d_road_6_20250517_5000_lx_4/result/")
    
    print(f"Checking track_root: {track_root}")
    print(f"Exists: {track_root.exists()}, Is directory: {track_root.is_dir()}")
    print(f"Checking drop_root: {drop_root}")
    print(f"Exists: {drop_root.exists()}, Is directory: {drop_root.is_dir()}")
    
    if not track_root.exists():
        print(f"Error: track_root path does not exist!")
        return
        
    if not drop_root.exists():
        print(f"Error: drop_root path does not exist!")
        return
        
    gt_json_dirs = list(result_base.rglob("result_json"))
    print(f"Found {len(gt_json_dirs)} result_json directories")

    # 收集所有 JSON 文件
    all_gt_json_files = []
    for gt_root in gt_json_dirs:
        if gt_root.exists():
            json_files = list(gt_root.glob("*.json"))
            all_gt_json_files.extend(json_files)
            print(f"Found {len(json_files)} json files in {gt_root}")

    # 为快速查找，建立一个字典：stem -> path
    gt_dict = {p.stem: p for p in all_gt_json_files}
    print(f"Total GT json files: {len(gt_dict)}")

    file_map = {}

    # 尝试多种模式查找txt文件
    patterns = ["*/*.txt", "*.txt", "**/*.txt"]
    txt_files = []
    
    for pattern in patterns:
        files = list(track_root.glob(pattern))
        print(f"Pattern '{pattern}' found {len(files)} files")
        if files:
            txt_files.extend(files)
    
    if not txt_files:
        print("No txt files found! Checking directory contents:")
        # 列出目录内容
        try:
            for item in track_root.iterdir():
                print(f"  {item} - {item.is_file() if item.exists() else 'Not exists'}")
        except Exception as e:
            print(f"Error reading directory: {e}")
        return

    print(f"Processing {len(txt_files)} txt files...")
    
    for track_file in tqdm(txt_files, desc="Processing txt files"):
        if not track_file.exists():
            continue
            
        stem = track_file.stem
        pred_file = pred_root / f"{stem}.txt"
        drop_file = drop_root / f"{stem}.txt"  # 对应的drop文件
        
        # 在 gt_dict 中找到对应的 json
        gt_file = gt_dict.get(stem)
        if gt_file is None:
            print(f"Warning: gt file not found for {stem} ({track_file})")
            continue
            
        if not pred_file.exists():
            print(f"Warning: pred file not found: {pred_file}")
            # 可以选择继续或跳过
            
        file_map[stem] = {
            "pred": str(pred_file),
            "gt": str(gt_file),
            "track": str(track_file),
            "drop": str(drop_file)  # 添加drop文件路径
        }
    
    save_path = Path(DEFAULT_JSON_FILENAME)
    
    print(f"Creating mapping for {len(file_map)} files")
    
    with open(save_path, "w") as f:
        json.dump(file_map, f, indent=4)
    
    print(f"file_map 已保存到 {save_path.resolve()}")

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
    box: [x, y, w, l, heading]
    """
    corners1 = rotate_corners(box1[2], box1[3], box1[4]) + np.array([box1[0], box1[1]])
    corners2 = rotate_corners(box2[2], box2[3], box2[4]) + np.array([box2[0], box2[1]])
    
    inter_area = intersect_poly(corners1, corners2)
    area1 = box1[2]*box1[3]
    area2 = box2[2]*box2[3]
    union_area = area1 + area2 - inter_area + 1e-6
    return inter_area / union_area


def read_pred_txt(file_path, score_threshold=1.5):
    """
    返回 list: [x,y,w,l,heading,score]
    """
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            class_name = class_map_refine[classname2id_7[vals[0]]]
            vals = list(map(float, vals[1:9]))
            h, w, l, x, y, z, yaw, confidence = vals
            
            # 添加score阈值过滤
            if confidence < score_threshold:
                continue
                
            bbox = [x, y, z, w, l, h, yaw, confidence, class_name]
            boxes.append(bbox)  
    return boxes

def read_detection_txt(file_path, score_threshold=1.5):
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
                    label = vals[0]
                    h, w, l, x, y, z, yaw_deg, score = map(float, vals[1:9])
                    
                    # 添加score阈值过滤
                    if score < score_threshold:
                        continue
                    
                    # 转换角度从度到弧度
                    yaw = np.radians(yaw_deg)
                    
                    # 将标签名转换为数字ID
                    if label in classname2id_7:
                        class_name = classname2id_7[label]  # 得到的是字符串，如 'Pedestrian'
                        # 应用类别映射得到数字ID
                        if class_name in class_map_refine:
                            refined_class_id = class_map_refine[class_name]
                        else:
                            print(f"Warning: No mapping for class '{class_name}' in line {line_num}")
                            continue
                    else:
                        print(f"Warning: Unknown label '{label}' in line {line_num}")
                        continue
                    
                    # 构造bbox: [x, y, z, w, l, h, yaw, score, class_id]
                    bbox = [x, y, z, w, l, h, yaw, score, refined_class_id]
                    boxes.append(bbox)
                except Exception as e:
                    print(f"Warning: Error parsing line {line_num} in {file_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        
    return boxes

def read_gt_json(file_path):
    """
    返回 list: [x, y, w, l, heading]
    """
    boxes = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for obj in data['result']['data']:
            center = obj['3Dcenter']
            size = obj['3Dsize']
            x = center['x']
            y = center['y']
            z = center['z']
            w = size['width']
            l = size['length']
            h = size['height']
            heading = size['alpha']  # 或者 alpha
            if class_mapping_7[obj['sublabel']] == 'ignore':
                continue
            class_name = class_map_refine[classname2id_7[class_mapping_7[obj['sublabel']]]]
            boxes.append([x, y, z, w, l, h, heading, class_name])
    return boxes


class MAPEvaluator:
    def __init__(self, map_classes, iou_thresholds=None):
        """
        Args:
            map_classes (list[str]): 类别名称列表
            iou_thresholds (list[float]): IoU 阈值集合
        """
        self.map_classes = map_classes
        if iou_thresholds is None:
            # iou_thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
            iou_thresholds = [0.5]
        self.iou_thresholds = iou_thresholds

    def evaluate_map(self, results):
        """
        results: list[dict]，每个 dict 结构:
            {
              "pred": [[x,y,z,w,l,h,yaw,score,class], ...],
              "gt":   [[x,y,z,w,l,h,yaw,class], ...]
            }
        """
        metrics = {}
        ap_per_class = {cls: [] for cls in self.map_classes}
        for thr in self.iou_thresholds:
            ap_results, _, _ = self._evaluate_single_threshold(results, thr)
            for cls, ap in ap_results.items():
                ap_per_class[cls].append(ap)

        # 保存结果
        for cls in self.map_classes:
            aps = ap_per_class[cls]
            if aps:  # 检查列表是否为空
                metrics[f"map/{cls}/iou@max"] = max(aps)
                for thr, ap in zip(self.iou_thresholds, aps):
                    metrics[f"map/{cls}/iou@{thr:.2f}"] = ap
            else:
                metrics[f"map/{cls}/iou@max"] = 0.0
                for thr in self.iou_thresholds:
                    metrics[f"map/{cls}/iou@{thr:.2f}"] = 0.0

        # 平均
        all_aps = [v for v in ap_per_class.values() if v]  # 过滤掉空列表
        if all_aps:
            metrics["map/mean/iou@max"] = np.mean([max(v) for v in all_aps])
            for i, thr in enumerate(self.iou_thresholds):
                metrics[f"map/mean/iou@{thr:.2f}"] = np.mean([ap_per_class[c][i] if ap_per_class[c] else 0.0 for c in self.map_classes])
        else:
            metrics["map/mean/iou@max"] = 0.0
            for thr in self.iou_thresholds:
                metrics[f"map/mean/iou@{thr:.2f}"] = 0.0

        return metrics, ap_per_class

    def _evaluate_single_threshold(self, results, iou_thr):
        """
        在单一 IoU 阈值下计算 AP
        """
        all_pred = []
        all_gt_dict = {}
        gt_counter = 0

        # 整合所有预测和GT
        for idx, item in enumerate(results):
            stem = f"sample_{idx}"
            preds = item["pred"]
            gts = item["gt"]
            all_gt_dict[stem] = gts
            gt_counter += len(gts)
            for p in preds:
                all_pred.append(p + [stem])

        # 按类别计算 AP
        class_pred = {}
        for p in all_pred:
            cls = p[-2]
            class_pred.setdefault(cls, []).append(p)

        ap_results, precision_results, recall_results = {}, {}, {}

        for cls, preds in class_pred.items():
            gt_matched_dict = {}
            gt_boxes_dict = {}
            for stem in all_gt_dict.keys():
                gt_boxes_cls = [g for g in all_gt_dict[stem] if g[-1] == cls]
                gt_boxes_dict[stem] = gt_boxes_cls
                gt_matched_dict[stem] = np.zeros(len(gt_boxes_cls))

            tp_list, fp_list = [], []
            preds.sort(key=lambda x: x[-3], reverse=True)  # score在倒数第三

            for pred in preds:
                stem = pred[-1]
                pred_box = [pred[0], pred[1], pred[3], pred[4], pred[6]]  # x,y,w,l,yaw

                gt_boxes = gt_boxes_dict[stem]
                matched = gt_matched_dict[stem]

                max_iou, best_idx = 0, -1
                for j, gt in enumerate(gt_boxes):
                    gt_box = [gt[0], gt[1], gt[3], gt[4], gt[6]]
                    iou = bev_iou(pred_box, gt_box)
                    if iou > max_iou:
                        max_iou, best_idx = iou, j

                if max_iou >= iou_thr and best_idx >= 0 and matched[best_idx] == 0:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched[best_idx] = 1
                else:
                    tp_list.append(0)
                    fp_list.append(1)

            tp_cum = np.cumsum(tp_list)
            fp_cum = np.cumsum(fp_list)
            total_gt = sum(len(v) for v in gt_boxes_dict.values())
            precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
            recalls = tp_cum / (total_gt + 1e-6)

            recalls_pad = np.concatenate(([0], recalls, [1]))
            precisions_pad = np.concatenate(([0], precisions, [0]))
            for i in range(len(precisions_pad)-2, -1, -1):
                precisions_pad[i] = max(precisions_pad[i], precisions_pad[i+1])
            ap = np.trapz(precisions_pad, recalls_pad)

            ap_results[cls] = ap
            precision_results[cls] = precisions.mean() if len(precisions) else 0
            recall_results[cls] = recalls.mean() if len(recalls) else 0

        return ap_results, precision_results, recall_results

    def plot_map_curves(self, ap_per_class, save_path=None):
        """
        可视化 AP vs IoU 阈值 曲线
        """
        plt.figure(figsize=(8, 6))
        for cls in self.map_classes:
            if ap_per_class[cls]:  # 只有当列表非空时才绘制
                plt.plot(
                    self.iou_thresholds,
                    ap_per_class[cls],
                    marker="o",
                    label=cls
                )

        # 计算平均曲线，只考虑非空的类别
        non_empty_classes = [c for c in self.map_classes if ap_per_class[c]]
        if non_empty_classes:  # 只有当存在非空类别时才绘制平均曲线
            mean_curve = np.mean([ap_per_class[c] for c in non_empty_classes], axis=0)
            plt.plot(
                self.iou_thresholds,
                mean_curve,
                marker="x",
                linestyle="--",
                color="black",
                label="mean"
            )

        plt.xlabel("IoU Threshold")
        plt.ylabel("AP")
        plt.title("AP vs IoU Threshold")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()



# ----------------- 主函数 -----------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate tracking results')
    parser.add_argument('--create-map', action='store_true', 
                       help='Create JSON mapping file')
    parser.add_argument('--track-only', action='store_true',
                       help='Only use track boxes, not include drop boxes')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                       help='Score threshold for filtering detection boxes')
    
    # parser.add_argument('--pred_root', type=str, 
    #                    default="/data1/turbo_data/lishuaiyin/4D_label/dataset_track/train_sh_3d_road_5_20250524_5000_lx/model_pred",
    #                    help='Prediction root directory')
    # parser.add_argument('--track_root', type=str,
    #                    default="/data1/turbo_data/lishuaiyin/4D_label/dataset_track/offline_tracked/train_sh_3d_road_5_20250524_5000_lx/20250524070526/splited",
    #                    help='Tracking result root directory')
    # parser.add_argument('--result_base', type=str,
    #                    default="/data1/turbo_data/RALG/data/3.0_pro/Inter+ Road/label/shanghai_road_5/p1_lx/train_sh_3d_road_5_20250524_5000_lx/result",
    #                    help='Ground truth result base directory')
    
    args = parser.parse_args()

    if args.create_map:
        create_json_map()
        sys.exit()
    with open(DEFAULT_JSON_FILENAME, "r") as f:
        file_map = json.load(f)

    # # track 框 AP
    # ap_results, precision_results, recall_results = calculate_total_ap(file_map, track=True, iou_threshold=0.3)
    # print("\n=== Refine AP per class ===")
    # for cls in ap_results.keys():
    #     print(f"{cls}: AP={ap_results[cls]:.4f}")
    #     print(f"Precision: {precision_results[cls]}")
    #     print(f"Recall   : {recall_results[cls]}\n")

    # # detection 框 AP
    # ap_results, precision_results, recall_results = calculate_total_ap(file_map, track=False, iou_threshold=0.3)
    # print("\n=== Detection AP per class ===")
    # for cls in ap_results.keys():
    #     print(f"{cls}: AP={ap_results[cls]:.4f}")
    #     print(f"Precision: {precision_results[cls]}")
    #     print(f"Recall   : {recall_results[cls]}\n")

    track = True

    # 1. 构造 results 格式
    results = []
    for stem, paths in tqdm(file_map.items(), desc="Loading files"):
        if track:
            # 读取track文件
            track_boxes = read_detection_txt(paths['track'], score_threshold=args.score_threshold) if Path(paths['track']).exists() else []
            # 读取drop文件（如果存在）
            drop_boxes = []
            if not args.track_only and 'drop' in paths and Path(paths['drop']).exists():
                drop_boxes = read_detection_txt(paths['drop'], score_threshold=args.score_threshold)
            # 合并track和drop的结果
            # print("drop boxes: ",drop_boxes)
            pred_boxes = track_boxes + drop_boxes
            
            # 添加调试信息
            if len(track_boxes) > 0 or len(drop_boxes) > 0:
                print(f"File {stem}: track_boxes={len(track_boxes)}, drop_boxes={len(drop_boxes)}, total={len(pred_boxes)}")
        else:
            pred_boxes = read_pred_txt(paths['pred'], score_threshold=args.score_threshold)
        gt_boxes = read_gt_json(paths['gt'])
        results.append({"pred": pred_boxes, "gt": gt_boxes})

    # 2. 计算 mAP
    if track:
        if args.track_only:
            print("Track Only")
        else:
            print("Track + Drop")
        evaluator = MAPEvaluator([1, 4, 3, 2])
    else:
        print("detection")
        evaluator = MAPEvaluator([1, 4, 3, 2])
    # evaluator = MAPEvaluator(["car", "cyclist", "truck"])
    metrics, ap_per_class = evaluator.evaluate_map(results)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 3. 可视化
    evaluator.plot_map_curves(ap_per_class, save_path="ap_curve.png")
