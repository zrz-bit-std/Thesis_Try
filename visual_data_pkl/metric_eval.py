import argparse
from pathlib import Path
from tqdm import tqdm
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from class_mapping import class_mapping_7, class_map_refine, classname2id_7

def create_json_map():
    pred_root = Path("/data1/turbo_data/yuanqingwen/data/4D_label/dataset_track/train_hy_4d_road_7_20250318_one_frame/model_pred")
    refine_root = Path("/data1/turbo_data/yuanqingwen/data/test_4D_label/origin/model_res/eval_res/train_hy_4d_road_7_20250318_one_frame")

    result_base = Path("/data1/turbo_data/RALG/data/3.0_pro/Inter+Road/label/hengyang_road_7/p2_truck_lx/train_hy_4d_road_7_20250318_one_frame/result")
    gt_json_dirs = list(result_base.rglob("result_json"))

    # 收集所有 JSON 文件
    all_gt_json_files = []
    for gt_root in gt_json_dirs:
        all_gt_json_files.extend(list(gt_root.glob("*.json")))

    # 为快速查找，建立一个字典：stem -> path
    gt_dict = {p.stem: p for p in all_gt_json_files}

    file_map = {}

    for refine_file in tqdm(refine_root.glob("*/*.txt")):
        stem = refine_file.stem  # 比如 "1742269524.900001"
        
        pred_file = pred_root / f"{stem}.txt"
        
        # 在 gt_dict 中找到对应的 json
        gt_file = gt_dict.get(stem)
        if gt_file is None:
            print(f"Warning: gt file not found for {stem}")
            continue  # 如果找不到 gt，可以跳过或做别的处理
        
        file_map[stem] = {
            "pred": str(pred_file),
            "gt": str(gt_file),
            "refine": str(refine_file)
        }
    save_path = Path("train_hy_4d_road_7_20250318_one_frame_eval_map.json")  # 当前目录下

    with open(save_path, "w") as f:
        json.dump(file_map, f, indent=4)  # indent=4 美化格式

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

def calculate_ap_bev(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    pred_boxes: List of [x,y,w,l,heading,score]
    gt_boxes: List of [x,y,w,l,heading]
    返回 AP (AUC法)
    """
    if len(pred_boxes) == 0:
        return 0.0
    if len(gt_boxes) == 0:
        return 0.0

    # 按置信度排序预测框
    pred_boxes = sorted(pred_boxes, key=lambda x: x[-1], reverse=True)
    
    gt_matched = np.zeros(len(gt_boxes))
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i, pred in enumerate(pred_boxes):
        max_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            iou = bev_iou(pred[:5], gt)  # BEV IoU
            if iou > max_iou:
                max_iou = iou
                best_gt_idx = j
        if max_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[best_gt_idx] = 1
        else:
            fp[i] = 1

    # Precision-Recall
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls = tp_cum / len(gt_boxes)

    # AP via trapezoidal integration
    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    ap = np.trapz(precisions, recalls)
    return ap


def read_pred_txt(file_path):
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
            bbox = [x, y, z, w, l, h, yaw, confidence, class_name]
            boxes.append(bbox)  
    return boxes

def read_refine_txt(file_path):
    """
    返回 list: [x,y,w,l,heading,score]
    """
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            vals = list(map(float, vals))
            x, y, z, l, w, h, yaw, class_name = vals
            confidence = 1.0
            bbox = [x, y, z+(h/2.0), w, l, h, yaw, confidence, class_name]
            boxes.append(bbox)  
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
            iou_thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
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
            metrics[f"map/{cls}/iou@max"] = max(aps)
            for thr, ap in zip(self.iou_thresholds, aps):
                metrics[f"map/{cls}/iou@{thr:.2f}"] = ap

        # 平均
        metrics["map/mean/iou@max"] = np.mean([max(v) for v in ap_per_class.values()])
        for i, thr in enumerate(self.iou_thresholds):
            metrics[f"map/mean/iou@{thr:.2f}"] = np.mean([ap_per_class[c][i] for c in self.map_classes])

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
            plt.plot(
                self.iou_thresholds,
                ap_per_class[cls],
                marker="o",
                label=cls
            )

        mean_curve = np.mean([ap_per_class[c] for c in self.map_classes], axis=0)
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

    with open("train_hy_4d_road_7_20250318_one_frame_eval_map.json", "r") as f:
        file_map = json.load(f)

    # # refine 框 AP
    # ap_results, precision_results, recall_results = calculate_total_ap(file_map, refine=True, iou_threshold=0.3)
    # print("\n=== Refine AP per class ===")
    # for cls in ap_results.keys():
    #     print(f"{cls}: AP={ap_results[cls]:.4f}")
    #     print(f"Precision: {precision_results[cls]}")
    #     print(f"Recall   : {recall_results[cls]}\n")

    # # detection 框 AP
    # ap_results, precision_results, recall_results = calculate_total_ap(file_map, refine=False, iou_threshold=0.3)
    # print("\n=== Detection AP per class ===")
    # for cls in ap_results.keys():
    #     print(f"{cls}: AP={ap_results[cls]:.4f}")
    #     print(f"Precision: {precision_results[cls]}")
    #     print(f"Recall   : {recall_results[cls]}\n")

    refine = False

    # 1. 构造 results 格式
    results = []
    for stem, paths in tqdm(file_map.items(), desc="Loading files"):
        if refine:
            pred_boxes = read_refine_txt(paths['refine'])
        else:
            pred_boxes = read_pred_txt(paths['pred'])
        gt_boxes = read_gt_json(paths['gt'])
        results.append({"pred": pred_boxes, "gt": gt_boxes})

    # 2. 计算 mAP
    if refine:
        evaluator = MAPEvaluator([1, 4, 3])
    else:
        evaluator = MAPEvaluator([1, 4, 3, 2])
    # evaluator = MAPEvaluator(["car", "cyclist", "truck"])
    metrics, ap_per_class = evaluator.evaluate_map(results)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 3. 可视化
    evaluator.plot_map_curves(ap_per_class, save_path="ap_curve.png")


    

    




    

    