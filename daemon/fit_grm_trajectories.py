import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import argparse
import sys
sys.path.insert(0, "/rss/liumengyuan/4D_label")
from fit_trajectory.trajectory_fitter import TrajectoryFitter

class_map_refine = {'Vehicle': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Truck': 4,}
id_to_class = {v: k for k, v in class_map_refine.items()}

LARGE_CLASSES = {"Vehicle", "Truck"}
SMALL_CLASSES = {"Pedestrian", "Cyclist"}

# === 工具函数 ===
def parse_name(raw_name):
    # 处理 list[tensor]
    if isinstance(raw_name, (list, tuple)):
        return [parse_name(x) for x in raw_name]

    # 处理 torch.Tensor
    if isinstance(raw_name, torch.Tensor):
        class_id = int(raw_name.item())  # 从 tensor 提取数值
        return id_to_class.get(class_id, "")

    # 处理数字
    if isinstance(raw_name, (int, float)):
        return id_to_class.get(int(raw_name), "")

    # 处理字符串
    if isinstance(raw_name, str):
        # 例如 "Vehicle"
        return raw_name if raw_name in class_map_refine else ""
    
    return ""

def is_valid_for_fitting(x, y):
    return not (np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.any(np.isnan(y)) or np.any(np.isinf(y)))

def is_static_trajectory_segment(boxes_np, is_small=False, is_large=False):
    # 默认阈值（中等）
    pos_thresh = 0.4
    yaw_thresh = 0.2

    # 小目标更严格
    if is_small:
        pos_thresh = 0.2
        yaw_thresh = 0.15

    # 大目标更宽松
    if is_large:
        pos_thresh = 0.5
        yaw_thresh = 0.25

    # breakpoint()
    pos_std = np.std(boxes_np[:, :2], axis=0)
    yaw_diff = np.abs(np.diff(boxes_np[:, 6]))
    yaw_diff = np.minimum(yaw_diff, 2*np.pi - yaw_diff)
    yaw_median = np.median(yaw_diff)

    # breakpoint()
    return (
        np.max(pos_std) < pos_thresh and 
        yaw_median < yaw_thresh
    )


def extract_timestamps_from_sample_idx(sample_idx_list, splited_txt_folder):
    # 获取所有txt文件，排序
    splited_files = sorted(os.listdir(splited_txt_folder))
    timestamps = []
    for idx in sample_idx_list:
        idx = int(idx)
        try:
            if idx < 0 or idx >= len(splited_files):
                print(f"[IndexError] idx={idx} 超出范围 (0 ~ {len(splited_files)-1})")
                timestamps.append(np.nan)
                continue
            filename = splited_files[idx]  # e.g. "1748041526.700000.txt"
            timestamp_str = filename.replace(".txt", "")
            # print(f"Processing file: {filename}, idx={idx}")
            timestamps.append(float(timestamp_str))  # 转换为 float 时间戳
        except Exception as e:
            timestamps.append(np.nan)

    return np.array(timestamps)

def flush_segment(segment, segment_idx, 
                  is_small, is_large, splited_txt_folder, fitter):
    """处理一段连续轨迹的拟合/平滑，返回新的结果列表"""
    results = []

    if len(segment) == 0:
        return results

    segment_np = np.array(segment)
    segment_idx_np = np.array(segment_idx)
    timestamp_np = extract_timestamps_from_sample_idx(segment_idx_np, splited_txt_folder)

    if len(segment_np) >= 4:
        if is_static_trajectory_segment(segment_np, is_small, is_large):
            mean_box = segment_np[0].copy()  # 取第一帧的 box 作为基础
            mean_box[0] = np.mean(segment_np[:, 0])  # x 均值
            mean_box[1] = np.mean(segment_np[:, 1])  # y 均值
            # mean_box[7] = 0.0  # vx
            # mean_box[8] = 0.0  # vy
            # 扩展为整个段长度
            static_boxes = np.tile(mean_box, (len(segment_np), 1))
            results.extend(static_boxes.tolist())
        else:
            if len(segment_np) != len(timestamp_np):
                print(f"Warning: Mismatched lengths after outlier removal. Segment length: {len(segment_np)}, Timestamps length: {len(timestamp_np)}")
                return results
            fitted, success = fitter.fit_with_time(segment_np, timestamp_np, is_small, is_large)
            results.extend(fitted.tolist())
    else:
        results.extend(segment_np.tolist())
    if len(results) != len(segment):
        print(f"Warning: Mismatched lengths after processing segment. Original length: {len(segment)}, Result length: {len(results)}")
    return results


def fit_boxes_with_segmentation(obj_data, splited_txt_folder):
    boxes = obj_data.get("boxes_lidar", [])
    sample_idx = obj_data.get("frame_id", [])

    if len(boxes) == 0 or len(sample_idx) == 0:
        return []

    fitter = TrajectoryFitter(verbose=True)

    boxes_np = np.array(boxes)
    boxes_np = boxes_np.squeeze(1)  # 去掉中间维度 1
    # print(boxes_np.shape)
    sample_idx_np = np.array(sample_idx)
    # print(f"Processing {len(boxes_np)} boxes with {len(sample_idx_np)} samples.")
    raw_name = obj_data.get("name", "")
    # print(f"Raw name: {raw_name}")
    # breakpoint()
    name = parse_name(raw_name)
    # print(name)
    # breakpoint()
    category = name[0] if name else ""  # 取第一个
    is_small = category in SMALL_CLASSES
    is_large = category in LARGE_CLASSES


    fitted_result = []
    segment = []
    segment_idx = []

    for i in range(len(sample_idx_np)):
        if i == 0 or int(sample_idx_np[i]) == int(sample_idx_np[i - 1]) + 1:
            segment.append(boxes_np[i])
            segment_idx.append(sample_idx_np[i])
        else:
            fitted_result.extend(
                flush_segment(segment, segment_idx, is_small, is_large, splited_txt_folder, fitter)
            )
            segment = [boxes_np[i]]
            segment_idx = [sample_idx_np[i]]

    fitted_result.extend(
        flush_segment(segment, segment_idx, is_small, is_large, splited_txt_folder, fitter)
    )

    return fitted_result


def fit_and_save_track(track_dict, save_dir, scene_name, splited_txt_folder):
    for token, obj_dict in track_dict.items():
        fitted_tracks = {}
        stats = {'total': 0, 'success': 0, 'fallback': 0}

        for obj_id, obj_data in obj_dict.items():
            stats['total'] += 1
            boxes = obj_data.get("boxes_lidar", [])
            if len(boxes) == 0:
                continue
            flattened_boxes = []
            for b in boxes:
                # 如果是 tensor，先转 numpy
                if isinstance(b, torch.Tensor):
                    arr = b.cpu().numpy()
                else:
                    arr = np.array(b)
                # 去掉多余维度
                arr = np.squeeze(arr)
                # 转成 list 并保存
                flattened_boxes.append(arr.tolist())

            fitted = fit_boxes_with_segmentation(obj_data, splited_txt_folder)

            if fitted is None or len(fitted) != len(boxes):
                stats['fallback'] += 1
            else:
                stats['success'] += 1

            # breakpoint()
            fitted_tracks[obj_id] = {
                "boxes_global_fitted": fitted,
                "original_boxes_global": flattened_boxes,
                "name":  parse_name(obj_data.get("name", [])),
                "sample_idx": obj_data.get("frame_id", [])
            }

        print(f"[Scene:{scene_name}] token={token} | total={stats['total']} | fitted={stats['success']} | fallback={stats['fallback']}")

        if fitted_tracks:
            save_path = os.path.join(save_dir, f"{scene_name}_fitted.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(fitted_tracks, f)
        else:
            print(f"[Skip] token={token} | No valid objects to fit in scene {scene_name}")

def main(src_dir, dist_dir, fit_folder):
    os.makedirs(dist_dir, exist_ok=True)
    tracking_files = glob(os.path.join(src_dir, "*", "eval/epoch_20/test", "Vehicle_geometry_test.pkl"))

    for tracking_file in tqdm(tracking_files, desc="Fitting 3D Trajectories"):
        try:
            with open(tracking_file, "rb") as f:
                tracking_data = pickle.load(f)

            # 提取 run_id（例如 "20250318114535"）
            run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(tracking_file)))))

            # 构造 splited_txt_folder
            splited_txt_folder = os.path.join(
                "/rss/yuanqingwen/data/4D_label/dataset_track/vals_data/offline_tracked",
                "train_sh_3d_road_6_20250517_5000_lx",
                run_id,
                "splited"
            )

            fit_dir = os.path.join(dist_dir, fit_folder)
            os.makedirs(fit_dir, exist_ok=True)
            fit_and_save_track(tracking_data, fit_dir, run_id, splited_txt_folder)

        except Exception as e:
            print(f"Failed to process file: {tracking_file}, Error: {e}")

    print(f"\nAll 3D trajectory fitting is completed and the results are saved in: {dist_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit 3D trajectories from tracked boxes and save.")
    parser.add_argument('--src', default="/rss/yuanqingwen/origin_version/DetZero/refining/output/ref_model_cfgs/vehicle_grm_model/shanghai6", help="Source directory containing tracking data")
    parser.add_argument('--dist', default="/rss/liumengyuan/4D_label/dataset_track/train_sh_3d_road_6_20250517_5000_lx", help="Destination directory to save fitted tracks")
    parser.add_argument("--fit_folder", default="fitted_tracks_grm", help="folder containing fitted tracks")

    args = parser.parse_args()
    main(args.src, args.dist, args.fit_folder)