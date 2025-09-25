import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import sys

sys.path.insert(0, "/rss/liumengyuan/4D_label")
from trajectory_fitter import TrajectoryFitter

LARGE_CLASSES = {"car", "truck", "bus"}
SMALL_CLASSES = {"pedestrian", "rider", "motorcycle"}

# === 工具函数 ===
def is_valid_for_fitting(x, y):
    return not (np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.any(np.isnan(y)) or np.any(np.isinf(y)))

def is_static_trajectory_segment(boxes_np, is_small=False, is_large=False):
    # 默认阈值（中等）
    pos_thresh = 0.6
    speed_thresh = 0.2
    yaw_thresh = 0.2

    # 小目标更严格
    if is_small:
        pos_thresh = 0.4
        speed_thresh = 0.05
        yaw_thresh = 0.15

    # 大目标更宽松
    if is_large:
        pos_thresh = 0.8
        speed_thresh = 0.25
        yaw_thresh = 0.25

    pos_std = np.std(boxes_np[:, :2], axis=0)
    yaw_diff = np.abs(np.diff(boxes_np[:, 6]))
    yaw_diff = np.minimum(yaw_diff, 2*np.pi - yaw_diff)
    yaw_median = np.median(yaw_diff)
    # breakpoint()
    # vx = np.zeros((boxes_np.shape[0], ))
    # vy = np.zeros((boxes_np.shape[0], ))
    vx = boxes_np[:, 7]
    vy = boxes_np[:, 8]
    speed = np.hypot(vx, vy)
    median_speed = np.median(speed)

    return (
        np.max(pos_std) < pos_thresh and 
        median_speed < speed_thresh and
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
            # print(f"Processing file: {filename}")
            timestamp_str = filename.replace(".txt", "")
            # print(f"Extracted timestamp: {timestamp_str}")
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
            # print(f"Segment is static, using mean box for {len(segment_np)} points.")
            # 仅对 x, y 求平均
            mean_box = segment_np[0].copy()  # 取第一帧的 box 作为基础
            mean_box[0] = np.mean(segment_np[:, 0])  # x 均值
            mean_box[1] = np.mean(segment_np[:, 1])  # y 均值
            mean_box[7] = 0.0  # vx
            mean_box[8] = 0.0  # vy
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
    # boxes = obj_data.get("boxes_global", [])
    boxes = obj_data.get("gt_boxes_global", [])
    sample_idx = obj_data.get("sample_idx", [])

    if len(boxes) == 0 or len(sample_idx) == 0:
        return []

    fitter = TrajectoryFitter(verbose=True)

    boxes_np = np.array(boxes)
    sample_idx_np = np.array(sample_idx)
    raw_name = obj_data.get("name", "")
    # breakpoint()
    is_small = raw_name[0] in SMALL_CLASSES
    is_large = raw_name[0] in LARGE_CLASSES

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
        if token == "gt_infos":
            break
        fitted_tracks = {}
        stats = {'total': 0, 'success': 0, 'fallback': 0}

        for obj_id, obj_data in obj_dict.items():
            stats['total'] += 1
            boxes = obj_data.get("boxes_global", [])
            if len(boxes) == 0:
                continue

            fitted = fit_boxes_with_segmentation(obj_data, splited_txt_folder)

            if fitted is None or len(fitted) != len(boxes):
                stats['fallback'] += 1
            else:
                stats['success'] += 1

            fitted_tracks[obj_id] = {
                "boxes_global_fitted": fitted,
                "original_boxes_global": boxes,
                "name": obj_data.get("name", []),
                "sample_idx": obj_data.get("sample_idx", [])
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
    tracking_files = glob(os.path.join(src_dir, "*", "tracking", "tracking-test-*.pkl"))
    # print(len(tracking_files))

    for tracking_file in tqdm(tracking_files, desc="Fitting 3D Trajectories"):
        try:
            with open(tracking_file, "rb") as f:
                tracking_data = pickle.load(f)

            scene_root = os.path.dirname(os.path.dirname(tracking_file))
            scene_name = os.path.basename(scene_root)
            splited_txt_folder = os.path.join(scene_root, "splited")
            # print(splited_txt_folder)
            fit_dir = os.path.join(dist_dir, fit_folder)
            os.makedirs(fit_dir, exist_ok=True)

            fit_and_save_track(tracking_data, fit_dir, scene_name, splited_txt_folder)
        except Exception as e:
            print(f"Failed to process file: {tracking_file}, Error: {e}")

    print(f"\nAll 3D trajectory fitting is completed and the results are saved in: {dist_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit 3D trajectories from tracked boxes and save.")
    parser.add_argument('--src', default="/rss/yuanqingwen/data/4D_label/dataset_track/vals_data/offline_tracked/train_sh_3d_road_6_20250517_5000_lx", help="Source directory containing tracking data")
    parser.add_argument('--dist', default="dataset_track/offline_tracked/train_sh_3d_road_6_20250517_5000_lx", help="Destination directory to save fitted tracks")
    parser.add_argument("--fit_folder", default="fitted_tracks_bev", help="folder containing fitted tracks")

    args = parser.parse_args()
    main(args.src, args.dist, args.fit_folder)