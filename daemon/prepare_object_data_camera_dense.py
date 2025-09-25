import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import glob
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


import numpy as np
import torch
import torch.nn.functional as F
import json
from collections import defaultdict

from fit_bev_trajectories import fit_boxes_with_segmentation

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sensor_modules import intersection_modules, roadsection_modules, intersection_lidar_modules
from base_config import class_mapping_7, geo_mapping, class_select
from detzero_utils.common_utils import create_logger, multi_processing
# from detzero_utils.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu_v2, points_in_boxes_bev_gpu
from detzero_utils.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu_v2

def pad_rois_to_same_size(roi_feats, target_size=(200, 200)):
    """
    :param roi_feats: Tensor of shape (B, C, H_i, W_i) representing a batch of ROIs.
    :param target_size: Target size for padding (H_out, W_out).
    :return: Tensor of shape (B, C, H_out, W_out), padded ROIs.
    """
    B, C, H, W = roi_feats.shape
    target_H, target_W = target_size

    # Calculate padding for height and width
    padding_top = max(0, target_H - H)
    padding_bottom = max(0, target_H - H - padding_top)
    padding_left = max(0, target_W - W)
    padding_right = max(0, target_W - W - padding_left)

    # Apply padding (padding order is left, right, top, bottom)
    padded_roi_feats = F.pad(roi_feats, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0)

    return padded_roi_feats

def bev_roi_pool(bev_feats, boxes, class_name):
    """
    :param bev_feats: (H, W, C) BEV feature map
    :param boxes: (B, 7) [x, y, z, l, w, h, yaw] for each box
    :param output_size: (out_x, out_y) output size of RoI pooling
    :return: (B, 11) pooled RoI features mapped to 11-dimensional vector
    """
    bev_feats = torch.tensor(bev_feats, dtype=torch.float32).permute(2, 0, 1)
    C, H, W = bev_feats.shape
    N, _ = boxes.shape

    boxes = torch.tensor(boxes, dtype=torch.float32)
    # Boxes: (B, 7) - [x, y, z, l, w, h, yaw]
    # Extract relevant parameters
    x, y, z, l, w, h, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6]

    # Convert to pixel space in BEV feature map
    scale_x = H / 200.0  # Assuming BEV map spans [-100, 100] in x-direction
    scale_y = W / 200.0  # Assuming BEV map spans [-100, 100] in y-direction

    min_x = x - l / 2  # 'l' is the length
    max_x = x + l / 2
    min_y = y - w / 2  # 'w' is the width
    max_y = y + w / 2

    # Map the 2D box to BEV pixel space
    min_x_pixel = ((min_x + 100) * scale_x).clamp(0, H-1).long()
    max_x_pixel = ((max_x + 100) * scale_x).clamp(0, H-1).long()
    min_y_pixel = ((min_y + 100) * scale_y).clamp(0, W-1).long()
    max_y_pixel = ((max_y + 100) * scale_y).clamp(0, W-1).long()

    pooled_features = []

    for i in range(N):
        # Crop out RoI from BEV map (using min_x, max_x, min_y, max_y)
        roi = bev_feats[:, min_x_pixel[i]:max_x_pixel[i], min_y_pixel[i]:max_y_pixel[i]]
        padding_roi = pad_rois_to_same_size(roi.unsqueeze(0), PADDING_SIZE[class_name])
        # Perform RoI pooling (e.g., average pooling)
        # pooled = F.adaptive_avg_pool2d(roi, output_size)
        pooled_features.append(padding_roi)

    pooled_features = torch.stack(pooled_features)  # Shape: (B, C, out_x, out_y)

    # Now, pooled_features has shape (B, C, out_x, out_y), apply a fully connected layer
    # to map it to 11-dimensional vector.
    pooled_features_flat = pooled_features.view(N, C, -1).permute(0, 2, 1)  # Flatten to (B, C * out_x * out_y)
    return pooled_features_flat
    # Fully connected layer to map the features to 11 dimensions
    fc_layer = torch.nn.Linear(pooled_features_flat.shape[1], 11)  # Map to 11 dimensions
    features_11d = fc_layer(pooled_features_flat)

    return features_11d

def vis_bev_feature(bev_feats, img_timestamp):
    pca = PCA(n_components=3)
    bev_rgb = pca.fit_transform(bev_feats)  
    bev_rgb -= bev_rgb.min()
    bev_rgb /= bev_rgb.max()
    bev_rgb = bev_rgb.reshape(200, 200, 3)
    saved_path = os.path.join("/rss/yuanqingwen/origin_version/DetZero/daemon", img_timestamp+".png")
    plt.imshow(bev_rgb)
    plt.axis("off")
    plt.savefig(saved_path)
    plt.close()


def fix_yaw_angles(yaw_angles):
    # 修复后的yaw角度列表
    fixed_yaw_angles = yaw_angles.copy()

    for i in range(1, len(yaw_angles)):
        prev_yaw = fixed_yaw_angles[i-1]
        curr_yaw = fixed_yaw_angles[i]
        
        # 判断是否发生了跳变
        if abs(curr_yaw - prev_yaw) > np.pi / 3:  # 如果差值大于pi/3
            # 如果跳变超过pi，表示跨越了正负边界，进行修正
            if curr_yaw > prev_yaw:
                fixed_yaw_angles[i] -= 2 * np.pi  # 调整为负向
            else:
                fixed_yaw_angles[i] += 2 * np.pi  # 调整为正向

    return fixed_yaw_angles

class ObjectDataPrepare():
    """
    Function:
        Prepare data infos used for refining module
    """
    def __init__(self, class_name, split='train', enlarge_scale=1.1, crop_on_bev=False, 
                 workers=1, logger=None, sequence=None, save_path=None):
        
        self.class_name = class_name
        self.split = split
        
        self.enlarge_scale = enlarge_scale
        self.crop_on_bev = crop_on_bev
        self.workers = workers
        self.logger = logger
        
        self.seq_path = sequence
        self.seq_name = sequence.split('/')[-1]
        save_path = os.path.join(save_path, class_name)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = os.path.join(save_path, self.seq_name+".pkl")

        self.vis = False
        self.fitted = False

    def init_infos_from_tracking(self):
        """
        Function:
            Load the results of upstream modules in the format of
            dict (sequence_id as keys), and corresponding infos
        """        
        self.logger.info('Loading generated object tracks for %s set.' % self.split)
        pcd_file_lists = []
        tracking_res = glob.glob(os.path.join(self.seq_path, "*", "tracking", "tracking*.pkl"))
        tracking_res = sorted(tracking_res)
        feats_res = os.path.join(self.seq_path.replace("/offline_tracked", ""), "bev_embed")
        pcd_file_lists = glob.glob(os.path.join(feats_res, "*.npy"))
        pcd_dict = {
            os.path.basename(file).split("_")[0]: file 
            for file in pcd_file_lists
        }
        self.logger.info('Total object sequences for 4Dlabel dataset: %d.' % len(pcd_file_lists))
        self.logger.info('Start to convert object tracks into frame-level format.')
        self.prepare_data_worker(tracking_res, pcd_dict)


    def prepare_data_worker(self, res_files, pcd_file_lists):
        category_threshold = {
            "pedestrian": 2.0,
            "bicycle": 3.0,
            "motorcycle": 3.0,
            "rider": 9.0,
            "car": 5.5,
            "truck": 7.0,  # 5.5, 8
            "bus": 7.0
        }

        for _, pred_files in enumerate(res_files):
            with open(pred_files, "rb") as file:
                data = pickle.load(file)
                tracking_infos = data[next(iter(data.keys()))]
            tmp = os.path.dirname(pred_files).replace("cam_res", "4D_label/dataset_track")
            gt_files = glob.glob(os.path.join(tmp, "tracking-test-*.pkl"))[0]
            with open(gt_files, "rb") as file:
                data = pickle.load(file)
                gt_infos = data["gt_infos"]
            feat_file_root = pred_files.split("/tracking/tracking")[0] + "/splited"
            feat_files_ = sorted([file.replace(".txt", "") for file in os.listdir(feat_file_root)])

            # ========= frame-level dict =========
            output_dict = {}

            for obj_idx, obj in tracking_infos.items():
                nums_error = 0
                sample_idxs = obj['sample_idx']
                pred_bbox_full = obj['boxes_global']

                yaws = pred_bbox_full[:, 6]
                new_yaw = fix_yaw_angles(yaws) 
                pred_bbox_full[:, 6] = new_yaw

                boxes_names = obj['name']
                
                if boxes_names[0] not in CLASSES_NAMES[self.class_name]:
                    continue
                
                # 遍历轨迹的每个帧
                for k, (frame_idx, pred_bbox, box_name) in enumerate(zip(sample_idxs, pred_bbox_full, boxes_names)):
                    gt_frame_boxes = gt_infos[frame_idx]['gt_boxes_lidar']
                    gt_boxes_new = [
                        [x, y, z, l, w, h, (yaw + math.pi / 2) % (2 * math.pi)]
                        for (x, y, z, w, l, h, yaw) in gt_frame_boxes
                    ]
                    gt_frame_boxes = np.array(gt_boxes_new)
                    gt_frame_names = gt_infos[frame_idx]['gt_names']

                    # ========= 每帧记录已经匹配过的 GT =========
                    if frame_idx not in output_dict:
                        output_dict[frame_idx] = {
                            'boxes_global': [], 'score': [], 'obj_id': [], 'name': [],
                            'hit': [], 'pose': [], 'state': [], 'matched': [], 'matched_tracklet': [],
                            'gt_boxes_global': [], 'gt_obj_id': [], 'gt_name': [],
                            'used_gt_indices': set()  # 用于记录已匹配 GT
                        }

                    frm_info = output_dict[frame_idx]

                    # 找出类别匹配且未匹配的 GT 索引
                    valid_gt_mask = np.array(gt_frame_names) == box_name
                    valid_gt_indices = [idx for idx in np.where(valid_gt_mask)[0] if idx not in frm_info['used_gt_indices']]

                    gt_index = None
                    if len(valid_gt_indices) > 0:
                        gt_frame_boxes_filtered = gt_frame_boxes[valid_gt_indices]

                        pred_center = pred_bbox[:2].reshape(1, 2)
                        gt_centers = gt_frame_boxes_filtered[:, :2]

                        dist_array = cdist(gt_centers, pred_center).reshape(-1)
                        thr = category_threshold[box_name]

                        min_dist = dist_array.min()
                        if min_dist <= thr:
                            min_idx = dist_array.argmin()
                            gt_index = valid_gt_indices[min_idx]
                            # 标记 GT 已匹配
                            frm_info['used_gt_indices'].add(gt_index)

                    # ========= 添加预测信息 =========
                    frm_info['boxes_global'].append(pred_bbox[:7])
                    frm_info['score'].append(obj['score'][k])
                    frm_info['obj_id'].append(obj_idx)
                    frm_info['name'].append(box_name)
                    frm_info['hit'].append(obj['hit'][k])
                    frm_info['pose'].append(obj['pose'][k])
                    frm_info['state'].append(obj['state'])
                    frm_info['matched'].append(True)
                    frm_info['matched_tracklet'].append(False)

                    if gt_index is not None:
                        frm_info['gt_boxes_global'].append(gt_frame_boxes[gt_index])
                        frm_info['gt_obj_id'].append(obj_idx)
                        frm_info['gt_name'].append(box_name)
                    else:
                        nums_error += 1
                        frm_info['gt_boxes_global'].append(pred_bbox[:7])
                        frm_info['gt_obj_id'].append(obj_idx)
                        frm_info['gt_name'].append(box_name)
            # ========= 聚合到 track-level data_info =========
            data_info = {}
            for frm_id in tqdm(output_dict.keys(), total=len(output_dict.keys())):
                if int(frm_id) >= len(feat_files_):
                    continue
                frm_info = output_dict[frm_id]

                boxes_enlarge = np.array(frm_info['boxes_global']).copy()
                boxes_enlarge[:, 3:6] *= self.enlarge_scale
                if self.crop_on_bev:
                    boxes_enlarge[:, 5] = 256

                bev_feats = np.load(pcd_file_lists[feat_files_[int(frm_id)]])[0]
                bev_feats_roi = bev_feats.reshape(200, 200, 256)
                if self.vis:
                    vis_bev_feature(bev_feats, feat_files_[int(frm_id)])
                bboxes_roi_feats = bev_roi_pool(bev_feats_roi, boxes_enlarge, self.class_name)

                for idx, obj_id in enumerate(frm_info['obj_id']):
                    if obj_id not in data_info:
                        obj_info_tmp = {
                            'sequence_name': self.seq_name, 'obj_id': obj_id, 'name': frm_info['name'][idx],
                            'boxes_global': [], 'score': [], 'sample_idx': [], 'hit': [], 'pose': [], 'state': frm_info['state'][idx],
                            'matched': [], 'matched_tracklet': [], 'pts': [],
                            'gt_boxes_global': [], 'gt_obj_id': [], 'gt_name': []
                        }
                    else:
                        obj_info_tmp = data_info[obj_id]

                    obj_info_tmp['boxes_global'].append(frm_info['boxes_global'][idx])
                    obj_info_tmp['score'].append(frm_info['score'][idx])
                    obj_info_tmp['sample_idx'].append(frm_id)
                    obj_info_tmp['hit'].append(frm_info['hit'][idx])
                    obj_info_tmp['pose'].append(frm_info['pose'][idx])
                    obj_info_tmp['matched'].append(frm_info['matched'][idx])
                    obj_info_tmp['matched_tracklet'].append(frm_info['matched_tracklet'][idx])
                    obj_info_tmp['gt_boxes_global'].append(frm_info['gt_boxes_global'][idx])
                    obj_info_tmp['gt_obj_id'].append(frm_info['gt_obj_id'][idx])
                    obj_info_tmp['gt_name'].append(frm_info['gt_name'][idx])
                    obj_info_tmp['pts'].append(bboxes_roi_feats[idx])

                    data_info[obj_id] = obj_info_tmp

            for obj_id, obj_info in data_info.items():
                if self.fitted:
                    gt_boxes_global = np.array(obj_info['gt_boxes_global'])
                    zeros = np.zeros((gt_boxes_global.shape[0], 2))
                    expanded_gt_boxes_global = np.hstack((gt_boxes_global, zeros))
                    obj_info['gt_boxes_global'] = expanded_gt_boxes_global

                    fitted = fit_boxes_with_segmentation(obj_info, feat_file_root)
                    obj_info['gt_boxes_global'] = np.array(fitted)[:, :7]
                for key in list(obj_info.keys()):
                    if key in ['sequence_name', 'obj_id', 'name', 'state']:
                        continue
                    elif key in ['pts', ]:
                        obj_info_tensor = torch.stack(obj_info[key])  # 堆叠成一个张量
                        obj_info_array = obj_info_tensor.numpy()
                        obj_info[key] = np.array(obj_info_array, dtype=np.float32)
                    else:
                        obj_info[key] = np.array(obj_info[key])
            # ========= 保存 =========
            new_name = self.save_path.replace(".pkl", f"_{pred_files.split('/')[-3]}.pkl")
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            with open(new_name, 'wb') as f:
                pickle.dump(data_info, f)
            del data_info
            print(new_name)

    """
    dict_keys([ 'sequence_name', 'obj_id', 'name', 'boxes_global', 
                'score', 'sample_idx', 'hit', 'pose', 'state', 'matched', 
                'matched_tracklet', 'pts', 'gt_boxes_global', 'gt_obj_id', 'gt_name'])
    """
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('root-path', type=str, required=True)
    parser.add_argument('save-path', type=str, required=True)
    parser.add_argument('--enlarge_scale', type=float, default=1.1,
                        help='scale-up raito of object proposals')
    parser.add_argument('--crop_on_bev', type=bool, default=False,
                        help='whether to crop object points on bird-eye view')
    parser.add_argument('--workers', type=int, default=1,
                        help='whether to use multi-process preparation')
    parser.add_argument('--split', type=str, default='train')
    
    args = parser.parse_args()


    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    logger = create_logger()

    CLASSES_NAMES = {
        'Vehicle': ['car', ], 
        'Cyclist': ['motorcycle', 'rider', 'bicycle'], 
        'Truck': ['truck', 'bus'], 
        # 'Pedestrian': ['pedestrian', ]
        }
    PADDING_SIZE = {
        'Vehicle': (8, 3), 
        'Cyclist': (3, 2), 
        'Truck': (32, 4), 
    }

    seq_lists = ["train_sh_3d_road_2_20250531_5000_lx", ]

    seq_lists = [os.path.join(args.root_path, seq) for seq in seq_lists]
    
    class_name_keys = CLASSES_NAMES.keys()
    for class_name in class_name_keys:
        for sequence in seq_lists:
            logger.info('Start to process %s data in sequence: %s ...' % (class_name, sequence.split('/')[-1])) 
            dataset = ObjectDataPrepare(
                class_name=class_name,
                split=args.split,
                enlarge_scale=args.enlarge_scale,
                crop_on_bev=args.crop_on_bev,
                workers=args.workers,
                logger=logger, 
                sequence=sequence, 
                save_path=args.save_path
                )
            dataset.init_infos_from_tracking()
