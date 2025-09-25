import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import glob
import time
import copy
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
        
        self.seq_name = sequence.split('/')[-1]
        self.seq_path = sequence.replace("dataset_track", "dataset_track/offline_tracked")
        self.feats_root = sequence
        self.gt_root = os.path.join("/rss/yuanqingwen/refine_data/fusion", self.seq_name, "result")
        
        save_path = os.path.join(save_path, class_name)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = os.path.join(save_path, self.seq_name+".pkl")

        self.split_objs = False
        self.fitted = False

    def init_infos_from_tracking(self):
        """
        Function:
            Load the results of upstream modules in the format of
            dict (sequence_id as keys), and corresponding infos
        """        
        self.logger.info('Loading generated object tracks for %s set.' % self.split)
        feat_file_lists = []
        tracking_res = glob.glob(os.path.join(self.seq_path, "*", "tracking", "tracking-test-*.pkl"))
        tracking_res = sorted(tracking_res)
        feat_file_lists = glob.glob(os.path.join(self.feats_root, "model_pred_pkl", "*.pkl"))
        feats_dict = {
            os.path.basename(file).split(".pkl")[0]: file 
            for file in feat_file_lists
        }
        self.logger.info('Total object sequences for 4Dlabel dataset: %d.' % len(tracking_res))
        self.logger.info('Start to convert object tracks into frame-level format.')
        self.prepare_data_worker(tracking_res, feats_dict)

    def load_gt_jsons(self, gt_file_path):
        frame_infos = {
            "gt_boxes_lidar": [], 
            "gt_names": [], 
        }
        with open(gt_file_path, "rb") as file:
            data = json.load(file)
            json_data = data['result']['data']
        for item in json_data:
            x, y, z = item['3Dcenter'].values()
            w, l, h, yaw, _, _, _ = item['3Dsize'].values()
            class_name_ = class_mapping_7[item["sublabel"]]
            if class_name_ not in CLASSES_NAMES[self.class_name]:
                continue
            frame_infos["gt_boxes_lidar"].append([x, y, z, l, w, h, yaw])
            frame_infos['gt_names'].append(class_name_)
        return frame_infos
    
    def prepare_data_worker(self, res_files, feat_file_lists):
    
        for _, pred_files in enumerate(res_files):
            with open(pred_files, "rb") as file:
                data = pickle.load(file)
                tracking_infos = data[next(iter(data.keys()))]
            scene_ = pred_files.split("/")[-3]
            timestamps = sorted(glob.glob(os.path.join(self.seq_path, scene_, "splited", "*.txt")))
            # ========= frame-level dict =========
            output_dict = {}
            drop_obj_ids = []
            count_all_nums = 0
            for obj_idx, obj in tracking_infos.items():
                sample_idxs = obj['sample_idx']
                pred_bbox_full = obj['boxes_global']
                uuids = obj['uuid']

                boxes_names = obj['name']
                if boxes_names[0] not in CLASSES_NAMES[self.class_name]:
                    continue
                count_all_nums += 1
                # 遍历轨迹的每个帧
                for k, (frame_idx, pred_bbox, box_name, uuid) in enumerate(zip(sample_idxs, pred_bbox_full, boxes_names, uuids)):

                    # ========= 每帧记录已经匹配过的 GT =========
                    if frame_idx not in output_dict:
                        output_dict[frame_idx] = {
                            'boxes_global': [], 'score': [], 'obj_id': [], 'name': [],
                            'hit': [], 'pose': [], 'state': [], 'matched': [], 'matched_tracklet': [],
                            'gt_boxes_global': [], 'gt_obj_id': [], 'gt_name': [],
                            'uuid': [], 
                        }

                    frm_info = output_dict[frame_idx]

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
                    frm_info['uuid'].append(uuid)

                    frm_info['gt_boxes_global'].append(pred_bbox[:7])
                    frm_info['gt_obj_id'].append(obj_idx)
                    frm_info['gt_name'].append(box_name)

            # ========= 聚合到 track-level data_info =========
            data_info = {}
            for frm_id in tqdm(output_dict.keys(), total=len(output_dict.keys())):
                frm_info = output_dict[frm_id]
                timestamp = os.path.basename(timestamps[int(frm_id)])[:-4]
                feats_file = feat_file_lists[timestamp]
                with open(feats_file, "rb")as file:
                    feats_array = pickle.load(file)
                uuids = frm_info['uuid']
                for idx, obj_id in enumerate(frm_info['obj_id']):
                    if obj_id in drop_obj_ids:
                        continue
                    if obj_id not in data_info:
                        obj_info_tmp = {
                            'sequence_name': self.seq_name, 'obj_id': obj_id, 'name': frm_info['name'][idx],
                            'boxes_global': [], 'score': [], 'sample_idx': [], 'hit': [], 'pose': [], 'state': frm_info['state'][idx],
                            'matched': [], 'matched_tracklet': [], 'pts': [],
                            'gt_boxes_global': [], 'gt_obj_id': [], 'gt_name': []
                        }
                    else:
                        obj_info_tmp = data_info[obj_id]

                    if int(uuids[idx]) == 9999:
                        bbox_feats = prev_feats
                    else:
                        bbox_feats = feats_array[int(uuids[idx])][9:]
                    prev_feats = bbox_feats

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

                    obj_info_tmp['pts'].append(bbox_feats)

                    data_info[obj_id] = obj_info_tmp
            
            for obj_id, obj_info in data_info.items():
                for key in list(obj_info.keys()):
                    if key in ['sequence_name', 'obj_id', 'name', 'state']:
                        continue
                    elif key in ['pts', ]:
                        obj_info[key] = np.array(obj_info[key], dtype=object)
                    else:
                        obj_info[key] = np.array(obj_info[key])

            # # ========= 保存 =========
            new_name = self.save_path.replace(".pkl", f"_{pred_files.split('/')[-3]}.pkl")
            os.makedirs(os.path.dirname(new_name), exist_ok=True)
            with open(new_name, 'wb') as f:
                pickle.dump(data_info, f)
            del data_info
            print(new_name)


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
        # 'Cyclist': ['motorcycle', 'rider', 'bicycle'], 
        'Truck': ['truck', 'bus'], 
        # 'Pedestrian': ['pedestrian', ]
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
