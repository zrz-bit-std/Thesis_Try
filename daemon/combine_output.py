import os
import pickle
from collections import defaultdict
import argparse
from pathlib import Path

import math
from tqdm import tqdm
from functools import partial
import concurrent.futures as futures
import numpy as np

from detzero_utils.common_utils import create_logger

from detzero_track.utils.transform_utils import transform_boxes3d
from detzero_track.utils.data_utils import sequence_list_to_dict, dict_to_sequence_list


def load_pkl(path):
    with open(path, 'rb') as f:
        info = pickle.load(f)
        if len(list(info.keys())) == 0:
            return None
        res_data = info[next(iter(info.keys()))]
    return res_data

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def combine_det(combine_data, drop_path):
    drop_data = load_pkl(drop_path)
    combine_data = sequence_list_to_dict(combine_data)
    seq_names = list(combine_data)
    
    for seq in seq_names:
        frames = list(combine_data[seq].keys())
        
        for frm in frames:
            for key in ['boxes_lidar', 'name', 'score']:
                combine_data[seq][frm][key] = np.concatenate([
                    combine_data[seq][frm][key],
                    drop_data[seq][frm][key]], axis=0)
    
    return dict_to_sequence_list(combine_data)


def convert_frame_format(track_data):
    """
    Function:
        convert track data from seq->obj_id dict format into frame-level list format
    Args:
        track_data: dict of track data {seq_n: {track_id: {xxxx}}}
    Returns:
        frame_res_list: list of frame-level result
    """
    frame_res_list = list()
    order_map = defaultdict(list)

    for tk_id, tk_info in track_data.items():
        sample_idx = tk_info['sample_idx']
        for i, sa_idx in enumerate(sample_idx):
            order_map[sa_idx].append([tk_id, i])

    frames = list(order_map.keys())
    for frm_id in frames:
        map_temp = np.stack(order_map[frm_id])
        obj_ids, orders = map_temp[:, 0], map_temp[:, 1]

        seq = track_data[obj_ids[0]]['sequence_name']
        pose = track_data[obj_ids[0]]['pose'][orders[0]]
        obj_num = len(obj_ids)

        boxes_lidar = np.zeros((obj_num, 7), dtype=np.float32)
        boxes_global = np.zeros((obj_num, 9), dtype=np.float32)
        score = np.zeros((obj_num), dtype=np.float32)
        name = np.full(obj_num, 'none', dtype=object)
        
        for i, obj_id in enumerate(obj_ids):
            idx = orders[i]
            
            if 'boxes_lidar' in track_data[obj_id]:
                boxes_lidar[i] = track_data[obj_id]['boxes_lidar'][idx]
            elif 'boxes_global' in track_data[obj_id]:
                boxes_global[i] = track_data[obj_id]['boxes_global'][idx]
                boxes_lidar[i] = transform_boxes3d(
                    boxes_global[i], pose, inverse=True).reshape(-1)
            
            score[i] = track_data[obj_id]['score'][idx]
            name[i] = track_data[obj_id]['name'][idx]
        
        frame_res_list.append({
            'sequence_name': seq,
            'frame_id': int(frm_id),
            'obj_ids': obj_ids,
            'name': name,
            'score': score,
            'boxes_lidar': boxes_lidar,
            'boxes_global': boxes_global,
            'pose': pose
        })
    
    return frame_res_list


def combine_final(prm_file, grm_file, save_path, logger):
    combine_dict = defaultdict(dict)

    geo_res = load_pkl(grm_file)
    pos_res = load_pkl(prm_file)

    if geo_res is None or pos_res is None:
        return
        
    obj_ids = pos_res.keys()
    for obj in obj_ids:
        boxes_geo = np.concatenate(geo_res[obj]['boxes_lidar'], axis=0)

        train_yaw = False
        train_coords = True  # pos_res[obj]['boxes_lidar'] = [x, y, z, l, w, h, yaw]
        train_grm = False
        if boxes_geo.shape[0] > 200:
            if train_grm:
                pos_res[obj]['boxes_lidar'][:, 0:7] = boxes_geo[:200, 0:7]
            else:
                if train_coords:
                    pos_res[obj]['boxes_lidar'][:, 3:6] = boxes_geo[:200, 3:6]
                if not train_yaw:
                    pos_res[obj]['boxes_lidar'][:, 6:7] = boxes_geo[:200, 6:7]
            
        else:
            if train_grm:
                pos_res[obj]['boxes_lidar'][:, 0:7] = boxes_geo[:, 0:7]
            else:
                if train_coords:
                    pos_res[obj]['boxes_lidar'][:, 3:6] = boxes_geo[:, 3:6]
                if not train_yaw:
                    pos_res[obj]['boxes_lidar'][:, 6:7] = boxes_geo[:, 6:7]

        pos_res[obj]['sample_idx'] = \
            np.array([str(x) for x in pos_res[obj]['frame_id']])
        
        combine_dict[obj] = pos_res[obj]

    save_pkl(combine_dict, save_path)
    logger.info('Track level final result is saved at %s' % save_path)


if __name__ == '__main__':

    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    logger = create_logger()
    save_path = os.path.join(ROOT_DIR, 'refining/output/ref_model_cfgs/train_fusion_0924_3w5_res_truck_trainxyz')

    os.makedirs(save_path, exist_ok=True)
    prm_root_path = os.path.join(ROOT_DIR, "refining/output/ref_model_cfgs/truck_prm_model/val_fusion_0924_3w5")
    grm_root_path = os.path.join(ROOT_DIR, "refining/output/ref_model_cfgs/truck_grm_model/val_fusion_0924_3w5")
    scene_lists = os.listdir(grm_root_path)
    map_dict = {}
    for idx, scene in enumerate(scene_lists):
        if scene.endswith(".txt"):
            continue
        
        prm_path = os.path.join(prm_root_path, scene, "eval/epoch_50/val_zrz/Truck_position_val_zrz.pkl")
        grm_path = os.path.join(grm_root_path, scene, "eval/epoch_30/val_zrz/Truck_geometry_val_zrz.pkl")
        if os.path.exists(prm_path):   # 确认 grm 下也有对应 scene
            map_dict[scene] = {
                "prm": prm_path,
                "grm": grm_path,
            }
        else:
            # map_dict[scene] = {
            #     "prm": None,
            #     "grm": grm_path,
            # }
            print(prm_path + " not exists!")
    for scene, files in map_dict.items():
        combine_final(
            prm_file=files["prm"],
            grm_file=files["grm"],
            save_path=os.path.join(save_path, scene+".pkl"),
            logger=logger,
        )
    
