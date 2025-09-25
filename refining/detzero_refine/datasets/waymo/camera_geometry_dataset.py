import copy
import random
from pathlib import Path

import numpy as np

from detzero_utils import common_utils

from detzero_refine.datasets.dataset import DatasetTemplate
from detzero_refine.utils.data_utils import sample_points, local_coords_transform
from detzero_refine.utils.geometry_augment import (augment_full_track,
                                                   augment_single_box,
                                                   test_time_augment)


class CameraGeometryDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, test_seq=None, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)

        self.query_num = self.dataset_cfg.get('QUERY_NUM', 3)
        self.query_pts_num = self.dataset_cfg.get('QUERY_POINTS_NUM', 256)
        self.memory_pts_num = self.dataset_cfg.get('MEMORY_POINTS_NUM', 4096)
        self.test_seq = test_seq
        self.logger = logger
        self.init_infos()

    def extract_track_feature(self, data_info):
        traj_all = data_info['boxes_global']
        
        score_all = data_info['score'].reshape(-1, 1)
        frame_id_all = data_info['sample_idx'].reshape(-1, 1)
        pose_all = data_info['pose']
        pts_all = data_info['pts']
        lenlen = len(traj_all)
        matched = data_info['matched'][:lenlen].astype(np.bool)
        mth_tk = data_info['matched_tracklet'][:lenlen].astype(np.bool)
        state = data_info['state']
        
        traj_gt_all = data_info['gt_boxes_global']

        # randomly sample the object track to increase data numbers
        if self.training:
            traj_len = len(traj_all)-1
            samples = random.sample(
                range(traj_len),
                random.randint(min(5, traj_len), traj_len)
            )
            
            score = score_all[samples]
            pose = pose_all[samples]
            frm_id = frame_id_all[samples]
            traj = traj_all[samples]
            traj_gt = traj_gt_all[samples]

            pts_mth = [pts_all[i] for i in range(len(traj_all))]
            pts = [pts_mth[ind] for ind in samples]
        
        else:
            pts = pts_all
            traj = traj_all
            traj_gt = traj_gt_all
            pose = pose_all
            frm_id = frame_id_all
            score = score_all

        # randomly sample the query proposals
        if self.training:
            if self.query_num > len(traj):
                self.query_num = len(traj)
            query_idx = np.random.choice(len(traj), self.query_num, replace=False)
        else:
            # for inference, we select the query based on the confidence score
            query_idx = np.argsort(score)[::-1][:self.query_num]
            self.query_num = len(query_idx)

            # query_idx = np.random.choice(len(traj), len(traj), replace=False)
        
        
        # transform the object points to the corresponding box center frame
        # pts = local_coords_transform(pts, traj)

        query_pts = [pts[ind].copy() for ind in query_idx]
        query_box = np.array([traj[ind, :].copy() for ind in query_idx])
        gt_box = np.array([traj_gt[ind, :].copy() for ind in query_idx])

        if not self.training and len(query_box.shape)==3 :
            query_box = query_box.squeeze(1)
            gt_box = gt_box.squeeze(1)
        # after all the transform, set box center and heading to 0
        query_box[:, [0, 1, 2, 6]] = 0
        gt_box[:, [0, 1, 2, 6]] = 0

        
        pts = np.stack(pts, axis=0).reshape(-1, 256)

        # sample points for each proposal as the value feature
        pts = sample_points(pts, sample_num=self.memory_pts_num)
        # breakpoint()
        for ind in range(self.query_num):
            query_pts[ind] = sample_points(query_pts[ind].reshape(1, -1), sample_num=self.query_pts_num)
           
        obj_info = {
            'sequence_name': data_info['sequence_name'],
            'frame': frm_id,
            'obj_id': data_info['obj_id'],
            'obj_cls': self.class_map[data_info['name']],
            # 'obj_cls': data_info['name'],
            'geo_query_num': self.query_num,
            'geo_query_boxes': query_box,
            'geo_query_points': query_pts,
            'geo_memory_points': pts,
            'geo_trajectory': traj,
            'geo_score': score,
            'gt_geo_query_boxes': gt_box,
            'gt_geo_trajectory': traj_gt,
            'pose': pose,
            'state': state,
            'matched': matched,
            'matched_tracklet': mth_tk
        }
        return obj_info

    @staticmethod
    def tta_operator(data_dict):
        return test_time_augment(data_dict)

    @staticmethod
    def revert_to_each_frame(data_dict):
        res_list = []

        for i, pred_box in enumerate(data_dict['pred_boxes']):
            traj = data_dict['geo_trajectory'][i]
            box_preds_world = copy.deepcopy(traj)
            box_preds_world[:, 3:6] = pred_box[3:6][None, :].repeat(len(traj), axis=0)
            pose = data_dict['pose'][i]
            box_preds_lidar_per_frm = []
            for ind in range(len(pose)):
                pose_inv = np.asarray(pose[ind], dtype=np.float32)
                r_t = np.linalg.inv(pose_inv)
                center = box_preds_world[[ind], :3]
                center = np.concatenate([center, np.ones((center.shape[0], 1))], axis=-1)
                center = center @ r_t.T
                heading = box_preds_world[[ind], [6]] + np.arctan2(r_t[1, 0], r_t[0, 0])
                
                box_lidar = np.concatenate([
                    center[:, :3],
                    box_preds_world[[ind], 3:6],
                    heading[None, :]], axis=-1)
                box_preds_lidar_per_frm.append(box_lidar)

            res_list.append(np.array(box_preds_lidar_per_frm))

        return res_list


    def generate_prediction_dicts(self, batch_dict, pred_dicts, single_pred_dict, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            single_pred_dict: dict, to save the result back

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_boxes = box_dict['pred_boxes']
            pred_dict = get_template_prediction(pred_boxes.shape[0])
            if pred_boxes.shape[0] == 0:
                print('-------------------ERROR--------------------')
                return pred_dict

            boxes_lidar = self.revert_to_each_frame(box_dict)

            pred_dict['boxes_lidar'] = boxes_lidar

            return boxes_lidar

        annos = []
        all_pred_res = generate_single_sample_dict(pred_dicts)
        for i in range(len(all_pred_res)):
            seq = batch_dict['sequence_name'][i]
            obj_id = batch_dict['obj_id'][i]
            
            # init the object result dict
            if seq not in single_pred_dict:
                single_pred_dict[seq] = {}
            single_pred_dict[seq][obj_id] = {
                'sequence_name': seq,
                'frame_id': [],
                'boxes_lidar': [],
                'score': [],
                'name': [],
                'pose': []
            }

            for idx, frm_id in enumerate(batch_dict['frame'][i]):
                single_pred_dict[seq][obj_id]['frame_id'].append(int(frm_id))
                single_pred_dict[seq][obj_id]['boxes_lidar'].append(all_pred_res[i][idx])
                single_pred_dict[seq][obj_id]['score'].append(batch_dict['geo_score'][i][idx])
                single_pred_dict[seq][obj_id]['name'].append(batch_dict['obj_cls'][i])
                single_pred_dict[seq][obj_id]['pose'].append(pred_dicts['pose'][i][idx])
        return single_pred_dict
        return annos
