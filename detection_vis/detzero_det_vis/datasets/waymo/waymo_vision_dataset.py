"""Waymo vision dataset for BEVFormer-based detection."""

import os
import sys
import pickle
import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Use local projects package; no external BevFormer path injection

from detzero_utils import common_utils
from ..dataset import DatasetTemplate


class WaymoVisionDataset(DatasetTemplate):
    """
    Waymo Dataset for Vision-based Detection using BEVFormer.
    Uses multi-view camera images for 3D object detection.
    """

    def __init__(self, dataset_cfg, class_names, root_path, training=True, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        
        self.data_path = self.root_path
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        
        # Load split file
        split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        
        # Camera configuration for Waymo
        self.camera_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
        self.num_cameras = len(self.camera_names)
        
        # Image normalization
        self.img_norm_cfg = dataset_cfg.get('IMG_NORM_CFG', dict(
            mean=[103.530, 116.280, 123.675], 
            std=[1.0, 1.0, 1.0], 
            to_rgb=False
        ))
        
        self.init_infos()

    def init_infos(self):
        """Initialize dataset infos."""
        self.infos = []
        
        # Load or create info file
        info_path = os.path.join(
            self.data_path, 
            f'waymo_infos_{self.split}_vision.pkl'
        )
        
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                self.infos = pickle.load(f)
            if self.logger:
                self.logger.info(f'Loaded {len(self.infos)} samples from {info_path}')
        else:
            # Create info from sample list
            for sample_idx, sample_name in enumerate(self.sample_sequence_list):
                info = {
                    'sample_idx': sample_idx,
                    'sample_name': sample_name,
                    'sequence_name': sample_name.split('_')[0],
                    'frame_id': sample_name.split('_')[1] if '_' in sample_name else '0',
                }
                self.infos.append(info)
            
            if self.logger:
                self.logger.info(f'Created {len(self.infos)} samples for {self.split}')

    def get_image_paths(self, info):
        """Get image paths for all cameras."""
        sequence_name = info['sequence_name']
        frame_id = info['frame_id']
        
        img_paths = []
        for cam_name in self.camera_names:
            img_path = os.path.join(
                self.data_path,
                'images',
                sequence_name,
                f'{cam_name}_{frame_id}.jpg'
            )
            img_paths.append(img_path)
        return img_paths

    def load_images(self, img_paths):
        """Load and normalize images."""
        imgs = []
        for img_path in img_paths:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = np.array(img, dtype=np.float32)
            else:
                # Create dummy image if not exists
                img = np.zeros((1280, 1920, 3), dtype=np.float32)
            
            # Normalize
            img = self.normalize_image(img)
            imgs.append(img)
        
        # Stack to [N, C, H, W]
        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        return imgs

    def normalize_image(self, img):
        """Normalize image."""
        mean = np.array(self.img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(self.img_norm_cfg['std'], dtype=np.float32)
        
        if self.img_norm_cfg.get('to_rgb', False):
            img = img[..., ::-1]  # BGR to RGB
        
        img = (img - mean) / std
        return img

    def get_annotations(self, info):
        """Get 3D box annotations."""
        # Load from annotation file
        sequence_name = info['sequence_name']
        frame_id = info['frame_id']
        
        anno_path = os.path.join(
            self.data_path,
            'annotations',
            sequence_name,
            f'{frame_id}.pkl'
        )
        
        if os.path.exists(anno_path):
            with open(anno_path, 'rb') as f:
                annos = pickle.load(f)
            return annos
        else:
            # Return empty annotations
            return {
                'gt_boxes': np.zeros((0, 7), dtype=np.float32),
                'gt_names': np.array([], dtype=str),
                'gt_labels': np.array([], dtype=np.int64),
            }

    def __getitem__(self, index):
        """Get item."""
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        
        # Load images
        img_paths = self.get_image_paths(info)
        imgs = self.load_images(img_paths)
        
        # Prepare image metas
        img_metas = {
            'sample_idx': info['sample_idx'],
            'sample_name': info['sample_name'],
            'sequence_name': info['sequence_name'],
            'frame_id': info['frame_id'],
            'img_shape': [(1280, 1920, 3)] * self.num_cameras,
            'scene_token': info['sequence_name'],
            'prev_bev_exists': index > 0,  # Simplified
            'can_bus': np.zeros(18, dtype=np.float32),  # Placeholder
            # Provide identity lidar2img for each camera to satisfy BEVFormer pipeline
            'lidar2img': np.tile(np.eye(4, dtype=np.float32), (self.num_cameras, 1, 1))
        }
        
        data_dict = {
            'img': imgs,
            'img_metas': img_metas,
        }
        
        # Add GT for training
        if self.training:
            annos = self.get_annotations(info)
            data_dict['gt_bboxes_3d'] = annos.get('gt_boxes', np.zeros((0, 7), dtype=np.float32))
            data_dict['gt_labels_3d'] = annos.get('gt_labels', np.array([], dtype=np.int64))

        # Fill additional meta fields expected by BEVFormer
        # mmcv/mmdet3d expects img_metas to be list[dict] per sample
        # ensure required calibration placeholders to avoid key errors
        if 'cam2ego' not in img_metas:
            img_metas['cam2ego'] = np.tile(np.eye(4, dtype=np.float32), (self.num_cameras, 1, 1))
        if 'ego2lidar' not in img_metas:
            img_metas['ego2lidar'] = np.eye(4, dtype=np.float32)
        if 'lidar2ego' not in img_metas:
            img_metas['lidar2ego'] = np.eye(4, dtype=np.float32)
        if 'pts_filename' not in img_metas:
            img_metas['pts_filename'] = ''
        
        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        """Evaluation function."""
        # Use Waymo evaluation metrics
        from detzero_det.datasets.waymo.waymo_eval_detection import waymo_evaluation
        
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [info['annot'] for info in self.infos]
        
        ap_result_str, ap_dict = waymo_evaluation(
            eval_det_annos,
            eval_gt_annos,
            class_names,
            distance_thresh=100,
            fake_gt_infos=self.dataset_cfg.get('FAKE_GT_INFOS', False)
        )
        
        return ap_result_str, ap_dict

