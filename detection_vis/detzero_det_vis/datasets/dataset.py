"""Base dataset template for vision detection."""

import copy
import pickle
from abc import abstractmethod
from collections import defaultdict

import numpy as np
import torch

from detzero_utils import box_utils, common_utils


class DatasetTemplate(torch.utils.data.Dataset):
    """
    The base class of datasets for vision detection.
    """
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.class_names = class_names
        self.training = training
        self.root_path = root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.logger = logger

        self.tta = False if self.training else getattr(self.dataset_cfg, "TTA", False)
        self.queue_length = self.dataset_cfg.get('QUEUE_LENGTH', 1)
        self.sampled_interval = self.dataset_cfg.SAMPLED_INTERVAL[self.mode]\
            if 'SAMPLED_INTERVAL' in self.dataset_cfg else None

        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __len__(self):
        return len(self.infos)

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        """Collate batch for dataloader."""
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['img', 'img_queue']:
                    # Stack images: [B, N, C, H, W] or [B, queue_len, N, C, H, W]
                    ret[key] = torch.stack(val, dim=0)
                elif key in ['gt_bboxes_3d', 'gt_labels_3d']:
                    # Keep as list
                    ret[key] = val
                elif key == 'img_metas':
                    # Keep as list
                    ret[key] = val
                elif isinstance(val[0], np.ndarray):
                    if key in ['voxels', 'voxel_num_points']:
                        ret[key] = np.concatenate(val, axis=0)
                    else:
                        ret[key] = np.stack(val, axis=0)
                else:
                    ret[key] = val
            except:
                print(f'Error in collate_batch: key={key}')
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

