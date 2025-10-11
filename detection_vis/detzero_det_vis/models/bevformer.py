"""
BEVFormer detector for DetZero vision detection module.
Adapted from BEVFormer (https://github.com/fundamentalvision/BEVFormer)
"""

import os
import sys

import torch
import torch.nn as nn
import numpy as np

from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer as BEVFormerBase
from mmdet3d.core import bbox3d2result


class BEVFormer(nn.Module):
    """BEVFormer wrapper for DetZero framework.
    
    This class wraps the original BEVFormer implementation to be compatible
    with DetZero's training and inference pipeline.
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        
        # Initialize BEVFormer model
        self.bevformer = self._build_bevformer()
        
    def _build_bevformer(self):
        """Build BEVFormer model from config using BEVFormerBase ctor."""
        BF = BEVFormerBase

        def _lower_keys(d):
            if not isinstance(d, dict):
                return d
            out = {}
            for k, v in d.items():
                nk = k.lower() if isinstance(k, str) else k
                out[nk] = _lower_keys(v) if isinstance(v, dict) else (
                    [_lower_keys(x) if isinstance(x, dict) else x for x in v] if isinstance(v, list) else v
                )
            return out

        def _normalize_norm_cfg(d):
            if not isinstance(d, dict):
                return d
            if 'norm_cfg' in d and isinstance(d['norm_cfg'], dict):
                t = d['norm_cfg'].get('type', None)
                if t == 'BN2d':
                    d['norm_cfg']['type'] = 'BN'
            return d

        def _convert_cfg(d):
            # recursively lower keys and fix known fields
            d = _lower_keys(d)
            d = _normalize_norm_cfg(d)
            # fix common ALLCAPS parameter names for mmcv components
            mapping = {
                'type': 'type',
                'bev_h': 'bev_h', 'bev_w': 'bev_w',
                'num_query': 'num_query', 'num_classes': 'num_classes',
                'in_channels': 'in_channels', 'out_channels': 'out_channels',
                'start_level': 'start_level', 'add_extra_convs': 'add_extra_convs',
                'num_outs': 'num_outs', 'relu_before_extra_convs': 'relu_before_extra_convs',
                'num_stages': 'num_stages', 'out_indices': 'out_indices', 'frozen_stages': 'frozen_stages',
                'sync_cls_avg_factor': 'sync_cls_avg_factor', 'with_box_refine': 'with_box_refine',
                'as_two_stage': 'as_two_stage',
                'embed_dims': 'embed_dims', 'rotate_prev_bev': 'rotate_prev_bev',
                'use_shift': 'use_shift', 'use_can_bus': 'use_can_bus',
                'pc_range': 'pc_range', 'num_layers': 'num_layers', 'return_intermediate': 'return_intermediate',
                'num_points_in_pillar': 'num_points_in_pillar',
                'post_center_range': 'post_center_range', 'max_num': 'max_num', 'voxel_size': 'voxel_size',
                'loss_weight': 'loss_weight', 'use_sigmoid': 'use_sigmoid', 'gamma': 'gamma', 'alpha': 'alpha'
            }
            # already lowered keys above; nothing else to map explicitly here
            # Just ensure nested dicts are processed
            for k, v in list(d.items()):
                if isinstance(v, dict):
                    d[k] = _convert_cfg(v)
                elif isinstance(v, list):
                    d[k] = [_convert_cfg(x) if isinstance(x, dict) else x for x in v]
            return d

        img_backbone_cfg = _convert_cfg(self.model_cfg.get('IMG_BACKBONE', {}))
        img_neck_cfg = _convert_cfg(self.model_cfg.get('IMG_NECK', {}))
        pts_bbox_head_cfg = _convert_cfg(self.model_cfg.get('PTS_BBOX_HEAD', {}))

        # Fill defaults for transformer encoder/decoder if missing
        transformer_cfg = pts_bbox_head_cfg.get('transformer', {})
        embed_dims = pts_bbox_head_cfg.get('in_channels', 256)
        num_feature_levels = img_neck_cfg.get('num_outs', 4)
        num_cams = getattr(self.dataset, 'num_cameras', 6)
        transformer_cfg.setdefault('type', 'PerceptionTransformer')
        transformer_cfg.setdefault('embed_dims', embed_dims)
        transformer_cfg.setdefault('num_feature_levels', num_feature_levels)
        transformer_cfg['num_cams'] = num_cams

        # Encoder defaults
        encoder_cfg = transformer_cfg.get('encoder', None)
        if encoder_cfg is None:
            encoder_cfg = {
                'type': 'BEVFormerEncoder',
                'num_layers': 6,
                'pc_range': self.dataset.dataset_cfg.get('POINT_CLOUD_RANGE', [-75, -75, -2, 75, 75, 4]),
                'num_points_in_pillar': 4,
                'return_intermediate': False,
                'transformerlayers': {
                    'type': 'BEVFormerLayer',
                    'attn_cfgs': [
                        {
                            'type': 'TemporalSelfAttention',
                            'embed_dims': embed_dims,
                        },
                        {
                            'type': 'SpatialCrossAttention',
                            'embed_dims': embed_dims,
                            'num_cams': num_cams,
                            'deformable_attention': {
                                'type': 'MSDeformableAttention3D',
                                'embed_dims': embed_dims,
                                'num_heads': 8,
                                'num_levels': num_feature_levels,
                                'num_points': 4
                            }
                        }
                    ],
                    'feedforward_channels': embed_dims * 2,
                    'ffn_dropout': 0.1,
                    'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                }
            }
        else:
            # ensure required inner keys
            encoder_cfg = _convert_cfg(encoder_cfg)
            encoder_cfg.setdefault('transformerlayers', {
                'type': 'BEVFormerLayer',
                'attn_cfgs': [
                    {
                        'type': 'TemporalSelfAttention',
                        'embed_dims': embed_dims,
                    },
                    {
                        'type': 'SpatialCrossAttention',
                        'embed_dims': embed_dims,
                        'num_cams': num_cams,
                        'deformable_attention': {
                            'type': 'MSDeformableAttention3D',
                            'embed_dims': embed_dims,
                            'num_heads': 8,
                            'num_levels': num_feature_levels,
                            'num_points': 4
                        }
                    }
                ],
                'feedforward_channels': embed_dims * 2,
                'ffn_dropout': 0.1,
                'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
            })

        # Decoder defaults
        decoder_cfg = transformer_cfg.get('decoder', None)
        if decoder_cfg is None:
            decoder_cfg = {
                'type': 'DetectionTransformerDecoder',
                'num_layers': 6,
                'return_intermediate': True,
                'transformerlayers': {
                    'type': 'DetrTransformerDecoderLayer',
                    'attn_cfgs': [
                        {
                            'type': 'MultiheadAttention',
                            'embed_dims': embed_dims,
                            'num_heads': 8,
                            'dropout': 0.1
                        },
                        {
                            'type': 'CustomMSDeformableAttention',
                            'embed_dims': embed_dims,
                            'num_heads': 8,
                            'num_levels': 1,
                            'num_points': 4
                        }
                    ],
                    'feedforward_channels': embed_dims * 2,
                    'ffn_dropout': 0.1,
                    'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                }
            }
        else:
            decoder_cfg = _convert_cfg(decoder_cfg)
            decoder_cfg.setdefault('transformerlayers', {
                'type': 'DetrTransformerDecoderLayer',
                'attn_cfgs': [
                    {
                        'type': 'MultiheadAttention',
                        'embed_dims': embed_dims,
                        'num_heads': 8,
                        'dropout': 0.1
                    },
                    {
                        'type': 'CustomMSDeformableAttention',
                        'embed_dims': embed_dims,
                        'num_heads': 8,
                        'num_levels': 1,
                        'num_points': 4
                    }
                ],
                'feedforward_channels': embed_dims * 2,
                'ffn_dropout': 0.1,
                'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
            })

        transformer_cfg['encoder'] = encoder_cfg
        transformer_cfg['decoder'] = decoder_cfg
        pts_bbox_head_cfg['transformer'] = transformer_cfg
        train_cfg = self.model_cfg.get('TRAIN_CFG', None)
        test_cfg = self.model_cfg.get('TEST_CFG', None)

        # uppercase TYPE to type at top-level of components
        for comp in (img_backbone_cfg, img_neck_cfg, pts_bbox_head_cfg):
            if 'TYPE' in comp:  # in case original dict leaked
                comp['type'] = comp.pop('TYPE')

        bevformer = BF(
            use_grid_mask=self.model_cfg.get('USE_GRID_MASK', True),
            video_test_mode=self.model_cfg.get('VIDEO_TEST_MODE', True),
            img_backbone=img_backbone_cfg,
            img_neck=img_neck_cfg,
            pts_bbox_head=pts_bbox_head_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        # Enforce correct num_cams to avoid shape mismatch
        try:
            bevformer.pts_bbox_head.transformer.num_cams = getattr(self.dataset, 'num_cameras', bevformer.pts_bbox_head.transformer.num_cams)
        except Exception:
            pass
        return bevformer
        
    def forward(self, batch_dict):
        """Forward pass.
        
        Args:
            batch_dict: Dictionary containing:
                - img: Image tensor [B, N, C, H, W]
                - img_metas: List of image metadata
                - gt_bboxes_3d: Ground truth 3D boxes (training only)
                - gt_labels_3d: Ground truth labels (training only)
                
        Returns:
            If training: (loss_dict, tb_dict, disp_dict)
            If testing: (pred_dicts, recall_dicts)
        """
        if self.training:
            return self._forward_train(batch_dict)
        else:
            return self._forward_test(batch_dict)
            
    def _forward_train(self, batch_dict):
        """Forward training."""
        # Extract from batch_dict
        img = batch_dict['img']
        img_metas = batch_dict['img_metas']
        gt_bboxes_3d = batch_dict.get('gt_bboxes_3d', None)
        gt_labels_3d = batch_dict.get('gt_labels_3d', None)
        
        # Adapt shapes to BEVFormer expectations
        if img.dim() == 5:
            # [B, N, C, H, W] -> [B, 1, N, C, H, W]
            img = img.unsqueeze(1)
        # img_metas: [B] list of dict -> [[dict]] per sample
        if isinstance(img_metas, list) and (len(img_metas) == img.shape[0]):
            img_metas_nested = [[m] for m in img_metas]
        else:
            img_metas_nested = img_metas

        # Forward through BEVFormer
        losses = self.bevformer.forward_train(
            img=img,
            img_metas=img_metas_nested,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d
        )
        
        # Convert to DetZero format
        total_loss = sum(losses.values())
        tb_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                   for k, v in losses.items()}
        disp_dict = {}
        
        ret_dict = {'loss': total_loss}
        return ret_dict, tb_dict, disp_dict
        
    def _forward_test(self, batch_dict):
        """Forward testing."""
        img = batch_dict['img']
        img_metas = batch_dict['img_metas']
        
        # Forward through BEVFormer: wrap for test-time API
        img_list = [img]
        img_metas_outer = [img_metas]  # [[dict]*B]
        bbox_results = self.bevformer.forward_test(
            img_metas=img_metas_outer,
            img=img_list
        )
        
        # Convert to DetZero format
        pred_dicts = []
        for result in bbox_results:
            pred_dict = {
                'pred_boxes': result['pts_bbox']['boxes_3d'].tensor.cpu().numpy(),
                'pred_scores': result['pts_bbox']['scores_3d'].cpu().numpy(),
                'pred_labels': result['pts_bbox']['labels_3d'].cpu().numpy(),
            }
            pred_dicts.append(pred_dict)
            
        recall_dicts = {}
        return pred_dicts, recall_dicts
        
    def load_params_from_file(self, filename, logger, to_cpu=False):
        """Load parameters from checkpoint file."""
        if not os.path.isfile(filename):
            raise FileNotFoundError
            
        logger.info(f'==> Loading parameters from checkpoint {filename} to GPU')
        checkpoint = torch.load(filename, map_location='cpu' if to_cpu else None)
        
        if 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
            
        logger.info(f'==> Done loading checkpoint')


def build_network(model_cfg, num_class, dataset):
    """Build BEVFormer network."""
    model = BEVFormer(
        model_cfg=model_cfg,
        num_class=num_class,
        dataset=dataset
    )
    return model


def model_fn_decorator():
    """Model function decorator for training."""
    def model_func(model, batch_dict):
        return model(batch_dict)
    return model_func

