"""Extract BEV features using BEVFormer (vision detection head).

This script loads the vision detection dataset and BEVFormer model, runs
inference to obtain BEV embeddings (bev_embed) per sample, and saves them
to disk for downstream usage.
"""

import os
import sys
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent / '../../'))

from detzero_utils import common_utils
from detzero_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from detzero_utils.model_utils import load_params_from_file

from detzero_det_vis.datasets import build_dataloader
from detzero_det_vis.models import build_network
import torchvision.models as tv_models
import torch.nn.functional as F


def parse_config():
    parser = argparse.ArgumentParser(description='Extract BEV features (vision)')
    parser.add_argument('--cfg_file', type=str, required=True, help='config file')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path (optional)')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save features')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=-1, help='limit number of samples; -1 for all')
    parser.add_argument('--feat_type', type=str, choices=['bev', 'image', 'image_torchvision'], default='image',
                        help='feature type to extract: bev or image (backbone+fpn)')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='override config, e.g. DATA_CONFIG.DATA_PATH /abs/path')

    args = parser.parse_args()

    # Ensure relative _BASE_CONFIG_ paths resolve from tools directory
    tools_dir = Path(__file__).resolve().parent
    os.chdir(str(tools_dir))
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(1024)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def save_bev_tensor(bev_tensor: torch.Tensor, meta: dict, base_dir: Path):
    """Save BEV tensor for a single sample.

    bev_tensor: [C, H, W] or [B, C, H, W] (handled per-sample)
    meta: dict containing identifiers such as sequence_name and frame_id
    base_dir: output base directory
    """
    sequence = str(meta.get('sequence_name', 'unknown'))
    frame_id = str(meta.get('frame_id', meta.get('sample_idx', '0')))

    out_dir = base_dir / sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{frame_id}_bev.pt"
    torch.save(bev_tensor.cpu(), out_path)


def save_image_feats(feat_list, meta: dict, base_dir: Path, cams: int):
    """Save multi-scale image features per camera.

    feat_list: list of tensors at multiple scales, each [BN, C, H, W]
    cams: number of cameras (N)
    """
    sequence = str(meta.get('sequence_name', 'unknown'))
    frame_id = str(meta.get('frame_id', meta.get('sample_idx', '0')))
    out_dir = base_dir / sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split BN back to [B, N]
    B = 1
    for scale_idx, feat in enumerate(feat_list):
        BN, C, H, W = feat.shape
        assert BN % cams == 0
        B = BN // cams
        feat = feat.view(B, cams, C, H, W)
        # save each camera tensor
        for cam_idx in range(cams):
            out_path = out_dir / f"{frame_id}_cam{cam_idx}_s{scale_idx}.pt"
            torch.save(feat[0, cam_idx].cpu(), out_path)


def main():
    args, _cfg = parse_config()

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_extract_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)

    logger.info('**********************Start logging (Feature Extraction)**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # Dataloader
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

    test_set, test_loader, _sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )

    # Fast path: image features via backbone+neck only
    if args.feat_type in ['image', 'image_torchvision']:
        # torchvision-based ResNet50 + simple FPN (P3-P6)
        try:
            resnet = tv_models.resnet50(weights=None).cuda().eval()
        except TypeError:
            resnet = tv_models.resnet50(pretrained=False).cuda().eval()

        # Stages
        stem = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool).cuda().eval()
        layer1 = resnet.layer1.cuda().eval()  # C2: 256
        layer2 = resnet.layer2.cuda().eval()  # C3: 512
        layer3 = resnet.layer3.cuda().eval()  # C4: 1024
        layer4 = resnet.layer4.cuda().eval()  # C5: 2048

        # FPN 256d
        c2_ch, c3_ch, c4_ch, c5_ch = 256, 512, 1024, 2048
        p5_1x1 = torch.nn.Conv2d(c5_ch, 256, 1).cuda().eval()
        p4_1x1 = torch.nn.Conv2d(c4_ch, 256, 1).cuda().eval()
        p3_1x1 = torch.nn.Conv2d(c3_ch, 256, 1).cuda().eval()
        p2_1x1 = torch.nn.Conv2d(c2_ch, 256, 1).cuda().eval()
        p5_3x3 = torch.nn.Conv2d(256, 256, 3, padding=1).cuda().eval()
        p4_3x3 = torch.nn.Conv2d(256, 256, 3, padding=1).cuda().eval()
        p3_3x3 = torch.nn.Conv2d(256, 256, 3, padding=1).cuda().eval()
        p2_3x3 = torch.nn.Conv2d(256, 256, 3, padding=1).cuda().eval()

        processed = 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(test_loader):
                imgs = batch_dict['img'].cuda(non_blocking=True)  # [B,N,C,H,W]
                B, N, C, H, W = imgs.shape
                x = imgs.view(B * N, C, H, W)

                c1 = stem(x)
                c2 = layer1(c1)
                c3 = layer2(c2)
                c4 = layer3(c3)
                c5 = layer4(c4)

                # top-down pathway
                p5 = p5_1x1(c5)
                p4 = p4_1x1(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
                p3 = p3_1x1(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
                p2 = p2_1x1(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

                # smoothing conv
                p5 = p5_3x3(p5)
                p4 = p4_3x3(p4)
                p3 = p3_3x3(p3)
                p2 = p2_3x3(p2)

                feats = [p2, p3, p4, p5]

                meta = batch_dict['img_metas'][0]
                save_image_feats(feats, meta, output_dir, cams=N)

                processed += 1
                if args.max_samples > 0 and processed >= args.max_samples:
                    logger.info(f"Reached max_samples={args.max_samples}, stopping.")
                    break

        logger.info(f"Done. Image features saved to: {output_dir}")
        return

    # Else: BEV features via full BEVFormer
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.cuda().eval()
    if args.ckpt is not None and len(str(args.ckpt)) > 0:
        load_params_from_file(model, args.ckpt, logger=logger)
    else:
        logger.warning('No checkpoint provided. Proceeding without loading weights (random-initialized features).')

    # Iterate and extract
    processed = 0
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(test_loader):
            batch_size = len(batch_dict['img_metas'])

            # Move images to GPU; metas remain on CPU as dicts
            batch_dict['img'] = batch_dict['img'].cuda(non_blocking=True)

            # Call underlying BEVFormer to fetch BEV features
            # simple_test returns (new_prev_bev, bbox_list)
            try:
                new_prev_bev, _ = model.bevformer.simple_test(
                    img_metas=batch_dict['img_metas'],
                    img=batch_dict['img']
                )
            except Exception as e:
                logger.error(f"BEV extraction failed at batch {batch_idx}: {e}")
                logger.error("Tip: ensure valid calibration in img_metas (lidar2img/cam2img) and correct num_feature_levels.")
                break

            # new_prev_bev is either [B, C, H, W] or list; ensure tensor
            if isinstance(new_prev_bev, (list, tuple)):
                bev_batch = new_prev_bev[0]
            else:
                bev_batch = new_prev_bev

            # Save per-sample
            for i in range(batch_size):
                meta = batch_dict['img_metas'][i]
                # If bev_batch is [B, C, H, W]
                if isinstance(bev_batch, torch.Tensor) and bev_batch.dim() == 4:
                    bev_tensor = bev_batch[i]
                else:
                    bev_tensor = bev_batch if isinstance(bev_batch, torch.Tensor) else torch.as_tensor(bev_batch)
                save_bev_tensor(bev_tensor, meta, output_dir)
                processed += 1

                if args.max_samples > 0 and processed >= args.max_samples:
                    logger.info(f"Reached max_samples={args.max_samples}, stopping.")
                    return

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {processed} samples...")

    logger.info(f"Done. Total saved samples: {processed}. Features at: {output_dir}")


if __name__ == '__main__':
    main()


