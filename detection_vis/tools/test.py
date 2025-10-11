"""Test script for vision detection module."""

import os
import sys
import glob
import time
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent / '../../'))

from detzero_utils import common_utils
from detzero_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from detzero_utils.model_utils import load_params_from_file

from detzero_det_vis.datasets import build_dataloader
from detzero_det_vis.models import build_network
import eval_utils


def parse_config():
    parser = argparse.ArgumentParser(description='Vision detection testing')
    parser.add_argument('--cfg_file', type=str, default=None, help='config file')
    
    parser.add_argument('--batch_size', type=int, default=None, required=False)
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--extra_tag', type=str, default='default')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--max_waiting_mins', type=int, default=30)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--eval_all', action='store_true', default=False)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--save_to_file', action='store_true', default=False)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    
    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    """Evaluate a single checkpoint."""
    # Start evaluation
    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    
    model.eval()
    
    if cfg.LOCAL_RANK == 0:
        progress_bar = common_utils.ProgressBar(len(test_loader), logger)
    
    num_samples = len(test_loader.dataset)
    num_sample_per_epoch = test_loader.sampler.__len__()
    
    start_time = time.time()
    
    det_annos = []
    
    for i, batch_dict in enumerate(test_loader):
        with torch.no_grad():
            batch_dict = common_utils.move_to_cuda(batch_dict)
            pred_dicts, ret_dict = model(batch_dict)
            
        annos = test_loader.dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, test_loader.dataset.class_names,
            output_path=eval_output_dir if args.save_to_file else None
        )
        det_annos += annos
        
        if cfg.LOCAL_RANK == 0:
            progress_bar.update()
    
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()
    
    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, num_samples, tmpdir=eval_output_dir / 'tmpdir')
    
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(det_annos)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
    
    if cfg.LOCAL_RANK != 0:
        return {}
    
    ret_dict = {}
    if args.save_to_file:
        with open(eval_output_dir / 'result.pkl', 'wb') as f:
            import pickle
            pickle.dump(det_annos, f)
    
    result_str, result_dict = test_loader.dataset.evaluation(
        det_annos, test_loader.dataset.class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=eval_output_dir
    )
    
    logger.info(result_str)
    ret_dict.update(result_dict)
    
    logger.info('Result is saved to %s' % eval_output_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    """Evaluate checkpoints repeatedly or single checkpoint."""
    # Evaluate a single checkpoint if ckpt is specified
    if args.ckpt is not None:
        load_params_from_file(model, args.ckpt, logger=logger)
        eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, 
                        epoch_id='checkpoint', dist_test=dist_test)
        return
    
    # Evaluate all checkpoints
    if args.eval_all:
        ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
        ckpt_list.sort(key=os.path.getmtime)
        
        evaluated_ckpt_list = []
        for cur_ckpt in ckpt_list:
            num_list = re.findall('checkpoint_epoch_(\\d+)', cur_ckpt)
            if num_list.__len__() == 0:
                continue
            epoch_id = num_list[-1]
            
            if 'optim' in cur_ckpt:
                continue
            if epoch_id in evaluated_ckpt_list:
                continue
            
            evaluated_ckpt_list.append(epoch_id)
            
            logger.info('*************** Loading checkpoint from %s *****************' % cur_ckpt)
            load_params_from_file(model, cur_ckpt, logger=logger)
            
            cur_eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id)
            cur_eval_output_dir.mkdir(parents=True, exist_ok=True)
            
            eval_single_ckpt(model, test_loader, args, cur_eval_output_dir, logger,
                           epoch_id=epoch_id, dist_test=dist_test)
        return
    
    # Wait and evaluate latest checkpoint
    start_epoch = args.start_epoch
    for cur_epoch in range(start_epoch, 100):
        ckpt_name = ckpt_dir / ('checkpoint_epoch_%d.pth' % cur_epoch)
        
        wait_second = 0
        while not ckpt_name.exists() and wait_second < args.max_waiting_mins * 60:
            time.sleep(10)
            wait_second += 10
        
        if not ckpt_name.exists():
            logger.info('*************** No more checkpoint found *****************')
            break
        
        logger.info('*************** Loading checkpoint from %s *****************' % ckpt_name)
        load_params_from_file(model, str(ckpt_name), logger=logger)
        
        cur_eval_output_dir = eval_output_dir / ('epoch_%d' % cur_epoch)
        cur_eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        eval_single_ckpt(model, test_loader, args, cur_eval_output_dir, logger,
                        epoch_id=cur_epoch, dist_test=dist_test)


def main():
    args, cfg = parse_config()
    
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'
    
    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'
    
    if args.ckpt_dir is not None:
        eval_output_dir = eval_output_dir / args.ckpt_dir
    
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # Log to console
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.cuda()

    # Start evaluation
    repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)


if __name__ == '__main__':
    import re
    main()

