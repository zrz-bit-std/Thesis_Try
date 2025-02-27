import time
import torch
import argparse

import numpy as np
from pathlib import Path

from detzero_utils.config_utils import cfg, cfg_from_yaml_file, log_cfg_info
from detzero_utils.common_utils import create_logger, get_log_info
import sys
sys.path.append('/data1/turbo_data/wangruihao/code/4D_label/tracking')
from detzero_track.models import build_model, run_model
from detzero_track.datasets import build_dataloader


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="/data1/turbo_data/wangruihao/code/4D_label/tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track_mogo_241216.yaml", help='load the config for tracking')
    parser.add_argument('--data_path', type=str,default = '/data1/turbo_data/4D_label_dataset/models_res/offline_tracked/train_hy_4d_road_7_20250208_lx/20250208070028/track.pkl', help='data path of detection results')
    parser.add_argument('--workers', type=int, default=1, help='num of cpu for running track module')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('--split', type=str, default='test', help='different data split') 
    parser.add_argument('--save_log', action="store_true", help="save log info into .txt")
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    return args, cfg

def main():
    args, cfg = parse_config()
    log_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    if args.save_log:
        log_file = cfg.ROOT_DIR/'log'
        if not log_file.is_dir():
            log_file.mkdir(parents=True, exist_ok=True)
        log_file = log_file/(__file__.split('/')[-1]+'-'+log_time+'.txt')
    else:
        log_file = None

    logger = create_logger(log_file)
    logger.info(get_log_info('DetZero Tracking Module'))

    cfg_str_list = list()
    for key, val in vars(args).items():
        cfg[key.upper()] = val

    log_cfg_info(cfg, cfg_str_list, logger)

    # cfg.DATA_CONFIG['DATA_PATH'] = '/data1/turbo_data/4D_label_dataset/models_res/offline_tracked/train_hy_4d_road_7_20250208_lx/20250208070028'
    dataset, dataloader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        data_path=args.data_path,
        log_time=log_time,
        batch_size=args.batch_size,
        workers=args.workers,
        split=args.split,
        logger=logger
    )
    cfg.update({'cls_num':7})
    model = build_model(cfg.MODEL, logger)
    print("model:{}".format(model))
    run_model(
        model=model, 
        dataloader=dataloader,
        dataset=dataset,
        workers=args.workers,
        cfgs=cfg,
        logger=logger,
    )

    logger.info(get_log_info('DetZero Tracking module Finished!'))

if __name__ == '__main__':
    main()
    '''
    /home/mogo/data/dev/DetZero/tracking/tools/run_track.py --cfg_file /home/mogo/data/dev/DetZero/tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track_mogo_241216.yaml --data_path /home/mogo/data/dev/DetZero/tracking/track_input_20250120.pkl --workers 1 --split test 
    '''

