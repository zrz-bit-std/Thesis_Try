import argparse
import time
import torch
import argparse
import os
import numpy as np
from pathlib import Path
from easydict import EasyDict
from detzero_utils.config_utils import cfg, cfg_from_yaml_file, log_cfg_info
from detzero_utils.common_utils import create_logger, get_log_info
import sys
from detzero_track.models import build_model, run_model
from detzero_track.datasets import build_dataloader

def get_tracking_res(cfg_file, data_root_path):
    data_path = os.path.join(data_root_path,'track.pkl')
    split = 'test'
    workers = 1
    batch_size = 8
    save_log = False
    cfg_tmp = EasyDict()
    cfg_from_yaml_file(cfg_file, cfg_tmp)
    cfg_tmp.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    log_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    if save_log:
        log_file = cfg_tmp.ROOT_DIR/'log'
        if not log_file.is_dir():
            log_file.mkdir(parents=True, exist_ok=True)
        log_file = log_file/(__file__.split('/')[-1]+'-'+log_time+'.txt')
    else:
        log_file = None

    logger = create_logger(log_file)
    logger.info(get_log_info('DetZero Tracking Module'))

    cfg_str_list = list()
    log_cfg_info(cfg_tmp, cfg_str_list, logger)
    cfg_tmp.DATA_CONFIG['DATA_PATH'] = data_root_path
    dataset, dataloader = build_dataloader(
        dataset_cfg=cfg_tmp.DATA_CONFIG,
        data_path=data_path,
        log_time=log_time,
        batch_size=batch_size,
        workers=workers,
        split=split,
        logger=logger
    )
    cfg_tmp.update({'cls_num':7})
    model = build_model(cfg_tmp.MODEL, logger)
    print("model:{}".format(model))
    run_model(
        model=model, 
        dataloader=dataloader,
        dataset=dataset,
        workers=workers,
        cfgs=cfg,
        logger=logger,
    )

    logger.info(get_log_info('DetZero Tracking module Finished!'))


def get_tracking_all(track_dir, sub_t, cfg_file):
    sub_track_dir = os.path.join(track_dir,sub_t)
    clip_list = os.listdir(sub_track_dir)
    for clip_name in clip_list:
        # try:
        clip_track_dir = os.path.join(sub_track_dir,clip_name)
        get_tracking_res(cfg_file, clip_track_dir)
        # except:
        #     print('*'*100,clip_track_dir)
        #     continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--cfg_file", type=str, default="")
    args = parser.parse_args()

    get_tracking_all(args.track_dir, args.dataset_name, args.cfg_file)

    # cfg_file = "/data1/turbo_data/wangruihao/code/4D_label/tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track_mogo_241216.yaml"
    # track_dir  = '/data1/turbo_data/4D_label_dataset/models_res/offline_tracked'
    # get_tracking_all('train_hy_4d_road_7_20250209_lx')

