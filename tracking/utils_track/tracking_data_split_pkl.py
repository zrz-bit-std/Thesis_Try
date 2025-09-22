import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import sys
import pickle
from tqdm import tqdm
import shutil
import argparse
import re


def remakedir(folder):
    #高频代码
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def combine_data_from_label(clip_merged_dir,track_pkl_path,txt_save_dir):
    frame_name_list = sorted([i.split('.t')[0] for i in os.listdir(clip_merged_dir)])
    remakedir(txt_save_dir)
    with open(track_pkl_path, 'rb') as file:
        res_track = pickle.load(file)#['20241216']
        res_track_ = res_track[list(res_track.keys())[0]]
    for frame_id_ in range(len(frame_name_list)):
        frame = frame_name_list[frame_id_]
        boxex_list_tracked = []
        for track_id in res_track_:
            sample_idx = res_track_[track_id]['sample_idx'] # 这是一条
            boxes_global = res_track_[track_id]['boxes_global']
            score = res_track_[track_id]['score']
            boxes_pose = res_track_[track_id]['pose']
            label_name = res_track_[track_id]['name']
            uuid_ = res_track_[track_id]['uuid']
            # boxes_global = smooth_boxes(boxes_global=boxes_global,score=score)
            if str(frame_id_) in sample_idx: 
                track_frame_idx = sample_idx.tolist().index(str(frame_id_))
                box_global_track_frame = boxes_global[track_frame_idx]
                box_pose_track_frame = boxes_pose[track_frame_idx]
                x,y,z,l,w,h,yaw,vx,vy = box_global_track_frame
                label = label_name[track_frame_idx]
                uuid = uuid_[track_frame_idx]
                s = score[track_frame_idx]
                # print(label)
                boxex_list_tracked.append([uuid,label,h,w,l,x,y,z,math.degrees(yaw),s])
        frame_save_txt_path = os.path.join(txt_save_dir,frame+'.txt')
        # with open(frame_save_txt_path)
        with open(frame_save_txt_path,'w') as f:
            for save_list_item in boxex_list_tracked:
                f.write('\t'.join([str(i) for i in save_list_item])+'\n')


def split_data(merged_dir, track_dir, sub_t, file_pattern='tracking-test'):
    sub_merged_dir = os.path.join(merged_dir,sub_t)
    sub_tracked_dir = os.path.join(track_dir,sub_t)
    clip_list = os.listdir(sub_tracked_dir)
    for clip_name in tqdm(clip_list):
        clip_merged_dir = os.path.join(sub_merged_dir,clip_name)
        clip_tracked_dir = os.path.join(sub_tracked_dir,clip_name,'tracking')
        clip_tracked_split_dir = os.path.join(sub_tracked_dir,clip_name,'splited')
        
        if file_pattern == 'tracking-test':
            # 查找以tracking-test-开头、后面跟日期时间戳的.pkl文件
            pattern = r'tracking-test-\d{8}-\d{6}\.pkl'
            target_files = [f for f in os.listdir(clip_tracked_dir) if re.match(pattern, f)]
            
            if not target_files:
                raise FileNotFoundError(f"No tracking-test-YYYYMMDD-HHMMSS.pkl file found in {clip_tracked_dir}")
            
            # 如果有多个匹配文件，选择最新的（按字母顺序最后一个）
            target_files.sort()
            target_file = target_files[-1]
            tracking_pkl_path = os.path.join(clip_tracked_dir, target_file)
            print(f"Processing {target_file}")
        elif file_pattern == 'BiTrack':
            # 处理BiTrack-final.pkl文件
            target_file = 'BiTrack-final.pkl'
            tracking_pkl_path = os.path.join(clip_tracked_dir, target_file)
            if not os.path.exists(tracking_pkl_path):
                raise FileNotFoundError(f"BiTrack-final.pkl not found in {clip_tracked_dir}")
            print(f"Processing {target_file}")
        elif file_pattern == 'MCTrack':
            # 处理MCTrack-final.pkl文件
            target_file = 'MCTrack-final.pkl'
            tracking_pkl_path = os.path.join(clip_tracked_dir, target_file)
            if not os.path.exists(tracking_pkl_path):
                raise FileNotFoundError(f"MCTrack-final.pkl not found in {clip_tracked_dir}")
            print(f"Processing {target_file}")
        else:
            raise ValueError(f"Unknown file_pattern: {file_pattern}. Use 'tracking-test' or 'BiTrack'")
        
        combine_data_from_label(clip_merged_dir, tracking_pkl_path, clip_tracked_split_dir)
        print('haha')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_dir", type=str, required=True)
    # parser.add_argument("--origin_dir", type=str, required=True)
    parser.add_argument("--merged_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--file_pattern", type=str, default="tracking-test", 
                        help="File pattern to process: 'tracking-test' or 'BiTrack' or 'MCTrack'")
    args = parser.parse_args()

    split_data(args.merged_dir, args.track_dir, args.dataset_name, args.file_pattern)

    # track_dir = '/data1/turbo_data/4D_label_dataset/models_res/offline_tracked'
    # origin_dir = '/data1/turbo_data/4D_label_dataset/origin'
    # merged_dir = '/data1/turbo_data/4D_label_dataset/models_res/merged'
    # split_data('train_hy_4d_road_7_20250209_lx')