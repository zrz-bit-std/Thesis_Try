import argparse
import pickle
import os
import numpy as np
import math
import shutil
from tqdm import tqdm

# save_dir = '/data1/turbo_data/4D_label_dataset/models_res/merged' #train_hy_4d_road_7_20250208_lx'
# orign_dir = '/data1/turbo_data/4D_label_dataset/origin'
# merged_dir = '/data1/turbo_data/4D_label_dataset/models_res/bevpro' # 临时
# track_dir= '/data1/turbo_data/4D_label_dataset/models_res/offline_tracked'


def remakedir(folder):
    #高频代码
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def get_timestamps(clip_dir):
    camera_names = [
        name for name in os.listdir(clip_dir) if name.startswith("camera")
    ]
    assert len(camera_names) > 0
    img_dir = os.path.join(clip_dir, camera_names[0])
    timestamps = [
        name.split(".jpg")[0] for name in os.listdir(img_dir)
    ]
    return timestamps


def split_clip_for_merged(save_dir, origin_dir, det_dir, sub_t):
    sub_t_dir = os.path.join(save_dir,sub_t)
    # os.makedirs(sub_t_dir,exist_ok=True)
    remakedir(sub_t_dir)
    orgin_sub_t = os.path.join(origin_dir,sub_t,'original_data')
    clip_list = os.listdir(orgin_sub_t)
    model_sub_t = os.path.join(det_dir, sub_t, 'model_pred')
    model_txt_set = set([i.split('.t')[0] for i in os.listdir(model_sub_t)])
    breakpoint()
    for clip_name in tqdm(clip_list):
        if 'txt' in clip_name:
            continue
        timestamps_str = get_timestamps(os.path.join(orgin_sub_t, clip_name))
        count = 0
        for ts in timestamps_str:
            if ts in model_txt_set:
                count += 1
        if count < len(timestamps_str):
            continue
        clip_save_dir = os.path.join(sub_t_dir, clip_name)
        # remakedir(clip_save_dir)
        os.makedirs(clip_save_dir, exist_ok=True)
        for ts in timestamps_str:
            if not os.path.exists(os.path.join(clip_save_dir, f"{ts}.txt")):
                shutil.copy(
                    os.path.join(model_sub_t, f"{ts}.txt"),
                    os.path.join(clip_save_dir, f"{ts}.txt"),
                )

def get_track_data(res_dir,save_pkl_path,sequence_name):
    # res = '/data1/turbo_data/wangruihao/data/4D_label/test/2024-12-13-15-19-36_19_cpp_sync_png_fisheye_wrh/model_res/merge_res'
    # save_pkl_path = '/data1/turbo_data/wangruihao/data/4D_label/test/2024-12-13-15-19-36_19_cpp_sync_png_fisheye_wrh/model_res/track_.pkl'
    # sequence_name = '20250120'
    res_list = sorted(os.listdir(res_dir))
    pose =np.array([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
    # name_combile_dict = [
    #     'car', #0
    #     'bus', #1
    #     'truck', #2
    #     'rider', #3
    #     'bicycle', #4
    #     'person', #5
    #     'NotUsed' #6
    #     ]
    # label_dict = {  # TODO:修改类别！！！！！！
    #     'car':0,
    #     'bus':1,
    #     'truck':2,
    #     'bike':4,
    #     'bicycle':4,
    #     'person':5,
    #     'motor':4,
    #     'motorcycle': 4,
    #     'NotUsed':6,
    #     'rider': 3,
    #     'pedestrain':5,
    #     'pedestrian':5
    # }

    # TODO:临时7类 !!!!
    name_combile_dict = [  # 对齐模型训练顺序！！！
        'car', #0
        'truck', # 1   
        'bus', # 2
        'rider', #3
        'bicycle', # 4
        'person', # 5
        'motorcycle', # 6
        'NotUsed' # 7
        ]
    label_dict = {  # TODO:修改类别！！！！！！
        'car':0,
        'truck':1,
        'bus':2,
        'bike':4,
        'bicycle':4,
        'person':5,
        'motor':6,
        'motorcycle':6,
        'NotUsed':7,
        'rider': 3,
        'pedestrain':5,
        'pedestrian':5
    }

    list_for_pickle = []
    frame_id = 0
    for item in res_list:
        time_str = float(item.split('.')[0])
        frame_path = os.path.join(res_dir,item)
        name_list = []
        score_list = []
        boxes_lidar_list = []
        with open(frame_path,'r') as f:
            for line_ in f.readlines():
                content_list = line_.strip().split()#'\t'
                name_list.append(name_combile_dict[label_dict[content_list[0]]])
                h,w,l,x,y,z,yaw,confidence = [float(i) for i in content_list[1:]] #c, h, w, l, new_center[0], new_center[1], new_center[2], yaw_new, confidence
                score_list.append(confidence)
                boxes_lidar_frame = np.array([x,y,z,l,w,h,math.radians(yaw),confidence,label_dict[content_list[0]]])
                boxes_lidar_list.append(boxes_lidar_frame)
        name_array = np.array(name_list,dtype=np.dtype('U22'))
        save_dict = {
            'name':name_array,
            'boxes_lidar':np.array(boxes_lidar_list),
            'frame_id':frame_id,
            'pose':pose,
            'score':np.array(score_list),
            'sequence_name':sequence_name
        }
        frame_id += 1
        list_for_pickle.append(save_dict)
    with open(save_pkl_path,'wb') as f:
        pickle.dump(list_for_pickle,f)



def make_data_for_track(save_dir, track_dir, sub_t):
    sub_t_dir = os.path.join(save_dir,sub_t)
    track_sub_t_dir = os.path.join(track_dir,sub_t)
    os.makedirs(track_sub_t_dir,exist_ok=True)
    remakedir(track_sub_t_dir)
    clip_list = os.listdir(sub_t_dir)
    for clip_name in tqdm(clip_list):
        clip_merged_dir = os.path.join(sub_t_dir,clip_name)
        clip_track_dir = os.path.join(track_sub_t_dir,clip_name)
        os.makedirs(clip_track_dir,exist_ok=True)
        save_pkl_dir = os.path.join(clip_track_dir,'track.pkl')
        if os.path.exists(save_pkl_dir):
            os.remove(save_pkl_dir)
        sequence_name = sub_t + '_' + clip_name
        get_track_data(clip_merged_dir,save_pkl_dir,sequence_name=sequence_name)


if __name__ == '__main__':
    # split_clip_for_merged('train_hy_4d_road_7_20250212_lx')
    # # make_data_for_track('train_hy_4d_road_7_20250209_lx')

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--origin_dir', type=str, required=True)
    parser.add_argument('--det_dir', type=str, required=True)
    parser.add_argument('--track_dir', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()

    split_clip_for_merged(
        args.save_dir, args.origin_dir, args.det_dir, args.dataset_name
    )
    make_data_for_track(args.save_dir, args.track_dir, args.dataset_name)

    # save_dir = '/data1/turbo_data/huben/data/test_4D_label/model_res/merged' #train_hy_4d_road_7_20250208_lx'
    # orign_dir = '/data1/turbo_data/huben/data/test_4D_label/origin'
    # det_dir = '/data1/turbo_data/huben/data/test_4D_label/model_res/bevpro'
    # # merged_dir = '/data1/turbo_data/4D_label_dataset/models_res/bevpro' # 临时
    # track_dir= '/data1/turbo_data/huben/data/test_4D_label/model_res/offline_tracked'