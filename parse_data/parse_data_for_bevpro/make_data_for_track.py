import pickle
import os
import numpy as np
import math
import shutil
from tqdm import tqdm

save_dir = '/data1/turbo_data/4D_label_dataset/models_res/merged' #train_hy_4d_road_7_20250208_lx'
orign_dir = '/data1/turbo_data/4D_label_dataset/origin'
merged_dir = '/data1/turbo_data/4D_label_dataset/models_res/bevpro' # 临时
track_dir= '/data1/turbo_data/4D_label_dataset/models_res/offline_tracked'

def remakedir(folder):
    #高频代码
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def split_clip_for_merged(sub_t):
    sub_t_dir = os.path.join(save_dir,sub_t)
    # os.makedirs(sub_t_dir,exist_ok=True)
    remakedir(sub_t_dir)
    orgin_sub_t = os.path.join(orign_dir,sub_t)
    clip_list = os.listdir(orgin_sub_t)
    model_sub_t = os.path.join(merged_dir,sub_t,'model_pred') #wangruihao
    model_txt_set = set([i.split('.t')[0] for i in os.listdir(model_sub_t)])
    for clip_name in tqdm(clip_list):
        if 'txt' in clip_name:
            continue
        # os.makedirs(clip_save_dir,exist_ok=True)
        orgin_clip_dir = os.path.join(orgin_sub_t,clip_name,'camera_0_8')
        orgin_file_list = os.listdir(orgin_clip_dir)
        count = 0
        for file_name in orgin_file_list:
            base_name = file_name.split('.jp')[0]
            if base_name in model_txt_set:
                count += 1
        if count < len(orgin_file_list):
            continue
        clip_save_dir = os.path.join(sub_t_dir,clip_name)
        remakedir(clip_save_dir)
        for file_name in orgin_file_list:
            base_name = file_name.split('.jp')[0]
            if not os.path.exists(os.path.join(clip_save_dir,base_name+'.txt')):
                shutil.copy(os.path.join(model_sub_t,base_name+'.txt'), os.path.join(clip_save_dir,base_name+'.txt'))

def get_track_data(res_dir,save_pkl_path,sequence_name):
    # res = '/data1/turbo_data/wangruihao/data/4D_label/test/2024-12-13-15-19-36_19_cpp_sync_png_fisheye_wrh/model_res/merge_res'
    # save_pkl_path = '/data1/turbo_data/wangruihao/data/4D_label/test/2024-12-13-15-19-36_19_cpp_sync_png_fisheye_wrh/model_res/track_.pkl'
    # sequence_name = '20250120'
    res_list = sorted(os.listdir(res_dir))
    pose =np.array([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
    name_combile_dict = [
        'car', #0
        'bus', #1
        'truck', #2
        'rider', #3
        'bicycle', #4
        'person', #5
        'NotUsed' #6
        ]
    label_dict = {
        'car':0,
        'bus':1,
        'truck':2,
        'bike':4,
        'bicycle':4,
        'person':5,
        'motor':4,
        'NotUsed':6,
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
                content_list = line_.strip().split('\t')
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



def make_data_for_track(sub_t):
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
    split_clip_for_merged('train_hy_4d_road_7_20250212_lx')
    # make_data_for_track('train_hy_4d_road_7_20250209_lx')