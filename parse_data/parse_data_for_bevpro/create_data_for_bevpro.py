import numpy as np
import os
# import open3d as o3d
import json
import shutil
from tqdm import tqdm
import copy
base_total_dir = '/data1/turbo_data/4D_label_dataset/origin/'
save_dir = '/data1/turbo_data/4D_label_dataset/models_res/bevpro'
example_data_path = '/data1/turbo_data/zhangchengyue/code/BEV/data/Intersection/test/qatest/2024-12-13-15-19-36_19/cpp_sync_png_fisheye_wrh/scences/test.json'

def get_fisheye_param_fisheye_7():
    intrinsic = np.array([
        [2.6825344481076627e+02, 0.0, 5.1297656203507881e+02, 0.0], 
        [0.0, 2.6878278304571751e+02, 5.1329052715433045e+02, 0.0], 
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    undist = np.array([1.1891781083111905e-01, -3.8331731142404291e-02,  #k1, k2
                5.6868161989457315e-02, -2.4351293556834262e-02])  #k3, k4 # 错误的json 系数
    camera_0_8_ex = np.linalg.inv(np.array([ -7.4411897129214200e-02, -9.9239089026138250e-01,
                            9.8097861810800685e-02, -4.6080716655589640e+01,
                            -9.9722269616376902e-01, 7.4358859134819044e-02,
                            -4.2017048957394484e-03, -7.0600274943280965e+00,
                            -3.1247114257178022e-03, -9.8138071075336605e-02,
                            -9.9516790301140545e-01, 5.3533511383741157e+00, 0., 0., 0., 1. ]).reshape(4,4))
    # camera_0_8_ex[:,:2] *= -1
    camera_1_8_ex = np.linalg.inv(np.array([ -9.6727440947129129e-01, 2.5371865412167660e-01,
       -2.6573168137854954e-03, 5.4604878216050565e+00,
       2.5367707688507524e-01, 9.6679080401659467e-01,
       -3.1040005347768868e-02, 5.0070519636152312e+01,
       -5.3063589218390675e-03, -3.0698303204427544e-02,
       -9.9951461056622948e-01, 6.1422552007077940e+00, 0., 0., 0., 1. ]).reshape(4,4))
    # camera_1_8_ex[:,:2] *= -1
    camera_2_8_ex = np.linalg.inv(np.array([ 4.5654120099610118e-02, 9.9877777119199418e-01,
       -1.8938455340461555e-02, 4.0777177474461496e+01,
       9.9847836333852125e-01, -4.6211018505724677e-02,
       -3.0091522286040302e-02, 1.0949246306670830e+01,
       -3.0929908870833474e-02, -1.7535835920076754e-02,
       -9.9936771770746424e-01, 7.0538489409727134e+00, 0., 0., 0., 1. ]).reshape(4,4))
    # camera_2_8_ex[:,:2] *= -1
    camera_3_8_ex = np.linalg.inv(np.array([ -9.7751236030234967e-01, 2.0784785117067953e-01,
       -3.5621008125267160e-02, -6.2453175554983318e+00,
       2.0703327972218499e-01, 9.7800810491737700e-01,
       2.5246144327335747e-02, -5.2295780788641423e+01,
       4.0084991500620615e-02, 1.7303683990763086e-02,
       -9.9904643334368759e-01, 6.4877459676119571e+00, 0., 0., 0., 1. ]).reshape(4,4))
    # camera_3_8_ex[:,:2] *= -1
    return  [camera_0_8_ex,camera_1_8_ex,camera_2_8_ex,camera_3_8_ex],intrinsic,undist

def get_lidar_path(lidar_lidar_path):
    line_start = 11
    points = [line.strip().split() for line in open(lidar_lidar_path, 'r').readlines()[line_start:]]
    return np.array(points, dtype=np.float32)

def create_data(sub_t):
    bev_pro_sub_dir = os.path.join(save_dir,sub_t)
    os.makedirs(bev_pro_sub_dir,exist_ok=True)
    ########################################################
    samples_path = os.path.join(bev_pro_sub_dir,'samples')
    os.makedirs(samples_path,exist_ok=True)
    bev_pro_lidar_path = os.path.join(samples_path,'lidar')
    os.makedirs(bev_pro_lidar_path,exist_ok=True)

    bev_pro_lidar_camera_0_0 = os.path.join(samples_path,'camera_0_0')
    os.makedirs(bev_pro_lidar_camera_0_0,exist_ok=True)

    bev_pro_lidar_camera_1_0 = os.path.join(samples_path,'camera_1_0')
    os.makedirs(bev_pro_lidar_camera_1_0,exist_ok=True)

    bev_pro_lidar_camera_2_0 = os.path.join(samples_path,'camera_2_0')
    os.makedirs(bev_pro_lidar_camera_2_0,exist_ok=True)

    bev_pro_lidar_camera_3_0 = os.path.join(samples_path,'camera_3_0')
    os.makedirs(bev_pro_lidar_camera_3_0,exist_ok=True)

    bev_pro_lidar_camera_0_8 = os.path.join(samples_path,'camera_0_8')
    os.makedirs(bev_pro_lidar_camera_0_8,exist_ok=True)

    bev_pro_lidar_camera_1_8 = os.path.join(samples_path,'camera_1_8')
    os.makedirs(bev_pro_lidar_camera_1_8,exist_ok=True)

    bev_pro_lidar_camera_2_8 = os.path.join(samples_path,'camera_2_8')
    os.makedirs(bev_pro_lidar_camera_2_8,exist_ok=True)

    bev_pro_lidar_camera_3_8 = os.path.join(samples_path,'camera_3_8')
    os.makedirs(bev_pro_lidar_camera_3_8,exist_ok=True)
    ############################
    sub_base_total_dir = os.path.join(base_total_dir,sub_t)
    base_dir_list = os.listdir(sub_base_total_dir)
    for base_d in tqdm(base_dir_list): #一个文件架子
        if 'txt' in base_d:
            continue
        base_dir = os.path.join(sub_base_total_dir,base_d)
        lidar_path_list = ['lidar_0_12','lidar_1_12','lidar_2_12','lidar_3_12']
        lidar_path_dir_0 = os.path.join(base_dir,lidar_path_list[0])
        base_name_path_list = os.listdir(lidar_path_dir_0)
        for base_name in base_name_path_list:
            ## lidar
            base_name_ = base_name.split('.p')[0]
            if not os.path.exists(os.path.join(bev_pro_lidar_path,base_name_+'.pcd.bin')):
                lidar_lidar_path_0 = os.path.join(base_dir,lidar_path_list[0],base_name)
                lidar_lidar_path_1 = os.path.join(base_dir,lidar_path_list[1],base_name)
                lidar_lidar_path_2 = os.path.join(base_dir,lidar_path_list[2],base_name)
                lidar_lidar_path_3 = os.path.join(base_dir,lidar_path_list[3],base_name)
                points_0 = get_lidar_path(lidar_lidar_path_0)#[line.strip().split() for line in open(lidar_lidar_path_0, 'r').readlines()[line_start:]]
                points_1 = get_lidar_path(lidar_lidar_path_1)#[line.strip().split() for line in open(lidar_lidar_path_1, 'r').readlines()[line_start:]]
                points_2 = get_lidar_path(lidar_lidar_path_2)#[line.strip().split() for line in open(lidar_lidar_path_2, 'r').readlines()[line_start:]]
                points_3 = get_lidar_path(lidar_lidar_path_3)#[line.strip().split() for line in open(lidar_lidar_path_3, 'r').readlines()[line_start:]]
                points_toal = np.concatenate([points_0,points_1,points_2,points_3],axis=0)
                with open(os.path.join(bev_pro_lidar_path,base_name_+'.pcd.bin'), 'wb') as fp:
                    fp.write(points_toal.tobytes())

            ## camera
            # shutil.copy(os.path.join(base_dir,'camera_0_0',base_name_+'.jpg'),os.path.join(bev_pro_lidar_camera_0_0,base_name_+'.jpg'))
            if not os.path.exists(os.path.join(bev_pro_lidar_camera_3_8,base_name_+'.jpg')): # 
                # shutil.copy(os.path.join(base_dir,'camera_0_0',base_name_+'.jpg'),os.path.join(bev_pro_lidar_camera_0_0,base_name_+'.jpg'))  # wangruihao
                shutil.copy(os.path.join(base_dir,'camera_1_0',base_name_+'.jpg'),os.path.join(bev_pro_lidar_camera_1_0,base_name_+'.jpg'))
                shutil.copy(os.path.join(base_dir,'camera_2_0',base_name_+'.jpg'),os.path.join(bev_pro_lidar_camera_2_0,base_name_+'.jpg'))
                shutil.copy(os.path.join(base_dir,'camera_3_0',base_name_+'.jpg'),os.path.join(bev_pro_lidar_camera_3_0,base_name_+'.jpg'))

                shutil.copy(os.path.join(base_dir,'camera_0_8',base_name_+'.jpg'),os.path.join(bev_pro_lidar_camera_0_8,base_name_+'.jpg'))
                shutil.copy(os.path.join(base_dir,'camera_1_8',base_name_+'.jpg'),os.path.join(bev_pro_lidar_camera_1_8,base_name_+'.jpg'))
                shutil.copy(os.path.join(base_dir,'camera_2_8',base_name_+'.jpg'),os.path.join(bev_pro_lidar_camera_2_8,base_name_+'.jpg'))
                shutil.copy(os.path.join(base_dir,'camera_3_8',base_name_+'.jpg'),os.path.join(bev_pro_lidar_camera_3_8,base_name_+'.jpg'))


def create_train_data(sub_t):
    # example_data_path = '/data1/turbo_data/zhangchengyue/code/BEV/data/Intersection/test/qatest/2024-12-13-15-19-36_19/cpp_sync_png_fisheye_wrh/scences/test.json'
    samples_total_path = os.path.join(save_dir,sub_t,'samples')
    scences_path = os.path.join(save_dir,sub_t,'scences')
    os.makedirs(scences_path,exist_ok=True)
    ex_param_list,inter_param,undist = get_fisheye_param_fisheye_7()
    ex_param_0 = ex_param_list[0].tolist()
    ex_param_1 = ex_param_list[1].tolist()
    ex_param_2 = ex_param_list[2].tolist()
    ex_param_3 = ex_param_list[3].tolist()
    in_param = inter_param.tolist()
    with open(example_data_path) as f:
        res = json.load(f)
    res_new = []
    camera_1_0_path = os.path.join(samples_total_path,'camera_1_0')
    file_list = [i.split('.j')[0] for i in os.listdir(camera_1_0_path)]
    for sub_res in res:
        # samples = sub_res['samples']
        samples_new = []
        for frame_id in tqdm(range(len(file_list))):
            time_name = file_list[frame_id]
            frame = {}
            frame['cam_fisheye_imgs'] = {'camera_0_8': 'camera_0_8/' + time_name + '.jpg', 
                                        'camera_1_8': 'camera_1_8/' + time_name + '.jpg', 
                                        'camera_2_8': 'camera_2_8/' + time_name + '.jpg', 
                                        'camera_3_8': 'camera_3_8/' + time_name + '.jpg'}
            
            frame['cam_imgs'] = {'camera_0_0': 'camera_1_0/' + time_name + '.jpg', # wangruihao
                                        'camera_1_0': 'camera_1_0/' + time_name + '.jpg', 
                                        'camera_2_0': 'camera_2_0/' + time_name + '.jpg', 
                                        'camera_3_0': 'camera_3_0/' + time_name + '.jpg'}
            
            frame['lidar_pts'] = {'lidar': 'lidar/' + time_name + '.pcd.bin',}
            frame['annos'] = []
            frame['token'] = time_name
            samples_new.append(frame)
            
        cam2img_fisheye = {
            'camera_0_8':in_param, 
            'camera_1_8':in_param,
            'camera_2_8':in_param,
            'camera_3_8':in_param,
        }
        lidar2cam_fisheye = {
            'camera_0_8':ex_param_0, # wangruihao 
            'camera_1_8':ex_param_1,
            'camera_2_8':ex_param_2,
            'camera_3_8':ex_param_3,
        }
        distort_fisheye = {
            'camera_0_8':undist.tolist(),
            'camera_1_8':undist.tolist(),
            'camera_2_8':undist.tolist(),
            'camera_3_8':undist.tolist(),
        }
        if len(samples_new) > 0:
            res_new.append({
                'nbr_sample': len(samples_new),
                'samples': samples_new,
                'lidar2cam_fisheye':lidar2cam_fisheye,
                'cam2img_fisheye': cam2img_fisheye,
                'cam2img': sub_res['cam2img'],
                'lidar2cam': sub_res['lidar2cam'],
                'distort_fisheye': distort_fisheye
            })
    with open(os.path.join(save_dir,sub_t,'scences','test.json'),'w') as f1:
        json.dump(res_new,f1)


if __name__ == '__main__':
    create_data('train_hy_4d_road_7_20250211_lx')
    create_train_data('train_hy_4d_road_7_20250211_lx')
    '''
    /opt/conda/bin/python /data1/turbo_data/wangruihao/code/4D_label/parse_data/parse_data_for_bevpro/create_data_for_bevpro.py
    '''
        