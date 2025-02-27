import mmcv
import numpy as np
import os
import os.path as osp
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
from tqdm import tqdm

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

mogo_categories = ('car', 'truck', 'bus',
                  'bicycle', 'pedestrian', 
                  'rider')



import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets.mogo_dataset import MogoDataset
import glob
import json

def parse_json_file(file_name):
    with open(file_name,'r') as fp:
        scence_list = json.load(fp)
    return scence_list

def create_mogo_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          use_fisheye = False):
    """Create info file of nuscene dataset.
    Given the raw data, generate its related info file in pkl format.
    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
        max_radar_sweeps (int): Max number of radar sweeps. 
            Default: 10
    """
   
    """
    create info file for mogo dataset
    Args:
        root_path (str): path of the data root
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional):  Version of the data. Defaults to "v1.0-trainval".
        max_sweeps (int, optional): Max number of sweeps. Defaults to 10.
    """
    scence_path = os.path.join(root_path,"scences")
    sample_path = os.path.join(root_path,'samples')
    scences = os.listdir(scence_path)
    
    scence_info = {}
    for json_file in scences:
        prefix = osp.splitext(json_file)[0]
        if prefix not in ['train', 'val','test']:
            continue
        json_file = scence_path + '/'+ json_file
        info = parse_json_file(json_file)
        scence_info[prefix] = info
   
    val_nusc_infos_ = _fill_trainval_infos(scence_info, False,use_fisheye=use_fisheye)

    val_nusc_infos = list()
    for sample in val_nusc_infos_:
        lidar_path = sample['lidar_path']
        abs_path = os.path.join(root_path,'samples/'+lidar_path)
        if not  os.path.exists(abs_path):
            print(abs_path)
            continue
        val_nusc_infos.append(sample)
    
    metadata= dict(version=version)
    data = dict(infos=val_nusc_infos_, metadata=metadata)
    info_path = osp.join(root_path,
                            '{}_infos_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_path) 


def parse_single_scence(scences, test=False):
    scence_info = []
    for scence in scences:
        cam2img = scence["cam2img"]
        lidar2cam =scence["lidar2cam"]
        for sample in tqdm(scence["samples"]):
            token = sample['token']
            lidar_path = sample['lidar_pts']["lidar"]
            timestamp = token.split("__")[-1] 
            info = {
                "lidar_path": lidar_path,
                "token": token,
                "cams": dict(),
                "timestamp": float(timestamp),
            } 
            for cam_key in cam2img.keys():
                lidar2camera = np.array(lidar2cam[cam_key]).reshape(4,4)
                cam_intrinsic = np.array(cam2img[cam_key])[:3,:3]
                rot = lidar2camera[:3,:3]
                trans = lidar2camera[:3,-1]
                c2l_R = rot.T
                c2l_T= -1 * np.matmul(rot.T,trans.reshape(3,1))
                if cam_key not in sample["cam_imgs"]:
                    continue
                cam_info ={
                    "camera_path": sample["cam_imgs"][cam_key],
                    "cam_intrinsic":cam_intrinsic,
                    "lidar2cam_rotation": rot,
                    "lidar2cam_translation":trans,
                    "cam2lidar_rotation": c2l_R,
                    "cam2lidar_translation": c2l_T
                }
                info["cams"].update({cam_key: cam_info})
            if not test:
                locs =[]
                dims =[]
                rots=[]
                valid_flgs =[]
                names =[]
                num_lidar_pts = []
                for anno in sample["annos"]:
                    locs.append([anno["center"][key] for key in anno["center"]])
                    dims.append([anno["size"][key] for key in anno["size"]])
                    rots.append([anno["rotation"]["yaw"] ])
                    valid_flgs.append(anno["pointnum"] > 0)
                    names.append(anno['type'])
                    num_lidar_pts.append(anno['pointnum'])
                locs = np.array(locs).reshape(-1,3)
                dims = np.array(dims).reshape(-1,3)
                rots = np.array(rots).reshape(-1,1)   
                valid_flag = np.array(valid_flgs)
                names = np.array(names)
                #change l,w,h -> w,l,h
                gt_boxes = np.concatenate([locs, dims, rots], axis=1)
                info["gt_boxes"] = gt_boxes
                info["gt_names"] = names
                info['num_lidar_pts'] = np.array(num_lidar_pts)
                info['valid_flag'] = valid_flag
                scence_info.append(info)

    return scence_info


def parse_single_scence_with_fisheye(scences, test=False): # wangruihao
    scence_info = []
    for scence in scences:
        cam2img = scence["cam2img"]
        lidar2cam =scence["lidar2cam"]
        cam2img_fisheye = scence["cam2img_fisheye"]
        lidar2cam_fisheye = scence["lidar2cam_fisheye"]
        distort_fisheye = scence["distort_fisheye"]
        for sample in tqdm(scence["samples"]):
            token = sample['token']
            lidar_path = sample['lidar_pts']["lidar"]
            timestamp = token.split("__")[-1] 
            info = {
                "lidar_path": lidar_path,
                "token": token,
                "cams": dict(),
                "cams_fisheye": dict(),
                "timestamp": float(timestamp),
            } 
            for cam_key in cam2img.keys():
                lidar2camera = np.array(lidar2cam[cam_key]).reshape(4,4)
                cam_intrinsic = np.array(cam2img[cam_key])[:3,:3]
                rot = lidar2camera[:3,:3]
                trans = lidar2camera[:3,-1]
                c2l_R = rot.T
                c2l_T= -1 * np.matmul(rot.T,trans.reshape(3,1))
                if cam_key not in sample["cam_imgs"]:
                    continue
                cam_info ={
                    "camera_path": sample["cam_imgs"][cam_key],
                    "cam_intrinsic":cam_intrinsic,
                    "lidar2cam_rotation": rot,
                    "lidar2cam_translation":trans,
                    "cam2lidar_rotation": c2l_R,
                    "cam2lidar_translation": c2l_T
                }
                info["cams"].update({cam_key: cam_info})
            for cam_key in cam2img_fisheye.keys():
                lidar2camera_fisheye = np.array(lidar2cam_fisheye[cam_key]).reshape(4,4)
                cam_fisheye_intrinsic = np.array(cam2img_fisheye[cam_key])[:3,:3]
                cam_fisheye_distort = np.array(distort_fisheye[cam_key])
                rot = lidar2camera_fisheye[:3,:3]
                trans = lidar2camera_fisheye[:3,-1]
                c2l_R = rot.T
                c2l_T= -1 * np.matmul(rot.T,trans.reshape(3,1))
                if cam_key not in sample["cam_fisheye_imgs"]:
                    continue
                cam_fisheye_info ={
                    "camera_fisheye_path": sample["cam_fisheye_imgs"][cam_key],
                    "cam_fisheye_intrinsic":cam_fisheye_intrinsic,
                    "lidar2cam_fisheye_rotation": rot,
                    "lidar2cam_fisheye_translation":trans,
                    "cam2lidar_fisheye_rotation": c2l_R,
                    "cam2lidar_fisheye_translation": c2l_T,
                    'cam_fisheye_distort':cam_fisheye_distort
                }
                info["cams_fisheye"].update({cam_key: cam_fisheye_info})
            if not test:
                locs =[]
                dims =[]
                rots=[]
                valid_flgs =[]
                names =[]
                num_lidar_pts = []
                for anno in sample["annos"]:
                    locs.append([anno["center"][key] for key in anno["center"]])
                    dims.append([anno["size"][key] for key in anno["size"]])
                    rots.append([anno["rotation"]["yaw"] ])
                    valid_flgs.append(anno["pointnum"] > 0)
                    names.append(anno['type'])
                    num_lidar_pts.append(anno['pointnum'])
                locs = np.array(locs).reshape(-1,3)
                dims = np.array(dims).reshape(-1,3)
                rots = np.array(rots).reshape(-1,1)   
                valid_flag = np.array(valid_flgs)
                names = np.array(names)
                #change l,w,h -> w,l,h
                gt_boxes = np.concatenate([locs, dims, rots], axis=1)
                info["gt_boxes"] = gt_boxes
                info["gt_names"] = names
                info['num_lidar_pts'] = np.array(num_lidar_pts)
                info['valid_flag'] = valid_flag
                scence_info.append(info)

    return scence_info


    
def _fill_trainval_infos(scenes,
                         test=False,
                         use_fisheye = False):
    """Generate the train/val infos from the raw data.
    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.
        max_radar_sweeps (int): Max number of radar sweeps. Default: 10.
    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    if use_fisheye:
        scence_info = parse_single_scence_with_fisheye(scenes['test']) #增加鱼眼
    return scence_info


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.
    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.
    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep






def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.
    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.
    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None




if __name__ == '__main__':
    # create_nuscenes_infos('data/nuscenes/', 'radar_nuscenes_5sweeps')
    pass
