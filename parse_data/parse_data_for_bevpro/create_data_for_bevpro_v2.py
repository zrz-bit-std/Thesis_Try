import numpy as np
import os
# import open3d as o3d
import json
import shutil
from tqdm import tqdm
import copy
import cv2
import argparse

from BEVFUSION.tools.utils import *
from BEVFUSION.tools.data_process.sensor_modules import (
    roadside_sensor_modules,
    # camera_lidar_pairs,
    single_pole_roadside_sensor_modules,
    road_camera_lidar_pairs,
)


def worker(args):
    info, save_root, param_info, mode = args
    # info = {"cams": {"camera_0_1": /path/to/img, ...}, "lidars": {"lidar_0_12": /path/to/lidar}, ...}}
    
    for cam_name, img_path in info["cams"].items():
        dst_path = os.path.join(
            save_root, "samples", cam_name, os.path.basename(img_path)
        )
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        # if os.path.exists(dst_path):
        #     continue
        cur_img = cv2.imread(img_path)
        if cam_name.endswith("_8"):  # fisheye
            cur_img = cv2.resize(cur_img, (1024, 1024))
        else:
            cur_img = cv2.resize(cur_img, (1280, 720))
        cv2.imwrite(dst_path, cur_img)
    
    # lidar
    lidar_info = param_info["lidar"]
    # 路口点云合并
    if mode == "intersection":
        points = []
        for lidar_name, lidar_path in info["lidars"].items():
            lp = readlidar(lidar_path, fulldata=True)[:, :5] #load dim=5
            cur_coordinates = lidar_info[lidar_name]["coordinates"]
            
            # 路口lidar点统一转到bev_local坐标系
            if cur_coordinates == "bev_local":
                pass
            elif cur_coordinates == "lidar":  # lidar
                lp[:, :3] = trans3dpoints(
                    lp[:, :3], lidar_info[lidar_name]['lidar2local']
                )
            else:
                raise NotImplementedError(
                    f"not support {mode} coord-system for intersection"
                )
            points.append(lp)
        points = np.concatenate(points, axis=0)
        basename = os.path.basename(list(info["lidars"].values())[0])
        lidar_save = os.path.join(
            save_root,
            "samples",
            "lidar",
            basename.replace(".pcd", ".pcd.bin"),
        )
        os.makedirs(os.path.dirname(lidar_save), exist_ok=True)
        with open(lidar_save, "wb") as fp:
            fp.write(points.tobytes())
    # 路段点云分开存
    elif mode == "roadsection":
        for lidar_name, lidar_path in info["lidars"].items():
            lp = readlidar(lidar_path, fulldata=True)[:, :5] #load dim=5
            cur_coordinates = lidar_info[lidar_name]["coordinates"]

            # lidar点统一转到percept坐标系
            if cur_coordinates == "bev_local":
                lp[:, :3] = trans3dpoints(
                    lp[:, :3], lidar_info[lidar_name]['local2percept']
                )
            elif cur_coordinates == "percept":
                pass
            else:  # lidar
                lp[:, :3] = trans3dpoints(
                    lp[:, :3],
                    lidar_info[lidar_name]['local2percept'] @ lidar_info[lidar_name]['lidar2local']
                )
            points = np.array(lp)
            basename = os.path.basename(lidar_path)
            lidar_save = os.path.join(
                save_root,
                "samples",
                lidar_name,
                basename.replace(".pcd", ".pcd.bin"),
            )
            os.makedirs(os.path.dirname(lidar_save), exist_ok=True)
            with open(lidar_save, "wb") as fp:
                fp.write(points.tobytes())
    else:
        raise NotImplementedError(f"not support {mode}")


def create_data(
    base_total_dir,
    sub_t,
    sensor_modules,
):

    sub_base_total_dir = os.path.join(base_total_dir, sub_t)
    
    camera_data_path_dict = {
        name: "" for name in sensor_modules
        if name.startswith("camera")
    }
    lidar_data_path_dict = {
        name: "" for name in sensor_modules
        if name.startswith("lidar")
    }
    group_data_per_ts = []
    for clip_name in os.listdir(sub_base_total_dir):
        if "txt" in clip_name or "result_json" in clip_name or "train_" in clip_name:
            continue
        clip_dir = os.path.join(sub_base_total_dir, clip_name)
        
        pcd_names = os.listdir(
            os.path.join(
                clip_dir, list(lidar_data_path_dict.keys())[0]
            )
        )
        for i, pcd_name in enumerate(pcd_names):
            info = {
                "cams": copy.deepcopy(camera_data_path_dict),
                "lidars": copy.deepcopy(lidar_data_path_dict),
            }
            
            for cam in camera_data_path_dict.keys():
                cur_img_path = os.path.join(
                    clip_dir, cam, pcd_name.replace(".pcd", ".jpg")
                )
                assert os.path.exists(cur_img_path)
                info["cams"][cam] = cur_img_path
            
            for lidar in lidar_data_path_dict.keys():
                cur_lidar_path = os.path.join(
                    clip_dir, lidar, pcd_name
                )
                assert os.path.exists(cur_lidar_path)
                info["lidars"][lidar] = cur_lidar_path
            group_data_per_ts.append(info)
    
    return group_data_per_ts

def create_train_data(
    save_root,
    mode,
    param_info,
    sensor_modules,
    road_id,
):
    samples_total_path = os.path.join(save_root, "samples")
    scences_path = os.path.join(save_root, "scences")
    os.makedirs(scences_path, exist_ok=True)

    local2percept = (mode == "roadsection")
    cam_names = [
        name for name in sensor_modules
        if name.startswith("camera") and not name.endswith("8")
    ]
    cam_names_fisheye = [
        name for name in sensor_modules
        if name.startswith("camera") and name.endswith("8")
    ]
    lidar_names = [
        name for name in sensor_modules if name.startswith("lidar")
    ]
    all_cam_names = cam_names + cam_names_fisheye
    cam2img = {
        cam:param_info['cam'][cam]['intrinsic'].tolist()
        for cam in all_cam_names
    }
    if not local2percept:
        lidar2cam = {
            cam:np.linalg.inv(param_info['cam'][cam]['cam2local']).tolist()
            for cam in all_cam_names
        }
    else:
        #更新 lidar2cam = percept2cam
        lidar2cam = {}
        for cam in all_cam_names:
            # cur_lidar = camera_lidar_pairs[cam]  # 需要注意为lidar坐标系的情况！！！！
            cur_lidar = road_camera_lidar_pairs[road_id][cam]  # TODO:需要注意为lidar坐标系的情况！！！！
            print(cam, cur_lidar)
            mat = np.linalg.inv(param_info['cam'][cam]['cam2local']) @ \
                  np.linalg.inv(param_info['lidar'][cur_lidar]['local2percept'])
            lidar2cam[cam] = mat.tolist()

    # group info for cam, fisheye
    group_cam2img = {cam: cam2img[cam] for cam in cam_names}
    group_cam2img_fisheye = {cam: cam2img[cam] for cam in cam_names_fisheye}
    group_lidar2cam = {cam: lidar2cam[cam] for cam in cam_names}
    group_lidar2cam_fisheye = {
        cam: lidar2cam[cam] for cam in cam_names_fisheye
    }
    distort_fisheye = {
        cam: param_info["cam"][cam]["distortion"].tolist()
        for cam in cam_names_fisheye
    }

    sync_timestamps = [
        img_name.split(".jpg")[0]
        for img_name in os.listdir(
            os.path.join(samples_total_path, cam_names[0])
        )
    ]
    json_save_path = os.path.join(scences_path, "test.json")
    res = []
    if mode == "intersection":
        assert len(cam_names_fisheye) > 0
        group_samples = []
        for cur_ts in tqdm(sync_timestamps):
            frame = {
                "cam_imgs": {},
                "cam_fisheye_imgs": {},
            }
            for cur_cam_name in cam_names:
                frame["cam_imgs"][cur_cam_name] = os.path.join(
                    cur_cam_name, f"{cur_ts}.jpg"
                )
            for cur_cam_name in cam_names_fisheye:
                frame["cam_fisheye_imgs"][cur_cam_name] = os.path.join(
                    cur_cam_name, f"{cur_ts}.jpg"
                )
            frame["lidar_pts"] = {"lidar": f"lidar/{cur_ts}.pcd.bin"}
            frame["annos"] = []
            frame["token"] = cur_ts
            frame["dataset_root"] = samples_total_path
            group_samples.append(frame)
        res.append(
            {
                "nbr_sample": len(group_samples),
                "cam2img": group_cam2img,
                "cam2img_fisheye": group_cam2img_fisheye,
                "lidar2cam": group_lidar2cam,
                "lidar2cam_fisheye": group_lidar2cam_fisheye,
                "distort_fisheye": distort_fisheye,
                "samples": group_samples,
            }
        )
        with open(json_save_path, "w") as f:
            f.write(json.dumps(res, indent=4))
    else:  # roadsection
        group_samples = []
        for cur_ts in tqdm(sync_timestamps):
            for cur_cam_name in cam_names:
                # # 路段每个点云分开存放
                frame = {
                    "cam_imgs": {},
                    "lidar_pts": {},
                }

                frame["cam_imgs"][cur_cam_name] = os.path.join(
                    cur_cam_name, f"{cur_ts}.jpg"
                )

                # cur_lidar_name = camera_lidar_pairs[cur_cam_name]
                cur_lidar_name = road_camera_lidar_pairs[road_id][cur_cam_name]
                frame["lidar_pts"]["lidar"] = os.path.join(
                    cur_lidar_name, f"{cur_ts}.pcd.bin"
                )
                frame["annos"] = []
                frame["token"] = cur_ts
                frame["dataset_root"] = samples_total_path
                group_samples.append(frame)
        
        res.append(
            {
                "nbr_sample": len(group_samples),
                "cam2img": group_cam2img,
                "lidar2cam": group_lidar2cam,
                "samples": group_samples,
            }
        )
        with open(json_save_path, "w") as f:
            f.write(json.dumps(res, indent=4))


if __name__ == '__main__':
    # create_data('train_hy_4d_road_7_20250304_lx')
    # create_train_data('train_hy_4d_road_7_20250304_lx')
    # '''
    # /opt/conda/bin/python /data1/turbo_data/wangruihao/code/4D_label/parse_data/parse_data_for_bevpro/create_data_for_bevpro.py
    # '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/data1/turbo_data/huben/data/test_4D_label/origin/")
    parser.add_argument('--save_dir', type=str, default="/data1/turbo_data/huben/data/test_4D_label/model_res/bevpro")
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--road', type=str, required=True)
    parser.add_argument('--calib', type=str, default='tools/coordinate_trans/BevCalib')
    parser.add_argument('--calib_version', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pnum', type=int, default=30)
    parser.add_argument(
        '--mode',
        type=str,
        default="intersection",
        choices=["intersection", "roadsection"],
        help="collect only one mode lidar data.",
    )
    parser.add_argument(
        '--pole',
        type=str,
        default="None",
        choices=["S0", "S1", "S2", "S3", "None"],
        help="collect only given module data.",
    )
    args = parser.parse_args()

    road_id = f"{args.location}_{args.road}"
    param_info = get_calib_params_v2(
        args.calib,
        "./BEVFUSION/tools/data_process/calibration_configs/calib_file.yaml",
        road_id,
        args.calib_version,
    )
    sensor_modules = roadside_sensor_modules[road_id][args.mode]
    if args.mode == "roadsection":
        assert args.pole in ["S0", "S1", "S2", "S3"]
        sensor_modules = (
            single_pole_roadside_sensor_modules[road_id][args.mode][args.pole]
        )
    
    # TODO: temp code for hy_7 202502 data without camera_0_0
    if road_id == "hy_7" and args.calib_version == "v20241114" and args.mode == "intersection":  # !!
        sensor_modules.pop(sensor_modules.index("camera_0_0"))


    group_data_per_ts = create_data(
        args.data_root, args.dataset_name, sensor_modules
    )
    save_root = os.path.join(args.save_dir, args.dataset_name)
    
    datas = [
        [info, save_root, param_info, args.mode]
        for info in group_data_per_ts
    ]

    poolprocess(datas, worker, args.pnum) 
    
    
    create_train_data(
        save_root, args.mode, param_info, sensor_modules, road_id
    )
        