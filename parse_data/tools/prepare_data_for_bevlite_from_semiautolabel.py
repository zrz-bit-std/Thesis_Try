import argparse
import os
import json
import math
import mmcv
import cv2

import numpy as np

from mmcv import Config
from torchpack.utils.config import configs
from tqdm import tqdm

from BEVFUSION.tools.data_process.sensor_modules import (
    roadside_sensor_modules,
    single_pole_roadside_sensor_modules,
)
from BEVFUSION.configs.base_config import (
    index2name_map, id2name_map, name2id_map
)
from BEVFUSION.tools.utils import (
    visualize_lidar,
    visualize_camera,
    visualize_camera_fisheye,
    imgvis,
    LiDARBaseBoxes,
    lidarvis,
)
from visual_data.utils.project_image import *
from visual_data.data_show_multiprocess import (
    OBJECT_PALETTE_FISHYE_DETECT,
    draw_pcr_on_cam,
    draw_pcr_on_fisheye,
)

from BEVFUSION.tools.visualize.get_infer_res import recursive_eval

from parse_data.tools.prepare_data_for_bevlite import (
    filter_ts_within_time_interval,
    satisfy_roi_region
)

# TODO:临时7类 !!!!对齐4d-label
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
    'ignore': 7,
    'rider': 3,
    'pedestrain':5,
    'pedestrian':5
}


def visualize(data_root, save_root, sample, mode, pcr):
    gt_bboxes = sample["gt_boxes"]
    gt_sublabels = sample["sublabels"]
    lim = [[-102, 102], [-102, 102]]
    pcd_file = os.path.join(data_root, sample["lidar_path"])
    assert os.path.exists(pcd_file)
    assert ".pcd.bin" in pcd_file
    ts = os.path.basename(pcd_file).replace(".pcd.bin", "")
    points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1,5) 
    boxes = LiDARBaseBoxes(gt_bboxes, box_dim=7, origin=(0.5,0.5,0.5))
    
    labels = np.array([name2id_map[key] for key in sample['gt_names']])

    visualize_lidar(
        os.path.join(save_root, f"{ts}_lidar.png"),
        points,
        bboxes=boxes,
        labels=labels,
        xlim=lim[0],
        ylim=lim[1],
        classes=id2name_map,
    )
    
    for cam, info in sample['cams'].items():
        cam_path = os.path.join(data_root, info['camera_path'])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = info['cam_intrinsic']
        lidar2camera = np.eye(4)
        lidar2camera[:3, :3] = info['lidar2cam_rotation']
        lidar2camera[:3, 3] = info['lidar2cam_translation']
        if not os.path.exists(cam_path):  # 3月之前数据存在枪机失效
            continue

        image = cv2.imread(cam_path)
        camera_matrix = intrinsic[:3, :3]
        camera_extrinsic = lidar2camera

        for i in range(gt_bboxes.shape[0]):
            x, y, z, w, l, h, yaw = gt_bboxes.copy()[i][:7]  # 注意方向
            cur_yaw = -yaw - np.pi / 2  # pkl中已经转了yaw角定义！！
            class_name_label = name_combile_dict[label_dict[index2name_map[gt_sublabels[i]]]]
            
            image, _ = draw_box_on_pinhole(
                (w,l,h,x,y,z),
                cur_yaw,
                image,
                camera_matrix,
                camera_extrinsic,
                color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1]
            )
        draw_pcr_on_cam(
            mode,
            road_id,
            image,
            pcr,
            camera_matrix,
            camera_extrinsic,
            None,
            cam,
            color=(255, 0, 0),
        )
        cv2.imwrite(os.path.join(save_root, f"{ts}_{cam}.png"), image)


    for cam, info in sample.get("cams_fisheye", {}).items():
        cam_path = os.path.join(data_root, info["camera_fisheye_path"])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = info["cam_fisheye_intrinsic"]
        lidar2camera = np.eye(4)
        lidar2camera[:3, :3] = info["lidar2cam_fisheye_rotation"]
        lidar2camera[:3, 3] = info['lidar2cam_fisheye_translation']
        distort_param = info["cam_fisheye_distort"]

        image = cv2.imread(cam_path)
        camera_matrix = intrinsic[:3, :3]
        camera_extrinsic = lidar2camera
        camera_distort_param = distort_param

        for i in range(gt_bboxes.shape[0]):
            x, y, z, w, l, h, yaw = gt_bboxes.copy()[i][:7]  # 注意方向
            cur_yaw = -yaw - np.pi / 2  # pkl中已经转了yaw角定义！！
            class_name_label = name_combile_dict[label_dict[index2name_map[gt_sublabels[i]]]]
            
            image, _ = draw_box_on_fisheye(
                (w,l,h,x,y,z),
                cur_yaw,
                image,
                camera_matrix,
                camera_extrinsic,
                distort=camera_distort_param,
                color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1]
            )
        # 可视化pcr边界
        draw_pcr_on_fisheye(
            mode,
            road_id,
            image,
            pcr,
            camera_matrix,
            camera_extrinsic,
            camera_distort_param,
            cam,
            color=(255, 0, 0),
        )
        cv2.imwrite(os.path.join(save_root, f"{ts}_{cam}.png"), image)


def collect_frame_ts_from_samples(samples):
    ts_list = []
    for sample in samples:
        fname = os.path.basename(sample["lidar_path"])
        assert ".pcd.bin" in fname
        ts_list.append(fname.replace(".pcd.bin", ""))
    assert len(ts_list) == len(set(ts_list))
    return ts_list


def collect_frame_ts_from_dataset_name(data_root, dataset_name):
    ori_dataset_dir = os.path.join(data_root, dataset_name, "original_data")
    assert os.path.exists(ori_dataset_dir)
    clip_dirs = [
        os.path.join(ori_dataset_dir, clip_name)
        for clip_name in os.listdir(ori_dataset_dir)
    ]
    ts_list = []
    for clip_dir in sorted(clip_dirs):
        if not os.path.isdir(clip_dir):
            continue
        pcd_dir = os.path.join(clip_dir, "3d_url")
        assert os.path.exists(pcd_dir)
        for fname in os.listdir(pcd_dir):
            assert ".pcd" in fname
            ts_list.append(fname.replace(".pcd", ""))
    assert len(ts_list) == len(set(ts_list))
    return ts_list


def write_semilabel_format(
    ts_list,
    samples,
    camera_sensors,
    origin_data_root,
    mode,
    pcr,
    save_root,
    dataset_name,
    road_id,
    extra_filter_roi,
    subfix="",
):
    final_save_dir = f"{save_root}/{dataset_name}"
    anno_save_dir = f"{final_save_dir}/annotations"
    vis_save_dir = f"{final_save_dir}/vis"
    os.makedirs(anno_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)
    sensor_data_infos = {}
    count = 0
    # for sample in tqdm(data["infos"]):
    for sample in tqdm(samples):
        lidar_path = os.path.basename(sample["lidar_path"])
        assert lidar_path.endswith(".pcd.bin")
        ts = lidar_path.replace(".pcd.bin", "")
        
        if ts not in ts_list:
            continue
        count += 1

        gt_bboxes = sample["gt_boxes"]
        gt_sublabels = sample["sublabels"]
        sensor_data_infos[ts] = {}
        for sensor in camera_sensors:
            
            if not sensor.endswith("_8"):
                assert sensor in sample["cams"]
                img_path = os.path.join(
                    origin_data_root, sample["cams"][sensor]["camera_path"]
                )
            else:
                assert sensor in sample["cams_fisheye"]
                img_path = os.path.join(
                    origin_data_root, sample["cams_fisheye"][sensor]["camera_fisheye_path"]
                )
            assert os.path.exists(img_path)
            sensor_data_infos[ts][sensor] = img_path
        
        anno_path = os.path.join(anno_save_dir, f"{ts}.txt")
        if not os.path.exists(anno_path):
            save_f = open(anno_path, "w")
            
            for i in range(gt_bboxes.shape[0]):
                x, y, z, w, l, h, yaw = gt_bboxes[i][:7]  # 注意方向
                cur_yaw = -yaw - np.pi / 2  # pkl中已经转了yaw角定义！！
                label = name_combile_dict[label_dict[index2name_map[gt_sublabels[i]]]]
                strs = f"{label} -1 -1 0.0 0.0 0.0 0.0 0.0 {h} {w} {l} {x} {y} {z} {cur_yaw} {1.0}\n"
                save_f.writelines(strs)
            save_f.close()
        sensor_data_infos[ts]["annos"] = anno_path

        # 临时可视化
        if count % 100 == 0 and count / 100 < 10:
            visualize(origin_data_root, vis_save_dir, sample, mode, pcr)
    if subfix == "":
        semi_anno_info_json = os.path.join(
            final_save_dir, "annotations_info.json"
        )
    else:
        semi_anno_info_json = os.path.join(
        final_save_dir, f"annotations_info_{subfix}.json"
    )
    with open(semi_anno_info_json, "w") as f:
        json.dump(sensor_data_infos, f, indent=4)

    if extra_filter_roi:
        satisfy_roi_region(road_id, mode.lower(), semi_anno_info_json)

def parse_semiautolable_to_bevlite(
    src_data_root,
    ann_file,
    save_root,
    dataset_name,
    mode,
    pcr,
    road_id,
    extra_filter_roi,
):
    origin_data_root = os.path.join(os.path.dirname(ann_file), "samples")
    
    assert os.path.exists(ann_file)
    assert os.path.exists(origin_data_root)
    
    data=mmcv.load(ann_file)
    camera_sensors = sorted([name for name in os.listdir(origin_data_root) if name.startswith("camera")])

    samples = data["infos"]
    # full_ts_list = collect_frame_ts_from_samples(samples)
    full_ts_list = collect_frame_ts_from_dataset_name(
        src_data_root, dataset_name
    )
    
    keep_ts_list, del_ts_list = filter_ts_within_time_interval(
        full_ts_list, time_interval=5
    )

    # 保存半自动标注数据（包括连续）
    write_semilabel_format(
        full_ts_list,
        samples,
        camera_sensors,
        origin_data_root,
        mode,
        pcr,
        save_root,
        dataset_name,
        road_id,
        extra_filter_roi,
        subfix="",
    )
    # 保存半自动单帧数据
    write_semilabel_format(
        keep_ts_list,
        samples,
        camera_sensors,
        origin_data_root,
        mode,
        pcr,
        save_root,
        dataset_name,
        road_id,
        extra_filter_roi,
        subfix="one_frame",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--data_root", type=str, required=True, help="semi-autolabel data root dir")
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--road', type=str, required=True)
    parser.add_argument(
        '--mode',
        type=str,
        default="intersection",
        choices=["intersection", "roadsection"],
        help="Prepare data for road type.",
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument(
        "--extra_filter_roi",
        action="store_true",
        help="whether filter object in roi region",
    )
    args = parser.parse_args()

    configs.load(args.config, recursive=True)

    cfg = Config(recursive_eval(configs), filename=args.config)
    pcr = cfg.pcr

    road_id = f"{args.location}_{args.road}"
    assert os.path.exists(args.data_root)
    if args.mode == "intersection":
        parse_semiautolable_to_bevlite(
            args.data_root,
            args.ann_file,
            args.save_root,
            args.dataset_name,
            args.mode,
            pcr,
            road_id,
            args.extra_filter_roi,
        )
    else:
        raise NotImplementedError(f"Currently only support Intersection")
    
    print("finished!!!")
