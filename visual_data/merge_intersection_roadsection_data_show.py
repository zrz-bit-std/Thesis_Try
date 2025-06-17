import argparse
import copy
import cv2
import json
import math
import os
import numpy as np

from visual_data.data_show_multiprocess import (
    confidence_type,
    draw_pcr_on_cam,
    draw_pcr_on_fisheye,
    get_palette,
    name_combile_list,
    name_old_dict,
    OBJECT_PALETTE_BEVFUSION,
    OBJECT_PALETTE_FISHYE_DETECT,
)
from visual_data.utils.nms import (
    mixed_nms,
    nms_angle,
    nms_by_dist,
    nms_by_iou,
)
from visual_data.utils.project_image import (
    draw_box_on_pinhole,
    draw_box_on_fisheye,
)
from BEVFUSION.tools.utils import (
    get_calib_params_v2,
    rotateanglexy,
    poolprocess,
    trans3dpoints,
)
from BEVFUSION.tools.visualize.get_infer_res import recursive_eval
from BEVFUSION.tools.data_process.sensor_modules import (
    single_pole_roadside_sensor_modules,
    camera_lidar_pairs,
)

from mmcv import Config
from torchpack.utils.config import configs
from tqdm import tqdm


class_score_thresholds = {
    "car": 0.3,
    "truck": 0.3,
    "bus": 0.3,
    "bike": 0.25,
    "bicycle": 0.25,
    "person": 0.25,
    "motor": 0.25,
    "motorcycle": 0.25,
    "NotUsed": 0.9,
    "rider":  0.3,
    "pedestrain": 0.25,
    "pedestrian": 0.25,
}

class_score_thresholds_distancewise = {
    "car": {(0, 60): 0.3, (60, 80): 0.2, (80, 1000): 0.15},
    "truck": {(0, 60): 0.3, (60, 80): 0.2, (80, 1000): 0.15},
    "bus": {(0, 60): 0.3, (60, 80): 0.2, (80, 1000): 0.15},
    "bike":{(0, 60): 0.25, (60, 80): 0.2, (80, 1000): 0.15},
    "bicycle": {(0, 60): 0.25, (60, 80): 0.2, (80, 1000): 0.15},
    "person": {(0, 60): 0.25, (60, 80): 0.2, (80, 1000): 0.15},
    "motor": {(0, 60): 0.25, (60, 80): 0.2, (80, 1000): 0.15},
    "motorcycle": {(0, 60): 0.25, (60, 80): 0.2, (80, 1000): 0.15},
    "NotUsed": {(0, 60): 0.9, (60, 80): 0.9, (80, 1000): 0.15},
    "rider":  {(0, 60): 0.3, (60, 80): 0.2, (80, 1000): 0.15},
    "pedestrain": {(0, 60): 0.25, (60, 80): 0.2, (80, 1000): 0.15},
    "pedestrian": {(0, 60): 0.25, (60, 80): 0.2, (80, 1000): 0.15},
}
DISTANCE_INTERVALS = list(
    class_score_thresholds_distancewise[
        list(class_score_thresholds_distancewise.keys())[0]
    ].keys()
)


def get_bev_dist_interval(x, y, distance_intervals):
    dist = (x ** 2 + y ** 2) ** 0.5
    res = distance_intervals[0]
    for dist_interval in distance_intervals:
        if dist >= dist_interval[0] and dist < dist_interval[1]:
            res = dist_interval
            break
    return res


def parse_roadsection_lidarname(road_id, pole):
    assert pole in ["S0", "S1", "S2", "S3"]
    modules = (
        single_pole_roadside_sensor_modules[road_id]["roadsection"][pole]
    )
    
    lidar_names = [name for name in modules if name.startswith("lidar")]
    assert len(lidar_names) == 1
    return lidar_names[0]

def parse_pole_by_camera_name(road_id, camera_name):
    single_pole_modules = (
        single_pole_roadside_sensor_modules[road_id]["roadsection"]
    )
    module2pole = {}
    for pole, modules in single_pole_modules.items():
        for module in modules:
            module2pole[module] = pole
    return module2pole[camera_name]


def parse_tracked_boxes(
    frame_tracked_path,
    is_percept=False,
    param_info=None,
    lidar_name=None,
):
    boxes_label_tracked = []
    boxes_label_tracked_degree_yaw = []
    with open(frame_tracked_path, "r") as f:
        for line_ in f.readlines():
            content_list = line_.strip().split('\t')
            h,w,l,x,y,z,yaw,confidence = [float(i) for i in content_list[1:]] #c, h, w, l, new_center[0], new_center[1], new_center[2], yaw_new, confidence
            # 路段需要转换到local坐标系, percept2local
            yaw = math.radians(yaw)
            if is_percept:
                yaw = -1 * (math.pi / 2.0) - yaw  # 保存txt时做了改变换
                trans_mat = np.linalg.inv(
                    param_info["lidar"][lidar_name]["local2percept"]
                )
                x, y = trans3dpoints(np.array([[x, y, 0]]), trans_mat)[0, :2]
                yaw = rotateanglexy(yaw, trans_mat)
                yaw = -1 * (math.pi / 2.0) - yaw  # 转到bev-local
            label_name = name_combile_list[name_old_dict[content_list[0]]]
            cur_dist_interval = get_bev_dist_interval(x, y, DISTANCE_INTERVALS)
            # if confidence >= class_score_thresholds[label_name]:
            if confidence >= class_score_thresholds_distancewise[label_name][cur_dist_interval]:
                boxes_label_tracked.append([label_name,h,w,l,x,y,z,yaw, confidence])# cx, cy, l, w, r
                boxes_label_tracked_degree_yaw.append([label_name,h,w,l,x,y,z,math.degrees(yaw), confidence])# cx, cy, l, w, r

    if len(boxes_label_tracked) > 0:
        nms_prepare_inputs = [[i_[4],i_[5],i_[3],i_[2],np.degrees(i_[-2]),i_[-1]] for i_ in boxes_label_tracked]
        _,nms_idx = nms_angle(nms_prepare_inputs,iou_thres=0.2)
        boxes_label_tracked_new  = [boxes_label_tracked[i] for i in nms_idx]
        boxes_label_tracked_degree_yaw_new  = [boxes_label_tracked_degree_yaw[i] for i in nms_idx]
    else:
        boxes_label_tracked_new = []
        boxes_label_tracked_degree_yaw_new = []

    return boxes_label_tracked_new, boxes_label_tracked_degree_yaw_new


def worker(args):
    (
        intersection_clip_tracked_dir,
        roadsection_clip_tracked_dir,
        intersection_clip_merged_dir,
        roadsection_clip_merged_dir,
        frame_name,
        save_path,
        intersection_test_json,
        roadsection_test_json,
        intersection_img_pinhole_list,
        roadsection_img_pinhole_list,
        img_fisheye_list,
        clip_origin_dir,
        image_save_size,
        color_bar,
        intersection_pcr,
        roadsection_pcr,
        road_id,
        poles,
        param_info,
        merged_res_clip_dir,
        # src_dir,
        # dataset_name,
        # clip_name,
    ) = args

    intersection_frame_tracked_name = os.path.join(
        intersection_clip_tracked_dir, frame_name
    )  # f"{timestamp}.txt"
    roadsection_frame_tracked_name = {
        pole: os.path.join(roadsection_clip_tracked_dir[pole], frame_name)
        for pole in poles
    }
    save_image_name = os.path.join(
        save_path, frame_name.replace(".txt", ".jpg")
    )

    (
        intersection_boxes_label_tracked, 
        intersection_boxes_label_tracked_degree_yaw,
    )= parse_tracked_boxes(
        intersection_frame_tracked_name, is_percept=False
    )  # (label_name, h, w, l, x, y, z, yaw, conf)
    roadsection_boxes_label_tracked_degree_yaw = {
        pole: parse_tracked_boxes(
            roadsection_frame_tracked_name[pole],
            is_percept=True,
            param_info=param_info,
            lidar_name=parse_roadsection_lidarname(road_id, pole)
        )[1] for pole in poles
    }

    # 路口和路段nms合并
    all_boxes = copy.deepcopy(intersection_boxes_label_tracked_degree_yaw)
    for pole in poles:
        all_boxes.extend(roadsection_boxes_label_tracked_degree_yaw[pole])  # (n, 9)
    
    # # TODO: 按中心点距离nms
    # xyz_confs = [[box[4], box[5], box[6], box[-1]] for box in all_boxes]
    # keep_inds = nms_by_dist(np.array(xyz_confs), dist_thresh=2.0)
    
    # # # 仅仅iou
    # boxes4iou = [
    #     [box[4], box[5], box[6], box[3], box[2], box[1], box[7], box[8]]
    #     for box in all_boxes
    # ]
    # keep_inds = nms_by_iou(np.array(boxes4iou), iou_thresh=0.2)

    # boxes_label = [all_boxes[i] for i in keep_inds]
    
    # 混合：大目标使用dist, 小目标使用iou
    boxes4mixednms = [
        [box[4], box[5], box[6], box[3], box[2], box[1], box[7], box[8]]
        for box in all_boxes
    ]
    label_names = [box[0] for box in all_boxes]
    if len(boxes4mixednms) > 1:
        keep_inds = mixed_nms(
            np.array(boxes4mixednms), label_names, iou_thresh=0.2, dist_thresh=2.0
        )
        boxes_label = [all_boxes[i] for i in keep_inds]
    else:
        boxes_label = all_boxes

    
    # # no_nms
    # boxes_label = all_boxes 

    # 保存合并后的整体目标！！！！
    merge_results_save_name = os.path.join(merged_res_clip_dir, frame_name)
    with open(merge_results_save_name, "w") as f:
        for save_list_item in boxes_label:
            f.write('\t'.join([str(i) for i in save_list_item])+'\n')

    # 路口枪机
    intersection_lidar2cam = intersection_test_json[0]["lidar2cam"]
    intersection_cam2img = intersection_test_json[0]["cam2img"]
    image_res_list = []
    for img_name in intersection_img_pinhole_list:
        if img_name == 'camera_0_0' and img_name not in intersection_cam2img: # TODO: temp hard code for mising camera_0_0
            image_res_list.append(np.zeros((800,800,3),np.uint8))
            continue
        img = cv2.imread(
            os.path.join(clip_origin_dir, img_name, frame_name.replace(".txt", ".jpg"))
        )
        camera_matrix = np.array(intersection_cam2img[img_name])[:3,:3]
        camera_extrinsic = np.array(intersection_lidar2cam[img_name])
        for label in boxes_label:
            class_name_label,h,w,l,x,y,z,yaw,confidence = label
            draw_box_on_pinhole(
                (w, l, h, x, y, z),
                math.radians(yaw),
                img,
                camera_matrix,
                camera_extrinsic,
                color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1],
            )
        draw_pcr_on_cam(
            "intersection",
            road_id,
            img,
            intersection_pcr,
            camera_matrix,
            camera_extrinsic,
            None,
            img_name,
            color=(255, 0, 0),
        )
        img = cv2.resize(img, image_save_size)
        image_res_list.append(img)
    image_pinhole = np.concatenate(image_res_list, axis=1)

    # 鱼眼
    lidar2cam_fisheye = intersection_test_json[0]["lidar2cam_fisheye"]
    cam2img_fisheye = intersection_test_json[0]["cam2img_fisheye"]
    distort_fisheye = intersection_test_json[0]['distort_fisheye']
    image_res_list = []
    for img_name in img_fisheye_list:
        img = cv2.imread(
            os.path.join(
                clip_origin_dir, img_name, frame_name.replace('txt','jpg')
            )
        )
        camera_matrix = np.array(cam2img_fisheye[img_name])[:3,:3]
        camera_extrinsic = np.array(lidar2cam_fisheye[img_name])
        camera_distort_param = np.array(distort_fisheye[img_name])
        for label in boxes_label:
                class_name_label,h,w,l,x,y,z,yaw,confidence = label
                # yaw = -1*math.pi/2.0 - math.radians(yaw)
                draw_box_on_fisheye(
                    (w,l,h,x,y,z),
                    math.radians(yaw),
                    img,
                    camera_matrix,
                    camera_extrinsic,
                    distort=camera_distort_param,
                    color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1],
                )
        # 可视化pcr边界
        draw_pcr_on_fisheye(
            "intersection",
            road_id,
            img,
            intersection_pcr,
            camera_matrix,
            camera_extrinsic,
            camera_distort_param,
            img_name,
            color=(255, 0, 0),
        )
        img = cv2.resize(img, image_save_size)
        image_res_list.append(img)
    image_fisheye = np.concatenate(image_res_list, axis=1)

    # 路段枪机
    image_res_list = []
    for img_name in roadsection_img_pinhole_list:
        cur_pole = parse_pole_by_camera_name(road_id, img_name)
        roadsection_lidar2cam = roadsection_test_json[cur_pole][0]["lidar2cam"]  # TODO：这里存储了percept2cam, 需要转到local2cam
        roadsection_cam2img = roadsection_test_json[cur_pole][0]["cam2img"]
        img = cv2.imread(
            os.path.join(clip_origin_dir, img_name, frame_name.replace(".txt", ".jpg"))
        )
        camera_matrix = np.array(roadsection_cam2img[img_name])[:3,:3]
        cur_lidarname = camera_lidar_pairs[img_name]
        local2percept = param_info["lidar"][cur_lidarname]["local2percept"]
        camera_extrinsic = np.array(
            np.array(roadsection_lidar2cam[img_name]) @ local2percept
        )  # !!! 需要转换local2cam, 因boxes已经转到local！！！
        for label in boxes_label:
            class_name_label,h,w,l,x,y,z,yaw,confidence = label
            draw_box_on_pinhole(
                (w, l, h, x, y, z),
                math.radians(yaw),
                img,
                camera_matrix,
                camera_extrinsic,
                color=OBJECT_PALETTE_FISHYE_DETECT[class_name_label][::-1],
            )
        # TODO: 支持画路段pcr边界区域 ！！！！
        # 对路段主要关注当前车道，内缩x, pcr: [-16.8, -120, -4.0, 16.8, -9.6, 4.0]
        roi_roadsection_pcr = [
            roadsection_pcr[0] + 4.0,
            roadsection_pcr[1] + 4.0,
            roadsection_pcr[2],
            roadsection_pcr[3] - 4.0,
            roadsection_pcr[4],
            roadsection_pcr[5],
        ]
        draw_pcr_on_cam(
            "roadsection",
            road_id,
            img,
            roi_roadsection_pcr,
            camera_matrix,
            np.array(roadsection_lidar2cam[img_name]),  # percept
            None,
            img_name,
            color=(255, 0, 0),
        )

        img = cv2.resize(img, image_save_size)
        image_res_list.append(img)
    roadsection_image_pinhole = np.concatenate(image_res_list, axis=1)

    image_concated = np.concatenate(
        [image_pinhole, image_fisheye, roadsection_image_pinhole], axis=0
    )
    image_concated = np.concatenate([image_concated, color_bar], axis=1)

    cv2.imwrite(save_image_name, image_concated)


def process_one_dataset(
    road_id,
    calib,
    calib_config_file,
    calib_version,
    src_dir,
    dataset_name,
    intersection_pcr,
    roadsection_pcr,
    pnum,
):

    # road_id = f"{args.location}_{args.road}"
    param_info = get_calib_params_v2(
        calib,
        # "./BEVFUSION/tools/data_process/calibration_configs/calib_file.yaml",
        calib_config_file,
        road_id,
        calib_version,
    )

    # src_dir = "/data1/turbo_data/huben/data/test_4D_label"
    # dataset_name = "train_hy_4d_road_7_20250304_1x_tmp"
    origin_data_root = os.path.join(src_dir, "origin")

    # intersection_pcr = [-102, -102, -4.0, 102, 102, 4.0]
    # roadsection_pcr = [-16.8, -120, -4.0, 16.8, -9.6, 4.0]

    image_save_size = (800, 800)
    color_bar = get_palette((800 * 3,350))  # cam + fisheye + cam
    
    # 临时标准十字路口, 注意10号路口有顺序反了！！！！
    intersection_img_pinhole_list = [
        'camera_0_0','camera_1_0','camera_2_0','camera_3_0',
    ]
    roadsection_img_pinhole_list = [
        'camera_0_1','camera_1_1','camera_2_1','camera_3_1',
    ]
    img_fisheye_list = ['camera_0_8','camera_1_8','camera_2_8','camera_3_8']

    # track results
    dataset_origin_dir = os.path.join(origin_data_root, dataset_name)
    intersection_dataset_label_dir = os.path.join(
        src_dir, "Intersection", "labels", dataset_name
    )
    intersection_dataset_tracked_dir = os.path.join(
        src_dir, "Intersection", "model_res", "offline_tracked", dataset_name
    )
    intersection_dataset_bevpro_dir = os.path.join(
        src_dir, "Intersection", "model_res", "bevpro", dataset_name
    )
    intersection_dataset_merged_dir = os.path.join(
        src_dir, "Intersection", "model_res", "merged", dataset_name
    )
    intersection_test_json_path = os.path.join(
        intersection_dataset_bevpro_dir, "scences","test.json"
    )
    # roadsection label stored in S0,S1,S2,S3 subdir
    poles = ["S0", "S1", "S2", "S3"]
    roadsection_dataset_label_dir = {
        pole: os.path.join(
            src_dir, "RoadSection", pole, "labels", dataset_name
        ) for pole in poles 
    }
    roadsection_dataset_tracked_dir = {
        pole: os.path.join(
            src_dir,
            "RoadSection",
            pole,
            "model_res",
            "offline_tracked",
            dataset_name,
        ) for pole in poles
    }
    roadsection_dataset_bevpro_dir = {
        pole: os.path.join(
            src_dir, "RoadSection", pole, "model_res", "bevpro", dataset_name
        ) for pole in poles
    }
    roadsection_dataset_merged_dir = {
        pole: os.path.join(
            src_dir, "RoadSection", pole, "model_res", "merged", dataset_name
        ) for pole in poles
    }
    roadsection_test_json_path = {
        pole: os.path.join(
            roadsection_dataset_bevpro_dir[pole], "scences", "test.json"
        ) for pole in poles
    }

    # 合并后存放路径
    save_path = os.path.join(
        src_dir, "IntersectionRoadsection", "labels", dataset_name, "selected"
    )
    os.makedirs(save_path, exist_ok=True)
    
    with open(intersection_test_json_path, "r") as f:
        intersection_test_json = json.load(f)
    roadsection_test_json = {}
    for pole in poles:
        with open(roadsection_test_json_path[pole], "r") as f:
            roadsection_test_json[pole] = json.load(f)
    
    clip_list = os.listdir(intersection_dataset_tracked_dir)  # 确保路口路段列表相同
    datas = []
    for clip_name in tqdm(clip_list):
        clip_origin_dir = os.path.join(dataset_origin_dir, clip_name)
        intersection_clip_tracked_dir = os.path.join(
            intersection_dataset_tracked_dir, clip_name, "splited"
        )
        roadsection_clip_tracked_dir = {
            pole: os.path.join(
                roadsection_dataset_tracked_dir[pole], clip_name, "splited")
            for pole in poles
        }
        intersection_clip_merged_dir = os.path.join(
            intersection_dataset_merged_dir, clip_name
        )
        roadsection_clip_merged_dir = {
            pole: os.path.join(
                roadsection_dataset_merged_dir[pole], clip_name
            ) for pole in poles
        }
        frame_list = os.listdir(intersection_clip_tracked_dir)  # TODO: 确保路口路段数据帧数相同！！！
        merged_splited_clip_res_dir = os.path.join(
            src_dir,
            f"IntersectionRoadsection/model_res/offline_tracked",
            f"{dataset_name}/{clip_name}/splited",
        )
        os.makedirs(merged_splited_clip_res_dir, exist_ok=True)
        for frame_name in tqdm(frame_list):
            datas.append(
                [
                    intersection_clip_tracked_dir,
                    roadsection_clip_tracked_dir,
                    intersection_clip_merged_dir,
                    roadsection_clip_merged_dir,
                    frame_name,
                    save_path,
                    intersection_test_json,
                    roadsection_test_json,
                    intersection_img_pinhole_list,
                    roadsection_img_pinhole_list,
                    img_fisheye_list,
                    clip_origin_dir,
                    image_save_size,
                    color_bar,
                    intersection_pcr,
                    roadsection_pcr,
                    road_id,
                    poles,
                    param_info,
                    merged_splited_clip_res_dir,
                ]
            )
    
    poolprocess(datas, worker, pnum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("intersection_config", metavar="FILE")
    parser.add_argument("roadsection_config", metavar="FILE")
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--road', type=str, required=True)
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--calib_version", type=str, required=True)
    parser.add_argument('--pnum', type=int, default=30)
    args = parser.parse_args()
    configs.load(args.intersection_config, recursive=True)
    intersection_cfg = Config(
        recursive_eval(configs), filename=args.intersection_config
    )
    intersection_pcr = intersection_cfg.pcr
    configs.clear()

    configs.load(args.roadsection_config, recursive=True)
    roadsection_cfg = Config(
        recursive_eval(configs), filename=args.roadsection_config
    )
    roadsection_pcr = roadsection_cfg.pcr

    road_id = f"{args.location}_{args.road}"
    calib_config_file = "./BEVFUSION/tools/data_process/calibration_configs/calib_file.yaml"

    process_one_dataset(
        road_id,
        args.calib,
        calib_config_file,
        args.calib_version,
        args.src_dir,
        args.dataset_name,
        intersection_pcr,
        roadsection_pcr,
        args.pnum,
    )




