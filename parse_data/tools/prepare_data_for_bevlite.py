import argparse
import os
import json
import math

import numpy as np

from tqdm import tqdm
from visual_data.data_show_multiprocess import confidence_type, nms_angle

from BEVFUSION.tools.data_process.sensor_modules import (
    roadside_sensor_modules,
    single_pole_roadside_sensor_modules,
)
from parse_data.tools.generate_semi_autolabeling import (
    filter_ts_within_time_interval,
    get_useful_data,
)
from shapely.geometry import Point, Polygon


roi_regions = {
    "sh_6": {
        "intersection": [  # bev-local: (x1,y1,x2,y2,...    )
            [-0.043,-31.129,-9.542,-30.895,-12.074,-23.567,-14.817,-19.863,-2.218,-20.553,-0.043,-31.129,-0.043,-31.129],
            [11.077,-5.312,-0.751,-9.035,-1.220,-1.844,3.600,6.024,6.664,21.839,16.370,20.904,11.077,-5.312,11.077,-5.312],
        ],
    },
}


def satisfy_roi_region(road_id, mode, anno_info_json):
    
    assert os.path.exists(anno_info_json)
    cur_roi_regions = roi_regions[road_id][mode]
    
    with open(anno_info_json, "r") as f:
        anno_info = json.load(f)
    

    roi_polygons = []
    for cur_region in cur_roi_regions:
        assert len(cur_region) % 2 == 0
        cur_polygon = Polygon([(x, y) for x, y in zip(cur_region[0::2], cur_region[1::2])])
        roi_polygons.append(cur_polygon)
    
    res = {}
    for ts, cur_anno in anno_info.items():
        anno_path = cur_anno["annos"]

        assert os.path.exists(anno_path)
        with open(anno_path, "r") as f:
            lines = f.readlines()
        
        count = 0
        satisfied = False
        for line in lines:
            x, y, z, yaw, _ = line.strip().split(" ")[-5:]
            cur_point = Point(float(x), float(y))

            # 一个区域至少一个
            # 判断点是否在多边形内或边界上
            for polygon in roi_polygons:
                in_polygon = polygon.contains(cur_point) or polygon.touches(cur_point)
                if in_polygon:
                    count += 1
                if count >= len(roi_polygons):
                    satisfied = True
                    break
            if satisfied:
                break
        if satisfied:
            res[ts] = cur_anno
    
    save_json = anno_info_json.replace(".json", "_roi.json")
    with open(save_json, "w") as f:
        json.dump(res, f, indent=4)
        
        


def prepare_bevlite_format(
    ts_list,
    save_root,
    dataset_name,
    gt_root,
    camera_sensors,
    origin_data_root,
    subfix,
    road_id,
    mode,
    extra_filter_roi,
    one_frame=True,
):
    timestamps = set(ts_list)
    final_save_dir = f"{save_root}/{dataset_name}"
    anno_save_dir = f"{final_save_dir}/annotations"
    os.makedirs(anno_save_dir, exist_ok=True)

    sensor_data_infos = {}
    if not one_frame:
        for clip_name in tqdm(os.listdir(gt_root)):
            cur_gt_dir = os.path.join(gt_root, clip_name, "splited_nms")
            assert os.path.exists(cur_gt_dir)

            for fname in os.listdir(cur_gt_dir):
                ts = os.path.splitext(fname)[0]
                if ts in timestamps:
                    sensor_data_infos[ts] = {}
                    for sensor in camera_sensors:
                        img_path = os.path.join(sensor, f"{ts}.jpg")
                        img_path = os.path.join(origin_data_root, img_path)
                        assert os.path.exists(img_path)
                        sensor_data_infos[ts][sensor] = img_path
                    
                    anno_path = os.path.join(anno_save_dir, fname)
                    
                    if not os.path.exists(anno_path):
                        save_f = open(anno_path, "w")
                        
                        gt_path = os.path.join(cur_gt_dir, fname)
                        with open(gt_path, "r") as f:
                            lines = f.read().splitlines()

                        # 使用splited_nms的结果：
                        boxes_label_tracked_new = []
                        for line in lines:
                            obj_id, label_name, h, w, l, x, y, z, yaw, confidence = line.strip().split("\t")
                            boxes_label_tracked_new.append(
                                [label_name, h, w, l, x, y, z, yaw, confidence]
                            )

                        for cur_box_label in boxes_label_tracked_new:
                            label, h, w, l, x, y, z, yaw, _ = cur_box_label
                            strs = f"{label} -1 -1 0.0 0.0 0.0 0.0 0.0 {h} {w} {l} {x} {y} {z} {yaw} {1.0}\n"
                            save_f.writelines(strs)
                        save_f.close()
                    sensor_data_infos[ts]["annos"] = anno_path
    else:  # 单帧模式
        cur_gt_dir = os.path.join(gt_root, "model_pred_nms")
        assert os.path.exists(cur_gt_dir)
        for fname in os.listdir(cur_gt_dir):
            ts = os.path.splitext(fname)[0]
            if ts in timestamps:
                sensor_data_infos[ts] = {}
                for sensor in camera_sensors:
                    img_path = os.path.join(sensor, f"{ts}.jpg")
                    img_path = os.path.join(origin_data_root, img_path)
                    assert os.path.exists(img_path)
                    sensor_data_infos[ts][sensor] = img_path
                
                anno_path = os.path.join(anno_save_dir, fname)
                
                if not os.path.exists(anno_path):
                    save_f = open(anno_path, "w")
                    
                    gt_path = os.path.join(cur_gt_dir, fname)
                    with open(gt_path, "r") as f:
                        lines = f.read().splitlines()

                    # 使用model_pred_nms的结果：
                    boxes_label_tracked_new = []
                    for line in lines:
                        obj_id, label_name, h, w, l, x, y, z, yaw, confidence = line.strip().split("\t")
                        boxes_label_tracked_new.append(
                            [label_name, h, w, l, x, y, z, yaw, confidence]
                        )

                    for cur_box_label in boxes_label_tracked_new:
                        label, h, w, l, x, y, z, yaw, _ = cur_box_label
                        strs = f"{label} -1 -1 0.0 0.0 0.0 0.0 0.0 {h} {w} {l} {x} {y} {z} {yaw} {1.0}\n"
                        save_f.writelines(strs)
                    save_f.close()
                sensor_data_infos[ts]["annos"] = anno_path
    
    if subfix != "":
        det_info_json = os.path.join(
            final_save_dir, f"annotations_info_{subfix}.json"
        )
    else:
        det_info_json = os.path.join(final_save_dir, "annotations_info.json")
    with open(det_info_json, "w") as f:
        json.dump(sensor_data_infos, f, indent=4)
    if extra_filter_roi:
        satisfy_roi_region(road_id, mode.lower(), det_info_json)

def collect_intersection_4d_label(
    raw_4d_label_dataset_root,
    mode,
    dataset_name,
    save_root,
    strict,
    road_id,
    extra_filter_roi,
    one_frame=False,
):
    data_root = os.path.join(raw_4d_label_dataset_root, mode)
    assert os.path.exists(data_root)

    raw_anno_dir = os.path.join(
        data_root, "labels", dataset_name, "raw_anno_info"
    )
    assert os.path.exists(raw_anno_dir), f"{raw_anno_dir} not exist"
    
    vehicle_clean_data_txt = os.path.join(os.path.dirname(raw_anno_dir), "useful_vehicle.txt")
    full_cls_clean_data_txt = os.path.join(os.path.dirname(raw_anno_dir), "useful.txt")
    
    vehicle_passed_ts_list = get_useful_data(
        raw_anno_dir, strict, vehicle_mode=True
    )
    full_cls_passed_ts_list = get_useful_data(
        raw_anno_dir, strict, vehicle_mode=False
    )
    # 过滤掉全类别通过的数据，其余用于自动标注
    vehicle_final_ts_list = list(set(vehicle_passed_ts_list) - set(full_cls_passed_ts_list))
    # 按5s间隔过滤后的
    vehicle_keep_ts_list, vehicle_del_ts_list =  filter_ts_within_time_interval(
        vehicle_final_ts_list, time_interval=5
    )
    full_cls_keep_ts_list, full_cls_del_ts_list = filter_ts_within_time_interval(
        full_cls_passed_ts_list, time_interval=5
    )
    print(f"vehicle_final_ts_list(contain continuous): {len(vehicle_final_ts_list)}")
    print(f"vehicle_keep_ts_list: {len(vehicle_keep_ts_list)}")
    print(f"full_cls_passed_ts_list: {len(full_cls_passed_ts_list)}")
    print(f"full_cls_keep_ts_list: {len(full_cls_keep_ts_list)}")

    # 分别保存大车通过、全类别通过的预标注json文件以及时间戳txt文件
    with open(vehicle_clean_data_txt, "w") as f:
        for cur_ts in vehicle_passed_ts_list:
            f.write(cur_ts + "\n")
    with open(full_cls_clean_data_txt, "w") as f:
        for cur_ts in full_cls_passed_ts_list:
            f.write(cur_ts + "\n")

    # 获取有效数据的路径
    origin_data_root = os.path.join(
        data_root, f"model_res/bevpro/{dataset_name}/samples"
    )
    if not one_frame:  # 非单帧使用track的结果
        gt_root = os.path.join(data_root, f"model_res/offline_tracked/{dataset_name}")
    else:
        gt_root = os.path.join(data_root, f"model_res/bevpro/{dataset_name}")
    assert os.path.exists(origin_data_root)
    assert os.path.exists(gt_root)
    camera_sensors = sorted([name for name in os.listdir(origin_data_root) if name.startswith("camera")])
    lidar_sensors = sorted([name for name in os.listdir(origin_data_root) if name.startswith("lidar")])

    # 保存全类别通过文件信息：单帧、连续
    if not one_frame:
        prepare_bevlite_format(
            full_cls_passed_ts_list,
            save_root,
            dataset_name,
            gt_root,
            camera_sensors,
            origin_data_root,
            subfix="",
            road_id=road_id,
            mode=mode,
            extra_filter_roi=extra_filter_roi,
            one_frame=one_frame,
        )
    prepare_bevlite_format(
        full_cls_keep_ts_list,
        save_root,
        dataset_name,
        gt_root,
        camera_sensors,
        origin_data_root,
        subfix="one_frame",
        road_id=road_id,
        mode=mode,
        extra_filter_roi=extra_filter_roi,
        one_frame=one_frame,
    )

    # 保存大车通过文件信息annotations_info.json:
    if not one_frame:
        prepare_bevlite_format(
            vehicle_final_ts_list,
            save_root,
            dataset_name,
            gt_root,
            camera_sensors,
            origin_data_root,
            subfix="vehicle",
            road_id=road_id,
            mode=mode,
            extra_filter_roi=extra_filter_roi,
            one_frame=one_frame,
        )
    prepare_bevlite_format(
        vehicle_keep_ts_list,
        save_root,
        dataset_name,
        gt_root,
        camera_sensors,
        origin_data_root,
        subfix="vehicle_one_frame",
        road_id=road_id,
        mode=mode,
        extra_filter_roi=extra_filter_roi,
        one_frame=one_frame,
    )


# TODO：需要适配splite_nms！！！！
def collect_intersection_roadsection_4d_label(
    raw_4d_label_dataset_root,
    dataset_name,
    new_format,
    save_root,
    strict,
    road_id,
):
    intersection_data_root = os.path.join(
        raw_4d_label_dataset_root, "Intersection")
    roadsection_data_root = os.path.join(
        raw_4d_label_dataset_root, "RoadSection"
    )
    assert os.path.exists(intersection_data_root)
    assert os.path.exists(roadsection_data_root)

    raw_anno_dir = os.path.join(
        os.path.join(raw_4d_label_dataset_root, "IntersectionRoadsection"),
        "labels",
        # dataset_name,
        f"{dataset_name}_12v",  # TODO: 临时-hard-code
        "raw_anno_info",
    )
    assert os.path.exists(raw_anno_dir), f"{raw_anno_dir} not exist"
    clean_data_txt = os.path.join(os.path.dirname(raw_anno_dir), "useful.txt")
    get_useful_data(
        raw_anno_dir, os.path.dirname(raw_anno_dir), new_format, strict
    )


    # 获取有效数据的路径
    # dataset_name = "train_sh_4d_road_2_20250501_lx"
    intersection_origin_data_root = os.path.join(
        intersection_data_root, f"model_res/bevpro/{dataset_name}/samples"
    )
    gt_root = os.path.join(
        os.path.join(raw_4d_label_dataset_root, "IntersectionRoadsection"),
        f"model_res/offline_tracked/{dataset_name}"
    )
    assert os.path.exists(intersection_origin_data_root)
    assert os.path.exists(gt_root)


    intersection_sensors = roadside_sensor_modules[road_id]["intersection"]
    roadsection_sensors = roadside_sensor_modules[road_id]["roadsection"]
    roadsection_sinle_pole_sensors = single_pole_roadside_sensor_modules[road_id]["roadsection"]
    sensors2poles = {}
    for cur_pole, cur_sensors in roadsection_sinle_pole_sensors.items():
        for cur_sensor in cur_sensors:
            sensors2poles[cur_sensor] = cur_pole

    with open(clean_data_txt, "r") as f:
        lines = f.read().splitlines()
    timestamps = set(lines)
    final_save_dir = f"{save_root}/{dataset_name}"
    anno_save_dir = f"{final_save_dir}/annotations"
    os.makedirs(anno_save_dir, exist_ok=True)
    sensor_data_infos = {}
    for clip_name in tqdm(os.listdir(gt_root)):
        cur_gt_dir = os.path.join(gt_root, clip_name, "splited")
        for fname in os.listdir(cur_gt_dir):
            ts = os.path.splitext(fname)[0]
            if ts in timestamps:
                sensor_data_infos[ts] = {}

                # bevlite仅仅写入camera信息
                for sensor in intersection_sensors:
                    if not sensor.startswith("camera"):
                        continue
                    img_path = os.path.join(sensor, f"{ts}.jpg")
                    img_path = os.path.join(intersection_origin_data_root, img_path)
                    
                    # TODO: 临时处理7号路口北路朝路口枪机失效！！！！
                    if road_id == "hy_7" and sensor == "camera_0_0":
                        if not os.path.exists(img_path):
                            continue
                    
                    assert os.path.exists(img_path)
                    sensor_data_infos[ts][sensor] = img_path

                for sensor in roadsection_sensors:
                    if not sensor.startswith("camera"):
                        continue
                    img_path = os.path.join(sensor, f"{ts}.jpg")
                    cur_pole = sensors2poles[sensor]
                    cur_origin_data_root = os.path.join(
                        roadsection_data_root,
                        f"{cur_pole}/model_res/bevpro/{dataset_name}/samples"
                    )
                    img_path = os.path.join(cur_origin_data_root, img_path)
                    assert os.path.exists(img_path)
                    sensor_data_infos[ts][sensor] = img_path

                
                anno_path = os.path.join(anno_save_dir, fname)
                save_f = open(anno_path, "w")
                
                gt_path = os.path.join(cur_gt_dir, fname)
                with open(gt_path, "r") as f:
                    lines = f.read().splitlines()
                boxes_label_tracked = []
                for line in lines:
                    label,h,w,l,x,y,z,degree,s = line.strip().split()
                    yaw = math.radians(float(degree))
                    
                    boxes_label_tracked.append(
                        [label, h, w, l, x, y, z, yaw, confidence_type[label]]
                    )
                
                nms_prepare_inputs = [
                    [i_[4], i_[5], i_[3], i_[2], np.degrees(i_[-2]), i_[-1]]
                    for i_ in boxes_label_tracked
                ]
                _,nms_idx = nms_angle(nms_prepare_inputs, iou_thres=0.2)
                boxes_label_tracked_new  = [boxes_label_tracked[i] for i in nms_idx]

                for cur_box_label in boxes_label_tracked_new:
                    label, h, w, l, x, y, z, yaw, _ = cur_box_label
                    strs = f"{label} -1 -1 0.0 0.0 0.0 0.0 0.0 {h} {w} {l} {x} {y} {z} {yaw} {1.0}\n"
                    save_f.writelines(strs)
                save_f.close()
                sensor_data_infos[ts]["annos"] = anno_path
    
    det_info_json = os.path.join(final_save_dir, "annotations_info.json")
    with open(det_info_json, "w") as f:
        json.dump(sensor_data_infos, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_4d_label_dataset_root", type=str, required=True)
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--road', type=str, required=True)
    parser.add_argument(
        '--mode',
        type=str,
        default="Intersection",
        choices=["Intersection", "RoadSection", "IntersectionRoadsection"],
        help="Prepare data for road type.",
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--new_format", action="store_true")
    parser.add_argument(
        "--strict", action="store_true", help="whether use strict filtering"
    )
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--extra_filter_roi", action="store_true", help="whether filter object in roi region")
    parser.add_argument("--one_frame", action="store_true", help="whether is one_frame")
    args = parser.parse_args()


    # 使用最严格模式进行
    assert args.strict
    assert args.new_format  # 后续只支持新格式

    road_id = f"{args.location}_{args.road}"
    print(road_id)
    if args.extra_filter_roi:
        assert road_id in roi_regions and args.mode.lower() in roi_regions[road_id]

    if args.mode == "Intersection":
        collect_intersection_4d_label(
            args.raw_4d_label_dataset_root,
            args.mode,
            args.dataset_name,
            args.save_root,
            args.strict,
            road_id,
            args.extra_filter_roi,
            args.one_frame,
        )
    elif args.mode == "IntersectionRoadsection":
        raise NotImplementedError(f"Currently not support Intersection+Roadsection")
        collect_intersection_roadsection_4d_label(
            args.raw_4d_label_dataset_root,
            args.dataset_name,
            args.new_format,
            args.save_root,
            args.strict,
            road_id,
        )
    else:
        raise NotImplementedError(f"Currently not support roadsection-only")
    
    print("finished!!!")
