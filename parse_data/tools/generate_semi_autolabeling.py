import argparse
import os
import json
import math

import numpy as np

from tqdm import tqdm
from visual_data.data_show_multiprocess import confidence_type, nms_angle


def filter_ts_within_time_interval(ts_list, time_interval=5):
    """_summary_

    Args:
        ts_list (_type_): list of timestamp str.
        time_interval (int, optional): time interval length, measured
            in second. Defaults to 5.
    """
    sorted_ts = sorted(ts_list)
    kept_ts = []
    deleted_ts = []
    
    kept_ts.append(sorted_ts[0])
    last_kept_index = 0
    for i in range(1, len(sorted_ts)):
        cur_ts = sorted_ts[i]
        last_kept_ts = sorted_ts[last_kept_index]

        ts_diff = float(cur_ts) - float(last_kept_ts)
        if ts_diff >= time_interval:
            kept_ts.append(cur_ts)
            last_kept_index = i
        else:
            deleted_ts.append(cur_ts)
    
    return kept_ts, deleted_ts


def get_useful_data(src_dir, strict=False, vehicle_mode=False):
    if vehicle_mode:
        assert strict

    # os.makedirs(dst_dir, exist_ok=True)
    all_frames_count = 0
    all_count_valid = 0
    all_count_use_1 = 0
    all_count_use_1_7 = 0
    all_count_use_1_8 = 0
    all_count_use_5 = 0
    all_count_except = 0
    if vehicle_mode:
        print(f"\n{os.path.basename(src_dir)}-vehicle frames statistics: ")
    else:
        print(f"\n{os.path.basename(src_dir)}-full-class frames statistics: ")

    final_res = []
    # with open(save_file, "w") as f1:
    for file_name in sorted(os.listdir(src_dir)):
        json_path = os.path.join(src_dir, file_name)

        count_valid = 0
        count_use_1 = 0
        count_use_1_7 = 0
        count_use_1_8 = 0
        count_use_5 = 0
        count_except = 0
        with open(json_path, "r") as f:
            res = json.load(f)
        all_frames_count += len(res["list"])
        for item in res['list']:
            try:
                # if new_format:
                assert "use" in item
                if vehicle_mode:
                    if "vehicle" not in item["use"]:
                        continue
                else:
                    if "vehicle" in item["use"]:
                        continue

                use_status = set(
                    item["use"].replace("vehicle-", "").split(",")
                )  # 兼容大车模式
                if use_status == set(["1"]):
                    count_use_1 += 1
                    count_valid += 1
                    assert False
                if use_status == set(["1", "7"]):
                    count_use_1_7 += 1
                    count_valid += 1
                    # f1.write(item['id']+'\n')
                    final_res.append(item["id"])
                if use_status == set(["1", "8"]):
                    count_use_1_8 += 1
                    count_valid += 1
                    if not strict:
                        # 可根据实际情况选择是否加入
                        final_res.append(item["id"])
                if use_status == set(["5"]):  # 暂定类别
                    count_use_5 += 1
                    count_valid += 1
                    if not strict:
                        # 可根据实际情况选择是否加入
                        final_res.append(item["id"])  
            except:
                count_except += 1
                print(item['img0'].keys())
                assert False, f"bad json content"
        all_count_use_1 += count_use_1
        all_count_use_1_7 += count_use_1_7
        all_count_use_1_8 += count_use_1_8
        all_count_valid += count_valid
        all_count_use_5 += count_use_5
        all_count_except += count_except
        print(f"{file_name}: , use=1: {count_use_1}, use_valid: {count_valid}, (use=1, use3=7): {count_use_1_7}, (use=1, use3=8): {count_use_1_8}, use=5: {count_use_5}")
    
    print(f"all_frames: {all_frames_count}")
    print(f"frames(use=1): {all_count_use_1}")
    print(f"frames(use=5): {all_count_use_5}")
    print(f"valid_frames: {all_count_valid}")
    print(f"frames(use=1, use3=7): {all_count_use_1_7}")
    print(f"frames(use=1, use3=8): {all_count_use_1_8}")
    print(f"frames_error_format: {all_count_except}")

    return final_res


def write_prelabel_annos(
    ts_list,
    save_root,
    dataset_name,
    gt_root,
    subfix="",
    one_frame=False,
):
    timestamps = set(ts_list)
    if subfix == "":
        final_save_dir = f"{save_root}/{dataset_name}"
    else:
        final_save_dir = f"{save_root}/{dataset_name}_{subfix}"
    sensor_data_infos = {}
    if not one_frame:  # 使用跟踪结果
        for clip_name in tqdm(os.listdir(gt_root)):
            cur_gt_dir = os.path.join(gt_root, clip_name, "splited_nms")
            assert os.path.exists(cur_gt_dir)
            for fname in os.listdir(cur_gt_dir):
                ts = os.path.splitext(fname)[0]
                if ts in timestamps:
                    anno_save_dir = f"{final_save_dir}/{clip_name}"
                    os.makedirs(anno_save_dir, exist_ok=True)

                    

                    sensor_data_infos[ts] = {}
                    sensor_data_infos[ts]["data_root"] = origin_data_root
                    for sensor in camera_sensors:
                        img_path = os.path.join(sensor, f"{ts}.jpg")
                        assert os.path.exists(os.path.join(origin_data_root, img_path))
                        sensor_data_infos[ts][sensor] = img_path
                    for sensor in lidar_sensors:
                        pcd_path = os.path.join(sensor, f"{ts}.pcd.bin")
                        assert os.path.exists(os.path.join(origin_data_root, pcd_path))
                        sensor_data_infos[ts][sensor] = pcd_path

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
                    anno_format_res = {
                        "data": {"3d_url": fname.replace(".txt", ".pcd")},
                        "nCloud": "507485",  # TODO：临时数据!!!!
                        "result": {"data": []},
                    }
                    object_id = 0
                    for cur_box_label in boxes_label_tracked_new:
                        label, h, w, l, x, y, z, yaw, _ = cur_box_label
                        if label == "car":
                            label_name = "5000"
                            sublabel_name = "5000"
                        elif label == "truck":
                            label_name = "6000"
                            sublabel_name = "6000"
                        elif label == "bus":
                            label_name = "7000"
                            sublabel_name = "7000"
                        else:
                            continue

                        object_id += 1

                        cur_anno = {}
                        cur_anno["3Dcenter"] = {
                            "x": x,
                            "y": y,
                            "z": z
                        }
                        cur_anno["3Dsize"] = {
                            "width": w,
                            "length": l,
                            "height": h,
                            "alpha": yaw,
                            "rx": 0,
                            "ry": 0,
                            "rz": yaw
                        }
                        cur_anno["group"] = "0"
                        cur_anno["ObjectID"] = object_id
                        cur_anno["label"] = label_name
                        cur_anno["sublabel"] = sublabel_name
                        cur_anno["pointnum"] = 300  # TODO: current fake !!!!
                        cur_anno["Pseudo_3D_larger_than_60cm"] = 0
                        cur_anno["is_excessive_layering"] = "False"
                        anno_format_res["result"]["data"].append(cur_anno)


                    anno_path = os.path.join(anno_save_dir, fname.replace(".txt", ".json"))
                    with open(anno_path, "w") as f:
                        json.dump(anno_format_res, f, indent=2)
    else:  # 单帧模式
        cur_gt_dir = os.path.join(gt_root, "model_pred_nms")
        assert os.path.exists(cur_gt_dir)
        for fname in os.listdir(cur_gt_dir):
            ts = os.path.splitext(fname)[0]
            if ts in timestamps:
                anno_save_dir = f"{final_save_dir}"
                os.makedirs(anno_save_dir, exist_ok=True)

                sensor_data_infos[ts] = {}
                sensor_data_infos[ts]["data_root"] = origin_data_root
                for sensor in camera_sensors:
                    img_path = os.path.join(sensor, f"{ts}.jpg")
                    assert os.path.exists(os.path.join(origin_data_root, img_path))
                    sensor_data_infos[ts][sensor] = img_path
                for sensor in lidar_sensors:
                    pcd_path = os.path.join(sensor, f"{ts}.pcd.bin")
                    assert os.path.exists(os.path.join(origin_data_root, pcd_path))
                    sensor_data_infos[ts][sensor] = pcd_path

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

                anno_format_res = {
                    "data": {"3d_url": fname.replace(".txt", ".pcd")},
                    "nCloud": "507485",  # TODO：临时数据!!!!
                    "result": {"data": []},
                }
                object_id = 0
                for cur_box_label in boxes_label_tracked_new:
                    label, h, w, l, x, y, z, yaw, _ = cur_box_label
                    if label == "car":
                        label_name = "5000"
                        sublabel_name = "5000"
                    elif label == "truck":
                        label_name = "6000"
                        sublabel_name = "6000"
                    elif label == "bus":
                        label_name = "7000"
                        sublabel_name = "7000"
                    else:
                        continue

                    object_id += 1

                    cur_anno = {}
                    cur_anno["3Dcenter"] = {
                        "x": x,
                        "y": y,
                        "z": z
                    }
                    cur_anno["3Dsize"] = {
                        "width": w,
                        "length": l,
                        "height": h,
                        "alpha": yaw,
                        "rx": 0,
                        "ry": 0,
                        "rz": yaw
                    }
                    cur_anno["group"] = "0"
                    cur_anno["ObjectID"] = object_id
                    cur_anno["label"] = label_name
                    cur_anno["sublabel"] = sublabel_name
                    cur_anno["pointnum"] = 300  # TODO: current fake !!!!
                    cur_anno["Pseudo_3D_larger_than_60cm"] = 0
                    cur_anno["is_excessive_layering"] = "False"
                    anno_format_res["result"]["data"].append(cur_anno)


                anno_path = os.path.join(anno_save_dir, fname.replace(".txt", ".json"))
                with open(anno_path, "w") as f:
                    json.dump(anno_format_res, f, indent=2)

def convert_to_anno_platform_format():
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--new_format", action="store_true")
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="apply strict filtering condition.")
    parser.add_argument("--one_frame", action="store_true", help="whether is one_frame")

    args = parser.parse_args()

    # if args.vehicle_mode:
    assert args.strict

    assert args.new_format  # 后续只支持新格式

    raw_anno_dir = os.path.join(
        args.data_root, "labels", args.dataset_name, "raw_anno_info"
    )  # vehicle和full_cls标注信息在同一个json文件
    vehicle_clean_data_txt = os.path.join(os.path.dirname(raw_anno_dir), "useful_vehicle.txt")
    full_cls_clean_data_txt = os.path.join(os.path.dirname(raw_anno_dir), "useful.txt")
    assert os.path.exists(raw_anno_dir), f"{raw_anno_dir} not exist"

    os.makedirs(os.path.dirname(vehicle_clean_data_txt), exist_ok=True)

    vehicle_passed_ts_list = get_useful_data(
        raw_anno_dir, args.strict, vehicle_mode=True
    )
    full_cls_passed_ts_list = get_useful_data(
        raw_anno_dir, args.strict, vehicle_mode=False, 
    )

    # 过滤掉全类别通过的数据，其余用于自动标注
    vehicle_final_ts_list = list(set(vehicle_passed_ts_list) - set(full_cls_passed_ts_list))
    # 按5s间隔过滤后的
    vehicle_keep_ts_list, vehicle_del_ts_list =  filter_ts_within_time_interval(
        vehicle_final_ts_list, time_interval=5
    )
    full_cls_keep_ts_list, full_cls_del_ts_list = filter_ts_within_time_interval(
        full_cls_passed_ts_list, time_interval=5
    )  # TODO: 根据需要设置time_interval
    print(f"vehicle_final_ts_list(contain continuous): {len(vehicle_final_ts_list)}")
    print(f"vehicle_keep_ts_list: {len(vehicle_keep_ts_list)}")
    print(f"full_cls_passed_ts_list: {len(full_cls_passed_ts_list)}")
    print(f"full_cls_keep_ts_list: {len(full_cls_keep_ts_list)}")


    # 分别保存大车通过、全类别通过的预标注json文件以及时间戳txt文件
    if not os.path.exists(vehicle_clean_data_txt):
        with open(vehicle_clean_data_txt, "w") as f:
            for cur_ts in vehicle_passed_ts_list:
                f.write(cur_ts + "\n")
    if not os.path.exists(full_cls_clean_data_txt):
        with open(full_cls_clean_data_txt, "w") as f:
            for cur_ts in full_cls_passed_ts_list:
                f.write(cur_ts + "\n")

    # 保存数据大车通过连续数据
    # 获取有效数据的路径
    origin_data_root = os.path.join(
        args.data_root, f"model_res/bevpro/{args.dataset_name}/samples"
    )
    if not args.one_frame:
        gt_root = os.path.join(
            args.data_root, f"model_res/offline_tracked/{args.dataset_name}"
        )
    else:
        gt_root = os.path.join(
            args.data_root, f"model_res/bevpro/{args.dataset_name}"
        )
    assert os.path.exists(origin_data_root)
    assert os.path.exists(gt_root)
    camera_sensors = sorted([name for name in os.listdir(origin_data_root) if name.startswith("camera")])
    lidar_sensors = sorted([name for name in os.listdir(origin_data_root) if name.startswith("lidar")])

    # 1. 保存连续大车数据（去除了全类别完全通过的数据）、单帧大车数据（间隔5s）
    write_prelabel_annos(
        vehicle_final_ts_list,
        args.save_root,
        args.dataset_name,
        gt_root,
        subfix="vehicle",
        one_frame=args.one_frame,
    )
    if not args.one_frame:
        write_prelabel_annos(
            vehicle_keep_ts_list,
            args.save_root,
            args.dataset_name,
            gt_root,
            subfix="vehicle_one_frame",
            one_frame=args.one_frame,
        )
    # # 保存全类别通过数据、单帧全类别通过数据
    # write_prelabel_annos(
    #     full_cls_passed_ts_list,
    #     args.save_root,
    #     args.dataset_name,
    #     gt_root,
    #     subfix="",
    #     one_frame=args.one_frame,
    # )
    # write_prelabel_annos(
    #     full_cls_keep_ts_list,
    #     args.save_root,
    #     args.dataset_name,
    #     gt_root,
    #     subfix="one_frame",
    #     one_frame=args.one_frame,
    # )
    print("finished!!!")
