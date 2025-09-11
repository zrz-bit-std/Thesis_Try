from mmdet.datasets import DATASETS
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionConfig


from class_mapping import class_name2refine_name, name2id_map, class_mapping_7


import os
import argparse
import glob
import json
import pyquaternion 
import numpy as np
import torch
from os import path as osp
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

import mmcv


class MogoDetectionConfig(DetectionConfig):
    """ Data class that specifies the detection evaluation settings. """

    def __init__(
        self,
        class_range: Dict[str, int],
        dist_fcn: str,
        dist_ths: List[float],
        dist_th_tp: float,
        min_recall: float,
        min_precision: float,
        max_boxes_per_sample: int,
        mean_ap_weight: int,
    ):

        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()

class MogoDetectionBox(EvalBox):
    """Detection box used for (de-)serialization of predictions/GT."""

    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_translation: Tuple[float, float, float] = (0, 0, 0),
        num_pts: int = -1,
        detection_name: str = "car",
        detection_score: float = -1.0,
        attribute_name: str = "",
    ):
        super().__init__(
            sample_token, translation, size, rotation, velocity, ego_translation, num_pts
        )
        assert detection_name is not None, "detection_name cannot be empty"
        assert isinstance(detection_score, float) and not np.isnan(
            detection_score
        ), "detection_score must be float"
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name or ""

    def serialize(self) -> dict:
        return {
            "sample_token": self.sample_token,
            "translation": self.translation,
            "size": self.size,
            "rotation": self.rotation,
            "velocity": self.velocity,
            "ego_translation": self.ego_translation,
            "num_pts": self.num_pts,
            "detection_name": self.detection_name,
            "detection_score": self.detection_score,
            "attribute_name": self.attribute_name,
        }

    @classmethod
    def deserialize(cls, content: dict):
        return cls(
            sample_token=content["sample_token"],
            translation=tuple(content["translation"]),
            size=tuple(content["size"]),
            rotation=tuple(content["rotation"]),
            velocity=tuple(content["velocity"]),
            ego_translation=tuple(content.get("ego_translation", (0.0, 0.0, 0.0))),
            num_pts=int(content.get("num_pts", -1)),
            detection_name=content["detection_name"],
            detection_score=float(content.get("detection_score", -1.0)),
            attribute_name=content.get("attribute_name", ""),
        )

@DATASETS.register_module()
class MogoDataset_evalrefine(torch.utils.data.Dataset):
    r"""Mogo Dataset.
    Args:
        data_root (str): Path of dataset root.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
    """
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }

    def __init__(
        self,
        data_root=None,
        sequence=None,
        load_interval=1,
        input_filter=None,
        flip_coord = False,
        class_mapping=None,
        classname2id=None,
        ignore_class_id=255,
        center_limit_range=[-102, -102, -4.0, 102, 102, 4.0],
        intersection_mode=True, 
        verbose=True
    ) -> None:
        self.load_interval = load_interval

        self.class_mapping = class_mapping
        self.classname2id = classname2id
        self.ignore_class_id = ignore_class_id
        self.center_limit_range = center_limit_range

        super().__init__()
        self.cat2id = classname2id
        self.CLASSES = tuple(
            [
                name for name, id in classname2id.items()
                if id != self.ignore_class_id
            ]
        )
        self.class_names = tuple(
            [
                self.classname2id[name] for name in self.CLASSES
            ]

        )
        self.input_filter = input_filter
        self.flip_coord = flip_coord
        self.verbose = verbose
        
        self.intersection_mode = intersection_mode
        self.gt_paths_root = "/rss/lishuaiyin/4D_label/origin/train_sh_3d_road_2_20250531_5000_lx/result"
        self.output_dir = "./output/"
        sequence_path = os.path.join(data_root, sequence)
        self.data_infos = {}
        self.load_annotations(sequence_path)
    
    def sort_gt_paths_by_timestamp(self, gt_paths, timestamps):
        sorted_gt_paths = []
        
        for timestamp in timestamps:
            # Find the gt_path corresponding to the current timestamp (assuming it's in the file name)
            matching_gt_paths = [gt_path for gt_path in gt_paths if os.path.basename(gt_path).replace(".json", "") == timestamp.replace(".txt", "")]
            
            # If a match is found, use the first match and remove it from the list
            if matching_gt_paths:
                sorted_gt_paths.append(matching_gt_paths[0])
                gt_paths.remove(matching_gt_paths[0])
        
        return sorted_gt_paths
    def load_annotations(self, sequence_path):
        # 获取所有的pred文件
        pred_root = "/rss/lishuaiyin/4D_label/dataset_track/merged/train_sh_3d_road_2_20250531_5000_lx"
        pred_scene_paths = sorted(os.listdir(pred_root))
        pred_res = []
        for pred_scene_path in pred_scene_paths:
            pred_files = sorted(os.listdir(os.path.join(pred_root, pred_scene_path)))
            pred_res.extend([os.path.join(pred_root, pred_scene_path, f) for f in pred_files])
        
        # 获取所有的tracking文件（从splited目录）
        tracking_root = "/rss/lishuaiyin/4D_label/dataset_track/offline_tracked/train_sh_3d_road_2_20250531_5000_lx"
        tracking_scene_paths = sorted(os.listdir(tracking_root))
        tracking_res = []
        for tracking_scene_path in tracking_scene_paths:
            splited_path = os.path.join(tracking_root, tracking_scene_path, "splited")
            if os.path.exists(splited_path):
                splited_files = sorted(os.listdir(splited_path))
                tracking_res.extend([os.path.join(splited_path, f) for f in splited_files])
        
        # 获取所有的droped文件
        droped_root = "/rss/lishuaiyin/4D_label/dataset_track/droped/train_sh_3d_road_2_20250531_5000_lx"
        droped_scene_paths = []
        if os.path.exists(droped_root):
            droped_scene_paths = sorted(os.listdir(droped_root))
        droped_res = []
        for droped_scene_path in droped_scene_paths:
            droped_path = os.path.join(droped_root, droped_scene_path)
            if os.path.exists(droped_path):
                droped_files = sorted(os.listdir(droped_path))
                droped_res.extend([os.path.join(droped_path, f) for f in droped_files])
        
        # 创建一个字典来存储droped文件，便于快速查找
        droped_files_dict = {os.path.basename(f): f for f in droped_res}
        
        # 对于gt，我们仍然使用原来的逻辑
        scenes = os.listdir(sequence_path)
        scenes_path = sorted([os.path.join(sequence_path, scene) for scene in scenes])
        
        for scene_path in scenes_path:
            scene = scene_path.split("/")[-1]
            timestamps = sorted(os.listdir(os.path.join(scene_path, "splited")))
            gt_paths = sorted(glob.glob(os.path.join(self.gt_paths_root, "*", scene, "result_json", "*.json")))
            gt_paths_sorted = self.sort_gt_paths_by_timestamp(gt_paths, timestamps)
            
            # 现在我们有3个列表：tracking_res, pred_res, gt_paths_sorted
            # 由于tracking_res和pred_res是全局的，我们需要根据timestamps筛选对应的部分
            # 假设文件名可以匹配
            filtered_tracking_res = []
            filtered_pred_res = []
            filtered_droped_res = []  # 用于存储对应的droped文件
            
            for timestamp in timestamps:
                # 在tracking_res中查找匹配的文件
                matching_tracking = [tr for tr in tracking_res if os.path.basename(tr) == timestamp]
                if matching_tracking:
                    filtered_tracking_res.append(matching_tracking[0])
                else:
                    # 如果没有找到tracking文件，添加一个空路径
                    filtered_tracking_res.append(None)
                
                # 在pred_res中查找匹配的文件
                matching_pred = [pr for pr in pred_res if os.path.basename(pr) == timestamp]
                if matching_pred:
                    filtered_pred_res.append(matching_pred[0])
                else:
                    # 如果没有找到pred文件，添加一个空路径
                    filtered_pred_res.append(None)
                
                # 在droped_files_dict中查找匹配的文件
                if timestamp in droped_files_dict:
                    filtered_droped_res.append(droped_files_dict[timestamp])
                else:
                    # 如果没有找到droped文件，添加None
                    filtered_droped_res.append(None)
            
            # Check if lengths match
            if not (len(filtered_tracking_res) == len(filtered_pred_res) == len(gt_paths_sorted)):
                print(f"Length mismatch in scene {scene}:")
                print(f"tracking_res: {len(filtered_tracking_res)}")
                print(f"pred_res: {len(filtered_pred_res)}")
                print(f"gt_paths: {len(gt_paths_sorted)}")
                # 不再使用断点，而是继续处理可以匹配的部分
            # 使用能够匹配上的数据
            min_len = min(len(filtered_tracking_res), len(filtered_pred_res), len(gt_paths_sorted))
            trimmed_tracking_res = filtered_tracking_res[:min_len]
            trimmed_pred_res = filtered_pred_res[:min_len]
            trimmed_droped_res = filtered_droped_res[:min_len]
            trimmed_gt_paths_sorted = gt_paths_sorted[:min_len]
            
            # 修改parse_ann_info调用，传入droped文件列表
            self.parse_ann_info(trimmed_tracking_res, trimmed_pred_res, trimmed_gt_paths_sorted, trimmed_droped_res)
        if self.verbose:
            print(f"[IoU Eval] GT samples: {len(self.data_infos.keys())}")
    
    def load_pred_txt(self, sample_token, file_path, refine=False, pred=True):
        """
        """
        all_annotations = EvalBoxes()

        with open(file_path, 'r') as f:
            boxes = []
            for line in f:
                vals = line.strip().split()
                if not refine:
                    class_name = self.classname2id[self.class_mapping[vals[0]]]
                else:
                    class_name = self.classname2id[vals[0]]
                if class_name == 255:
                    continue
                vals = list(map(float, vals[1:]))
                h, l, w, x, y, z, yaw, confidence = vals
                yaw = np.radians(yaw)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=yaw)
                boxes.append(
                    MogoDetectionBox(
                        sample_token=sample_token,
                        translation=(x, y, z),
                        size=(w, l, h) ,#if pred else (l, w, h),
                        # size=(l, w, h),
                        rotation=quat.q,
                        velocity=(0.0, 0.0),
                        num_pts=-1,
                        detection_name=class_name,
                        detection_score=confidence,
                        attribute_name="",
                    )
                )
            all_annotations.add_boxes(sample_token, boxes)
        boxes = self.filter_eval_bbox(all_annotations)
        return boxes
    
    def load_gt_json(self, sample_token, file_path):
        """
        """
        all_annotations = EvalBoxes()
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            data = json_data['result']['data']
            boxes = []
            for item in data:
                x, y, z = item['3Dcenter'].values()
                w, l, h, yaw, _, _, _ = item['3Dsize'].values()
                class_name_ = class_mapping_7[item["sublabel"]]
                class_name = self.classname2id[self.class_mapping[class_name_]]
                if class_name == 255:
                    continue
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=yaw)
                boxes.append(
                    MogoDetectionBox(
                        sample_token=sample_token,
                        translation=(x, y, z),
                        size=(l, w, h),
                        # size=(w, l, h),
                        rotation=quat.q,
                        velocity=(0.0, 0.0),
                        num_pts=item["pointnum"],
                        detection_name=class_name,
                        detection_score=-1.0,
                        attribute_name="",
                    )
                )
            all_annotations.add_boxes(sample_token, boxes)
        boxes = self.filter_eval_bbox(all_annotations)
        return boxes
    
    def filter_eval_bbox(self, eval_boxes: EvalBoxes):
        total, kept = 0, 0
        x1, y1, z1, x2, y2, z2 = self.center_limit_range
        for token in list(eval_boxes.sample_tokens):
            old_len = len(eval_boxes[token])
            total += old_len
            eval_boxes.boxes[token] = [
                b
                for b in eval_boxes[token]
                if (
                    x1 <= b.translation[0] <= x2
                    and y1 <= b.translation[1] <= y2
                    and z1 <= b.translation[2] <= z2
                    and all(s > 0 for s in b.size)
                )
            ]
            kept += len(eval_boxes[token])
        # if self.verbose:
        #     print(f"[IoU Eval] Filter by range: {total} -> {kept}")
        return eval_boxes

    def parse_ann_info(self, tracking_res, pred_res, gt_paths, droped_res=None):
        r"""
        Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - [class_name, h, w, l, x, y, z, yaw-degrees, score]
        """
        if droped_res is None:
            droped_res = [None] * len(tracking_res)
            
        if self.verbose:
            print("[IoU Eval] Converting GT to EvalBoxes...")

        for i in tqdm(range(0, len(tracking_res), self.load_interval)):
            sample = {}
            token = os.path.basename(tracking_res[i] or pred_res[i]).replace(".txt", "") if tracking_res[i] else os.path.basename(pred_res[i]).replace(".txt", "")
            sample["token"] = token

            # Load different types of bounding boxes
            sample["tracking_bboxes"] = self.load_pred_txt(token, tracking_res[i], refine=False, pred=False) if tracking_res[i] and os.path.exists(tracking_res[i]) else EvalBoxes()
            
            # 加载检测结果
            sample["detection_bboxes"] = self.load_pred_txt(token, pred_res[i], refine=False, pred=True) if pred_res[i] and os.path.exists(pred_res[i]) else EvalBoxes()
            
            # 加载真值
            sample["gt_bboxes"] = self.load_gt_json(token, gt_paths[i]) if os.path.exists(gt_paths[i]) else EvalBoxes()
            
            # 创建合并的跟踪+drop结果
            sample["tracking_with_drop_bboxes"] = self.load_pred_txt(token, tracking_res[i], refine=False, pred=False) if tracking_res[i] and os.path.exists(tracking_res[i]) else EvalBoxes()
            
            # 如果有对应的droped文件，将其内容合并到tracking_with_drop_bboxes中
            if droped_res[i] and os.path.exists(droped_res[i]):
                droped_bboxes = self.load_pred_txt(token, droped_res[i], refine=False, pred=False)
                # 合并tracking_bboxes和droped_bboxes
                for droped_token in droped_bboxes.sample_tokens:
                    if droped_token in sample["tracking_with_drop_bboxes"].boxes:
                        sample["tracking_with_drop_bboxes"].boxes[droped_token].extend(droped_bboxes.boxes[droped_token])
                    else:
                        sample["tracking_with_drop_bboxes"].boxes[droped_token] = droped_bboxes.boxes[droped_token]

            self.data_infos[token] = sample
    def evaluate(
        self,
        eval_metric: str = "iou",  # 'iou' or 'center_distance'
        eval_boxes_types: list = None,  # 可选 ["detection_bboxes", "tracking_bboxes", "tracking_with_drop_bboxes"] 等
        **kwargs
    ):
        """
        根据 eval_boxes_types 参数评估指定类型的 bounding boxes。
        支持的评估类型:
        - 'detection_bboxes': 检测结果框
        - 'tracking_bboxes': 跟踪结果框
        - 'tracking_with_drop_bboxes': 跟踪结果框+drop框
        """
        metrics = {}

        # 默认评估所有支持的类型
        if eval_boxes_types is None:
            eval_boxes_types = ["detection_bboxes", "tracking_bboxes", "tracking_with_drop_bboxes"]
        
        # 确保 eval_boxes_types 中的类型都是支持的类型
        supported_box_types = ["detection_bboxes", "tracking_bboxes", "tracking_with_drop_bboxes"]
        for box_type in eval_boxes_types:
            if box_type not in supported_box_types:
                raise ValueError(f"Unsupported box type: {box_type}. "
                                f"Supported types: {supported_box_types}")

        # 依次评估每种类型
        for eval_boxes_type in eval_boxes_types:
            res_name = eval_boxes_type.replace("_bboxes", "")
            print(f"Evaluating bboxes of {res_name}")

            # 提取指定类型的预测框和 GT 框
            pred_bboxes = [item[eval_boxes_type] for key, item in self.data_infos.items()]
            gt_bboxes = [item["gt_bboxes"] for key, item in self.data_infos.items()]

            # 执行 IoU 评估
            self._main_iou(gt_bboxes, pred_bboxes)

        return metrics

    # ---------- IoU evaluation core ----------
    def _main_iou(self, gt_boxes, pred_boxes):
        """
        计算每类在 IoU 阈值 0.5 和 0.7 下的 AP，并保存/打印。
        """
        iou_ths = [0.5, 0.7]  # 只评两个阈值
        class_names = list(self.class_names)
        label_aps = {cls: {} for cls in class_names}

        # 逐类、逐样本、逐阈值累加 TP/FP
        for cls in class_names:
            per_thr_tp = {thr: [] for thr in iou_ths}
            per_thr_fp = {thr: [] for thr in iou_ths}
            per_thr_sc = {thr: [] for thr in iou_ths}
            num_gt_total = 0
            # breakpoint()
            for gt_box, pred_box in zip(gt_boxes, pred_boxes):
                for token in gt_box.sample_tokens:
                    gt_list = [
                        b for b in gt_box[token] if b.detection_name == cls
                    ]
                    pred_token_list = pred_box.boxes.get(token, [])
                    pred_all = [b for b in pred_token_list if b.detection_name == cls]
                    pred_list = sorted(
                        pred_all, key=lambda b: b.detection_score, reverse=True
                    )

                    num_gt_total += len(gt_list)
                    if len(pred_list) == 0:
                        continue

                    for thr in iou_ths:
                        tp, fp, sc = self._match_by_iou_greedy(pred_list, gt_list, thr)
                        per_thr_tp[thr].extend(tp)
                        per_thr_fp[thr].extend(fp)
                        per_thr_sc[thr].extend(sc)
                # breakpoint()
            for thr in iou_ths:
                ap = self._compute_ap_from_scores(
                    per_thr_tp[thr], per_thr_fp[thr], per_thr_sc[thr], num_gt_total
                )
                label_aps[cls][str(thr)] = ap

        # 跨类别宏平均
        def mean_over_classes(th):
            vals = []
            for c in class_names:
                v = label_aps[c].get(str(th), None)
                if v is not None:
                    vals.append(v)
            return float(np.mean(vals)) if len(vals) > 0 else 0.0

        mean_ap_05 = mean_over_classes(0.5)
        mean_ap_07 = mean_over_classes(0.7)

        # 也构造便于直接打印的结构
        AP_at_05 = {c: float(label_aps[c].get("0.5", 0.0)) for c in class_names}
        AP_at_07 = {c: float(label_aps[c].get("0.7", 0.0)) for c in class_names}

        # 为兼容旧逻辑，mean_ap 就等于 AP@0.5
        metrics = {
            # 兼容字段
            "label_aps": label_aps,
            "mean_ap": float(mean_ap_05),
            "mean_ap_0.5": float(mean_ap_05),
            "mean_ap_0.7": float(mean_ap_07),
            # 新增直观字段（打印更方便）
            "AP@0.5": AP_at_05,
            "AP@0.7": AP_at_07,
            "mAP@0.5": float(mean_ap_05),
            "mAP@0.7": float(mean_ap_07),
        }

        # 保存 + 打印
        mmcv.mkdir_or_exist(self.output_dir)
        out_json = osp.join(self.output_dir, "metrics_summary.json")
        mmcv.dump(metrics, out_json)
        if self.verbose:
            print(
                f"[IoU Eval] AP@0.5(mAP)={mean_ap_05:.4f}, AP@0.7(mAP)={mean_ap_07:.4f}; saved to: {out_json}"
            )
        self._print_console_summary(metrics)

    # ---------- Utilities ----------
    def _boxes_to_bev_xywlr_numpy(self, boxes: List[MogoDetectionBox]):
        """
        Convert a list of boxes to [x, y, w, l, yaw] for BEV IoU.
        注意：这里假设 size 的顺序是 (w, l, h)。请与生成预测时保持一致。
        """
        N = len(boxes)
        arr = np.zeros((N, 5), dtype=np.float32)
        for i, b in enumerate(boxes):
            arr[i, 0] = float(b.translation[0])
            arr[i, 1] = float(b.translation[1])
            w, l = float(b.size[0]), float(b.size[1])
            arr[i, 2] = w
            arr[i, 3] = l
            q = pyquaternion.Quaternion(b.rotation)
            arr[i, 4] = float(q.yaw_pitch_roll[0])  # 弧度（与 mmcv.ops.box_iou_rotated 常用格式一致）
        return arr

    def _fast_iou_rotated_bev(self, a_xywlr: np.ndarray, b_xywlr: np.ndarray) -> np.ndarray:
        """
        Compute BEV rotated IoU matrix (Na x Nb).
        优先使用 mmcv.ops.box_iou_rotated；否则退化为 AABB IoU（近似）。
        """
        if a_xywlr.shape[0] == 0 or b_xywlr.shape[0] == 0:
            return np.zeros((a_xywlr.shape[0], b_xywlr.shape[0]), dtype=np.float32)

        try:
            from mmcv.ops import box_iou_rotated
            import torch
            # mmcv 的入参格式：[x_ctr, y_ctr, w, h, angle]（弧度）
            # a = torch.tensor(a_xywlr[:, [0, 1, 3, 2, 4]], dtype=torch.float32)
            # b = torch.tensor(b_xywlr[:, [0, 1, 3, 2, 4]], dtype=torch.float32)
            a = torch.tensor(a_xywlr[:, [0, 1, 2, 3, 4]], dtype=torch.float32)
            b = torch.tensor(b_xywlr[:, [0, 1, 2, 3, 4]], dtype=torch.float32)
            iou = box_iou_rotated(a, b)  # (Na, Nb)
            return iou.cpu().numpy()
        except Exception:
            # Fallback：将旋转框转成 AABB 近似（粗糙但通用）
            def to_aabb(xywlr):
                x, y, w, l, _ = xywlr
                hw, hl = w / 2.0, l / 2.0
                return np.array([x - hw, y - hl, x + hw, y + hl], dtype=np.float32)

            Na, Nb = a_xywlr.shape[0], b_xywlr.shape[0]
            a_aabb = np.stack([to_aabb(a_xywlr[i]) for i in range(Na)], axis=0)
            b_aabb = np.stack([to_aabb(b_xywlr[j]) for j in range(Nb)], axis=0)
            iou = np.zeros((Na, Nb), dtype=np.float32)

            for i in range(Na):
                ax1, ay1, ax2, ay2 = a_aabb[i]
                a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                for j in range(Nb):
                    bx1, by1, bx2, by2 = b_aabb[j]
                    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                    ix1 = max(ax1, bx1)
                    iy1 = max(ay1, by1)
                    ix2 = min(ax2, bx2)
                    iy2 = min(ay2, by2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    union = a_area + b_area - inter + 1e-6
                    iou[i, j] = inter / union
            return iou

    def _match_by_iou_greedy(
        self, pred_boxes: List[MogoDetectionBox], gt_boxes: List[MogoDetectionBox], iou_thr: float
    ):
        """
        Per-sample, per-class greedy matching by IoU.
        返回：tp_flags, fp_flags, scores
        """
        if len(pred_boxes) == 0:
            return [], [], []
        scores = [float(b.detection_score) for b in pred_boxes]
        pred_xywlr = self._boxes_to_bev_xywlr_numpy(pred_boxes)
        gt_xywlr = (
            self._boxes_to_bev_xywlr_numpy(gt_boxes)
            if len(gt_boxes) > 0
            else np.zeros((0, 5), dtype=np.float32)
        )
        if len(gt_boxes) == 0:
            return [0] * len(pred_boxes), [1] * len(pred_boxes), scores

        iou_mat = self._fast_iou_rotated_bev(pred_xywlr, gt_xywlr)  # (Np, Ng)
        matched_gt = set()
        tp = [0] * len(pred_boxes)
        fp = [0] * len(pred_boxes)
        # breakpoint()
        # 这里调用方已按 score 降序
        for i in range(len(pred_boxes)):
            j_best = int(np.argmax(iou_mat[i])) if gt_xywlr.shape[0] > 0 else -1
            best = iou_mat[i, j_best] if j_best >= 0 else 0.0
            if best >= iou_thr and j_best not in matched_gt:
                tp[i] = 1
                matched_gt.add(j_best)
            else:
                fp[i] = 1
        return tp, fp, scores

    def _compute_ap_from_scores(self, tp_list, fp_list, score_list, num_gt):
        """
        Interpolated AP（PR 上凸包）
        """
        if len(score_list) == 0 or num_gt == 0:
            return 0.0
        order = np.argsort(-np.array(score_list))
        tp = np.array(tp_list, dtype=np.float32)[order]
        fp = np.array(fp_list, dtype=np.float32)[order]
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / (num_gt + 1e-6)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

        # 插值保证 precision 非增
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        return ap

    # ---------- 控制台打印：只打印 IoU=0.5/0.7 的 mAP 与各类 AP ----------
    def _print_console_summary(self, metrics: Dict[str, Any]) -> None:
        """
        仅输出：
          - mAP@0.5 与各类 AP@0.5
          - mAP@0.7 与各类 AP@0.7
        """
        print("====== IoU Evaluation Results ======")

        # 先取我们新增的直观字段（若缺失则从兼容字段计算）
        mAP05 = float(metrics.get("mAP@0.5", metrics.get("mean_ap_0.5", 0.0)))
        mAP07 = float(metrics.get("mAP@0.7", metrics.get("mean_ap_0.7", 0.0)))
        AP05 = metrics.get("AP@0.5", None)
        AP07 = metrics.get("AP@0.7", None)

        # 兜底：从 label_aps 里还原
        if AP05 is None or AP07 is None:
            AP05, AP07 = {}, {}
            label_aps = metrics.get("label_aps", {})
            for cls, d in label_aps.items():
                AP05[cls] = float(d.get("0.5", 0.0))
                AP07[cls] = float(d.get("0.7", 0.0))

        # 打印 IoU=0.5
        print("\nResults at IoU = 0.5")
        print(f"mAP@0.5: {mAP05:.4f}")
        print(f"{'Object Class':<16} {'AP@0.5':>8}")
        for cls in sorted(AP05.keys()):
            print(f"{cls:<16} {AP05[cls]:>8.3f}")

        # 打印 IoU=0.7
        print("\nResults at IoU = 0.7")
        print(f"mAP@0.7: {mAP07:.4f}")
        print(f"{'Object Class':<16} {'AP@0.7':>8}")
        for cls in sorted(AP07.keys()):
            print(f"{cls:<16} {AP07[cls]:>8.3f}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data-root', type=str, default='/rss/lishuaiyin/4D_label/dataset_track/offline_tracked',
                        help='eval dataset root path')
    parser.add_argument('--sequence', type=str, default='train_sh_3d_road_2_20250531_5000_lx',
                        help='sequence name')
    parser.add_argument('--metrics', type=str, default='iou',
                        help='metrics name')
    
    args = parser.parse_args()

    dataset = MogoDataset_evalrefine(data_root=args.data_root, sequence=args.sequence, 
                                     classname2id=name2id_map, class_mapping=class_name2refine_name)
    
    dataset.evaluate(eval_metric=args.metrics, eval_boxes_types=["detection_bboxes", "tracking_bboxes","tracking_with_drop_bboxes"])
