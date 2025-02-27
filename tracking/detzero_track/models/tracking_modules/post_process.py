import copy
from functools import partial

import torch
import math
import numpy as np

from .data_association import bev_overlap_gpu


class PostProcessor():
    def __init__(self, processor_configs):
        self.post_process_queue = []
        for cur_cfg in processor_configs.CONFIG_LIST:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.post_process_queue.append(cur_processor)

    def forward(self, data_dict):
        for cur_processor in self.post_process_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict

    def empty_track_delete(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.empty_track_delete, config=config)

        remove_tk_id = list()
        for tk_id, tk_data in data_dict.items():
            track_history = len(tk_data['hit'])
            hit_count = np.sum(tk_data['hit'] > 0)

            if hit_count < config.LEAST_AGE:
                remove_tk_id.append(tk_id)
            else:
                if hit_count != track_history:
                    remove_indexs = list()
                    for idx in range(track_history):
                        if tk_data['hit'][idx] >= 1: break
                        else: remove_indexs.append(idx)
                    for idx in reversed(range(track_history)):
                        if tk_data['hit'][idx] >= 1: break
                        else: remove_indexs.append(idx)

                    # remove_indexs=sorted(remove_indexs, reverse=True)
                    if config.END_REMOVE:
                        for key in tk_data.keys():
                            tk_data[key] = np.delete(
                                tk_data[key], remove_indexs, axis=0
                            )

        for tk_id in remove_tk_id:
            data_dict.pop(tk_id)
        return data_dict

    def velocity_optimize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.velocity_optimize, config=config)

        header_len = config.HEADER_LENGTH
        for tk_id, tk_data in data_dict.items():
            track_len = len(tk_data['boxes_global'])

            if track_len < 2: continue
            process_len = header_len if track_len > header_len else track_len-1
            for idx in range(process_len):
                speed = (tk_data['boxes_global'][idx+1, :2] - tk_data['boxes_global'][idx, :2])*10.
                data_dict[tk_id]['boxes_global'][idx, 7:9] = copy.deepcopy(speed)
            if process_len == track_len:
                data_dict[tk_id]['boxes_global'][-1, 7:9] = copy.deepcopy(tk_data['boxes_global'][-2, 7:9])
        return data_dict

    def motion_classify(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.motion_classify, config=config)

        for tk_id, tk_data in data_dict.items():
            hit_index = np.flatnonzero(tk_data['hit'] == 1)
            track_len = len(hit_index)
            if track_len < 2:
                data_dict[tk_id]['state'] = "static"
            else:
                track_box = torch.from_numpy(tk_data['boxes_global'][hit_index, :7]).float().cuda()
                bev_iou_mat = bev_overlap_gpu(track_box, track_box)
                if np.any(bev_iou_mat  <= 1e-4):
                    data_dict[tk_id]['state'] = "dynamic"
                else:
                    data_dict[tk_id]['state'] = "static"

        return data_dict

    def static_drift_eliminate(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.static_drift_eliminate, config=config)

        for tk_id, tk_data in data_dict.items():
            if tk_data['state'] == 'static' and tk_data['name'][0] == 'Vehicle':
                hit_idxs = np.flatnonzero(tk_data['hit'] == 1)
                temp_max_score_idx = np.argsort(tk_data['score'][hit_idxs])[-1]
                hit_max_score_idx = hit_idxs[temp_max_score_idx]

                track_history = len(tk_data['hit'])
                for idx in reversed(range(track_history)):
                    if tk_data['hit'][idx] >= 1: break
                    else:
                        data_dict[tk_id]['boxes_global'][idx] = copy.deepcopy(
                            tk_data['boxes_global'][hit_max_score_idx])
        return data_dict

    def box_info_update(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.box_info_update, config=config) 
        
        # 对尺寸进行修正
        data_dict = self.box_size_update(data_dict=data_dict, config = config)

        # 对齐目标车尾与点云末端（跟踪目标与检测目标存在位置偏差，强制对齐）
        # self.move_trackobj_position(data_dict, config)
        
        # 修正目标朝向（车辆朝向反向或者由于跟踪出现的较大偏差）
        self.update_trackobj_heading(data_dict, config)
        return data_dict

    def box_size_update(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.box_size_update, config=config)

        for tk_id, tk_data in data_dict.items():
            scores = tk_data['score']
            boxes = tk_data['boxes_global']

            if config.METHOD == 'max_score_box':
                max_score_indexs = np.where(scores == np.max(scores))[0]
                max_boxes = np.zeros(3, dtype=np.float32)
                for idx in max_score_indexs:
                    max_boxes += boxes[idx, 3:6]
                max_boxes = max_boxes / (len(max_score_indexs))
                data_dict[tk_id]['boxes_global'][:, 3:6] = max_boxes

            elif config.METHOD == 'score_weigthed_box':
                weighted_boxes = np.zeros(3, dtype=np.float32)
                for idx in range(len(scores)):
                    weighted_boxes += scores[idx] * boxes[idx, 3:6]
                weighted_boxes = weighted_boxes/np.sum(scores)
                data_dict[tk_id]['boxes_global'][:, 3:6] = weighted_boxes

            elif config.METHOD == 'largest_box':
                area = boxes[:, 3] * boxes[:, 4] * boxes[:, 5]
                largest_idx = np.argsort(area)[-1]
                largest_boxes = copy.deepcopy(boxes[largest_idx, 3:6])
                data_dict[tk_id]['boxes_global'][:, 3:6] = largest_boxes

        return data_dict
    
    # 修正目标朝向（车辆朝向反向或者由于跟踪出现的较大偏差）
    def update_trackobj_heading(self, data_dict=None, config=None):
        # 朝向角可靠对应的最小目标置信度
        score_thre = config.SCORE_THRESHOLD
        head_thre = math.pi * 0.25

        for tk_id, tk_data in data_dict.items():
            # sample_id = tk_data['sample_idx']
            boxes = tk_data['boxes_global']
            scores = tk_data['score']
            type_name = tk_data['name'][0]
            frame_num = len(scores)
            
            # 获取运动状态以及最大score对应的朝向
            reliable_head_array = []
            move_near_head = np.zeros(len(scores))
            move_flag = np.zeros(len(scores))
            max_score_index = 0
            max_score_head = 0
            max_score = 0

            test_move_velocity = np.zeros(len(scores))
            test_move_dy = np.zeros(len(scores))
            test_move_dx = np.zeros(len(scores))
            test_is_change = False
            
            for idx in range(frame_num):
                if (scores[idx] > max_score):
                    max_score_head = boxes[idx, 6]
                    max_score = scores[idx]
                    max_score_index = idx
                if (scores[idx] > (score_thre[type_name] - 0.05)):
                    reliable_head_array.append(scores[idx])
                # 寻找朝向相同或者相反的相邻帧
                compare_idx = idx + 1
                if (idx == frame_num -1) : compare_idx = idx -1
                similar_head = abs(abs(boxes[idx, 6]) - abs(boxes[compare_idx, 6])) < head_thre or \
                               abs(abs(boxes[idx, 6]) - abs(boxes[compare_idx, 6])) > (math.pi - head_thre)
                if (not similar_head and idx > 0 and idx < (frame_num -1)) :
                    compare_idx = idx -1
                    similar_head = abs(abs(boxes[idx, 6]) - abs(boxes[compare_idx, 6])) < head_thre or \
                                   abs(abs(boxes[idx, 6]) - abs(boxes[compare_idx, 6])) > (math.pi - head_thre)
                if (similar_head) :
                    # 有相同朝向的帧间才计算速度（进一步判断目标运动状态），突然翻转90度的不用于计算速度
                    dy = boxes[compare_idx, 1] - boxes[idx, 1]
                    dx = boxes[compare_idx, 0] - boxes[idx, 0]
                    if (compare_idx < idx):
                        dy = boxes[idx, 1] - boxes[compare_idx, 1]
                        dx = boxes[idx, 0] - boxes[compare_idx, 0]
                    if (math.sqrt(dy*dy + dx*dx) > 0.3) : 
                        move_near_head[idx] = math.atan2(dy, dx)
                        move_flag[idx] = 2
                    else :
                        move_flag[idx] = 1

                    test_move_velocity[idx] = math.sqrt(dy*dy + dx*dx)
                    test_move_dy[idx] = dy
                    test_move_dx[idx] = dx
            
            # TODO： 后续对短轨迹以及低置信度目标和行人进行朝向修复
            if (type_name == 'TYPE_BARRIER' or type_name == 'TYPE_PED_ADULT') : continue
            if (frame_num < 10 or max_score < score_thre[type_name]) : continue

            # 分场景处理较大偏差的朝向
            turning_scene = (max(reliable_head_array) - min(reliable_head_array)) > head_thre
            static_scene = np.max(move_flag) < 2 or (5 * np.sum(move_flag == 2) < np.sum(move_flag == 1))

            for idx in range(frame_num) :
                if (turning_scene) :
                    # 转弯场景只处理动态轨迹中的置信度低的部分(只考虑朝向翻转180度的情况)
                    if (move_flag[idx] == 2 and scores[idx] < score_thre[type_name]):
                        if(abs(abs(boxes[idx, 6]) - abs(move_near_head[idx])) > math.pi*0.75):
                            data_dict[tk_id]['boxes_global'][idx, 6] = self.flip_head(data_dict[tk_id]['boxes_global'][idx, 6])
                elif (static_scene):
                    # 静态场景只对低置信度部分处理（防止轻微运动造成的影响），处理所有较大偏差角度
                    if (abs(abs(boxes[idx, 6]) - abs(max_score_head)) < head_thre) :
                        continue     
                    if (scores[idx] < score_thre[type_name]) :
                        data_dict[tk_id]['boxes_global'][idx, 0:7] = data_dict[tk_id]['boxes_global'][max_score_index, 0:7]
                else:
                    if (abs(abs(boxes[idx, 6]) - abs(max_score_head)) < head_thre) :
                        continue   
                    # 可靠的运动轨迹只处理朝向180度翻转问题，不处理90度翻转
                    if (move_flag[idx] == 2):
                        compare_head = max_score_head
                        if(abs(abs(boxes[idx, 6]) - abs(compare_head)) > math.pi*0.75):
                            data_dict[tk_id]['boxes_global'][idx, 6] = self.flip_head(data_dict[tk_id]['boxes_global'][idx, 6])
                    else:
                        # 静态轨迹或者无法获取相邻朝向的轨迹，寻找该段轨迹中的最大score对应的朝向
                        near_max_score = 0
                        near_valid_head = 0
                        for i in reversed(range(0, idx)):
                            if (move_flag[i] == 2): break
                            if (scores[i] > near_max_score):
                                near_max_score = scores[i]
                                near_valid_head = boxes[i, 6]
                        for i in range(idx, frame_num):
                            if (move_flag[i] == 2): break
                            if (scores[i] > near_max_score):
                                near_max_score = scores[i]
                                near_valid_head = boxes[i, 6]
                        if (near_max_score < (score_thre[type_name] - 0.05)) :
                            continue
                        if (abs(abs(boxes[idx, 6]) - abs(near_valid_head)) > head_thre) :
                            data_dict[tk_id]['boxes_global'][idx, 6] = near_valid_head

            # 加入整个轨迹朝向滤波（用于解决遗漏的180度翻转问题）
            heads_global = copy.deepcopy(data_dict[tk_id]['boxes_global'][:,6])
            compare_head_thre = head_thre if turning_scene else head_thre * 0.5
            for idx in range(frame_num):
                mean_head = 0
                filter_n = 0
                for i in range(1, 4):
                    if (idx + i < frame_num): 
                        filter_n += 1
                        mean_head += abs(heads_global[(idx+i)])
                    if (idx - i >= 0): 
                        filter_n += 1
                        mean_head += abs(heads_global[(idx-i)])
                mean_head = mean_head / filter_n
                if (abs(abs(mean_head) - abs(heads_global[idx])) > math.pi * 0.9) :
                    # 当前帧与相邻帧朝向均反向
                    data_dict[tk_id]['boxes_global'][idx,6] = mean_head
                    heads_global[idx] = mean_head
                elif (abs(abs(mean_head) - abs(heads_global[idx])) > compare_head_thre) :
                    # 当前帧与相邻帧中朝向存在相同的也存在反向的
                    near_max_score = scores[idx]
                    valid_head = heads_global[idx]
                    for i in range(1, 4):
                        if (idx + i < frame_num):
                            if(scores[(idx+i)] > near_max_score): 
                                near_max_score = scores[(idx+i)]
                                valid_head = heads_global[idx + i]
                        if (idx - i >= 0):
                            if(scores[(idx-i)] > near_max_score): 
                                near_max_score = scores[(idx-i)]
                                valid_head = heads_global[idx - i]
                    data_dict[tk_id]['boxes_global'][idx,6] = valid_head
                    heads_global[idx] = valid_head


    def flip_head(self, yaw):
        if (yaw >= 0) :
            return (yaw - math.pi)
        else:
            return (yaw + math.pi)
