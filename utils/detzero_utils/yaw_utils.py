import numpy as np
import math

def normalize_angle_deg(angle):
    """
    将角度标准化到 [-180, 180] 范围内
    
    参数:
    angle: 输入角度（度）
    
    返回:
    标准化后的角度
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def correct_initial_yaw_direction(objs, window_size=10):
    """
    校正轨迹第一帧的角度方向，通过检查前几帧数据的一致性
    
    参数:
    objs: 对象列表
    window_size: 用于判断方向的窗口大小，默认为10帧
    """
    if len(objs) < 2:
        return
    
    # 获取前window_size帧的角度数据
    window_size = min(window_size, len(objs))
    yaw_angles = np.array([obj.ry for obj in objs[:window_size]])
    
    # 标准化所有角度到[-180, 180]范围
    yaw_angles = np.array([normalize_angle_deg(angle) for angle in yaw_angles])
    
    # 计算相邻帧之间的角度差
    diffs = np.diff(yaw_angles)
    
    # 如果第一帧与后续帧存在明显不一致，进行校正
    if window_size > 1:
        # 计算第一帧与第二帧的角度差
        first_diff = normalize_angle_deg(objs[1].ry - objs[0].ry)
        
        # 如果角度差过大（超过90度），可能第一帧方向错误
        if abs(first_diff) > 90:
            # 检查后续帧的角度变化趋势
            if len(diffs) > 1:
                # 计算后续帧的平均角度变化
                avg_diff = np.mean(diffs[1:]) if len(diffs) > 2 else diffs[0]
                
                # 如果第一帧与第二帧的变化方向与后续帧不一致，可能需要翻转
                if np.sign(first_diff) != np.sign(avg_diff) and abs(avg_diff) < 45:
                    # 翻转第一帧的角度（加180度）
                    objs[0].ry = normalize_angle_deg(objs[0].ry + 180)
                    
                    # 同时检查是否需要翻转第二帧（如果仍然不一致）
                    second_diff = normalize_angle_deg(objs[1].ry - objs[0].ry)
                    if abs(second_diff) > 90:
                        objs[1].ry = normalize_angle_deg(objs[1].ry + 180)

def unwrap_yaw_angles_deg(angles):
    """
    解包yaw角度（度数），避免跳变问题
    
    参数:
    angles: 角度数组（已标准化到[-180, 180]范围内）
    
    返回:
    处理后的角度数组（连续变化的形式）
    """
    if len(angles) <= 1:
        return angles
        
    # 将角度转换为连续变化的形式
    diff = np.diff(angles)
    
    jumps = np.abs(diff) > 150
    # 计算需要添加的补偿值
    corrections = np.zeros_like(angles)
    # 对于正向跳变（从正到负，且差值<-162度），添加180度
    corrections[1:][jumps & (diff > 180)] += 180
    # 对于负向跳变（从负到正，且差值>162度），减去180度
    corrections[1:][jumps & (diff < 180)] -= 180
    # 累积补偿值
    corrections = np.cumsum(corrections)
    
    # 返回修正后的角度
    return angles + corrections

class TrackingObject:
    """
    用于表示跟踪对象的简单类
    """
    def __init__(self, box, name, score, sample_idx, obj_id, pose, state, hit=1, num_points=0):
        self.box = box
        self.name = name
        self.score = score
        self.sample_idx = sample_idx
        self.id = obj_id
        self.pose = pose
        self.state = state
        self.hit = hit
        self.num_points = num_points
        # ry 是 box 数组的第 6 个元素（索引为 6）
        self.ry = box[6] if len(box) > 6 else 0

def fix_yaw_discontinuities(model_outputs):
    """
    修复model_outputs中yaw角度的不连续性问题
    
    参数:
    model_outputs: 模型输出，格式为列表，每个元素是一个字典，包含跟踪序列
    
    返回:
    修复后的model_outputs
    """
    # 处理列表格式的情况
    if isinstance(model_outputs, list):
        # 遍历每个序列（每个序列是一个字典）
        for seq_dict in model_outputs:
            if not isinstance(seq_dict, dict):
                continue
                
            # 遍历字典中的每个轨迹ID
            for obj_id, obj_data in seq_dict.items():
                # 检查是否是 defaultdict(<class 'list'>, {...}) 格式
                if not isinstance(obj_data, dict) or 'boxes_global' not in obj_data:
                    continue
                print(f"Processing sequence {obj_id}")    
                # 从 obj_data 中提取所有帧的数据
                boxes = obj_data.get('boxes_global', [])
                names = obj_data.get('name', [])
                scores = obj_data.get('score', [])
                sample_idxs = obj_data.get('sample_idx', [])
                obj_ids = obj_data.get('obj_ids', [])
                poses = obj_data.get('pose', [])
                states = obj_data.get('state', [])
                hits = obj_data.get('hit', [])
                num_points_list = obj_data.get('num_points', [])
                
                # 确保所有数组长度一致
                if len(boxes) == 0:
                    continue
                    
                # 创建对象列表
                track_objects = []
                for i in range(len(boxes)):
                    box = boxes[i]
                    name = names[i] if i < len(names) else ''
                    score = scores[i] if i < len(scores) else 0
                    sample_idx = sample_idxs[i] if i < len(sample_idxs) else ''
                    obj_id_val = obj_ids[i] if i < len(obj_ids) else obj_id
                    pose = poses[i] if i < len(poses) else np.eye(4)
                    state = states if isinstance(states, str) else (states[i] if i < len(states) else 'static')
                    hit = hits[i] if i < len(hits) else 1
                    num_points = num_points_list[i] if i < len(num_points_list) else 0
                    
                    track_obj = TrackingObject(box, name, score, sample_idx, obj_id_val, pose, state, hit, num_points)
                    track_objects.append(track_obj)
                
                # 如果轨迹对象少于2个，跳过处理
                if len(track_objects) < 2:
                    continue
                
                # 首先将所有yaw角转换为角度
                for obj in track_objects:
                    # 将弧度转换为角度
                    obj.ry = math.degrees(obj.ry)
                
                # 然后校正初始方向
                correct_initial_yaw_direction(track_objects)
                
                # 提取所有角度并标准化
                yaw_angles = np.array([normalize_angle_deg(obj.ry) for obj in track_objects])
                
                # 解包角度以避免跳变
                corrected_angles = unwrap_yaw_angles_deg(yaw_angles)
                
                # 将修正后的角度转换回弧度并赋值回对象
                for i, obj in enumerate(track_objects):
                    obj.ry = math.radians(corrected_angles[i])
                    # 更新原始 box 中的角度值
                    if len(obj.box) > 6:
                        obj.box[6] = obj.ry
                
        return model_outputs
    
    # 如果不是列表格式，直接返回
    else:
        print("Invalid input format. Input must be a list.")
        return model_outputs