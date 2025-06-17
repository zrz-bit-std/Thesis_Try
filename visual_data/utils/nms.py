import numpy as np

from math import pi, cos, sin
from shapely.geometry import Polygon


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x * v.y - self.y * v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a * p.x + self.b * p.y + self.c

    def intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a * other.b - self.b * other.a
        return Vector(
            (self.b * other.c - self.c * other.b) / (w+1e-10),
            (self.c * other.a - self.a * other.c) / (w+1e-10)
        )


def rectangle_vertices(cx, cy, w, h, r):
    angle = pi * r / 180
    dx = w / 2
    dy = h / 2
    dxcos = dx * cos(angle)
    dxsin = dx * sin(angle)
    dycos = dy * cos(angle)
    dysin = dy * sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector(dxcos - -dysin, dxsin + -dycos),
        Vector(cx, cy) + Vector(dxcos - dysin, dxsin + dycos),
        Vector(cx, cy) + Vector(-dxcos - dysin, -dxsin + dycos)
    )


def intersection_area(r1, r2):
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices

    rect1 = rectangle_vertices(*r1)
    rect2 = rectangle_vertices(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break  # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
                intersection, intersection[1:] + intersection[:1],
                line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x * q.y - p.y * q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))


def bb_intersection_over_union(boxA, boxB):
    def _get_box(boxA):
        # cx, cy, l, w, r
        # return boxA[0], boxA[1], boxA[4], boxA[3], boxA[6] * 180 / pi
        return float(boxA[1]), float(boxA[2]), float(boxA[6]), float(boxA[4]), float(boxA[7]) * 180 / pi

    def _get_boxArea(boxA):
        # w*l
        # return boxA[3] * boxA[4]
        return float(boxA[4]) * float(boxA[6])

    _boxA = _get_box(boxA)
    _boxB = _get_box(boxB)
    boxAArea = _get_boxArea(boxA)
    boxBArea = _get_boxArea(boxB)
    interArea = intersection_area(_boxA, _boxB)

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def bb_intersection_over_union_angle(boxA, boxB):
    def _get_box(boxA):
        # cx, cy, l, w, r
        # return boxA[0], boxA[1], boxA[4], boxA[3], boxA[6] * 180 / pi
        return float(boxA[0]), float(boxA[1]), float(boxA[2]), float(boxA[3]), float(boxA[4]) 

    def _get_boxArea(boxA):
        # w*l
        # return boxA[3] * boxA[4]
        return float(boxA[2]) * float(boxA[3])

    _boxA = _get_box(boxA)
    _boxB = _get_box(boxB)
    boxAArea = _get_boxArea(boxA)
    boxBArea = _get_boxArea(boxB)
    interArea = intersection_area(_boxA, _boxB)

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def get_rotated_rect_vertices(center_x, center_y, length, width, angle):
    """
    计算旋转矩形的四个顶点坐标
    
    参数:
    center_x, center_y: 矩形中心点坐标
    length: 矩形长度（沿车头方向）
    width: 矩形宽度
    angle: 矩形旋转角度（弧度），逆时针为正
    
    返回:
    四个顶点的坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    # 计算半长和半宽
    half_length = length / 2
    half_width = width / 2
    
    # 定义矩形在局部坐标系下的四个顶点（以中心点为原点，未旋转）
    local_vertices = [
        (half_length, half_width),    # 右上
        (half_length, -half_width),   # 右下
        (-half_length, -half_width),  # 左下
        (-half_length, half_width)    # 左上
    ]
    
    # 旋转矩阵
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # 旋转并平移到全局坐标系
    global_vertices = []
    for x_local, y_local in local_vertices:
        # 旋转
        x_rotated = x_local * cos_angle - y_local * sin_angle
        y_rotated = x_local * sin_angle + y_local * cos_angle
        # 平移
        x_global = x_rotated + center_x
        y_global = y_rotated + center_y
        global_vertices.append((x_global, y_global))
    
    return global_vertices

def calculate_bev_iou(box1, box2):
    """
    计算两个BEV视角下旋转矩形的IOU
    
    参数:
    box1, box2: 旋转矩形参数，格式为 [x, y, z, l, w, h, yaw, score]
                其中yaw为弧度
    
    返回:
    IOU值
    """
    # 解析框参数
    cx1, cy1, _, l1, w1, _, a1, _ = box1
    cx2, cy2, _, l2, w2, _, a2, _ = box2
    
    # 计算顶点
    vertices1 = get_rotated_rect_vertices(cx1, cy1, l1, w1, a1)
    vertices2 = get_rotated_rect_vertices(cx2, cy2, l2, w2, a2)
    
    # 创建多边形
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)
    
    # 计算交集和并集面积
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    
    # 计算IOU
    if union > 0:
        iou = intersection / union
    else:
        iou = 0.0
    
    return iou

def nms(boxes, iou_thres):
    """ 非极大值抑制 """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2-x1) * (y2-y1)
    keep = []

    # 按置信度进行排序
    index = np.argsort(scores)[::-1]

    while(index.size):
        # 置信度最高的框
        i = index[0]
        keep.append(index[0])

        if(index.size == 1): # 如果只剩一个框，直接返回
            break

        # 计算交集左下角与右上角坐标
        inter_x1 = np.maximum(x1[i], x1[index[1:]])
        inter_y1 = np.maximum(y1[i], y1[index[1:]])
        inter_x2 = np.minimum(x2[i], x2[index[1:]])
        inter_y2 = np.minimum(y2[i], y2[index[1:]])
        # 计算交集的面积
        inter_area = np.maximum(inter_x2-inter_x1, 0) * np.maximum(inter_y2-inter_y1, 0)
        # 计算当前框与其余框的iou
        iou = inter_area / (areas[index[1:]] + areas[i] - inter_area)
        ids = np.where(iou < iou_thres)[0]
        index = index[ids+1] # 相当于取出来，ids+1 是因为 index[1:]

    return boxes[keep]


def nms_by_dist(boxes, dist_thresh=1.0):
    """_summary_

    Args:
        boxes (ndarray): (n, 8), ordered (x, y, z, )
        dist_thresh (float, optional): dist thresh for nms. Defaults to 1.0.
    """

    centers = boxes[:, :2]  # in x-y
    scores = boxes[:, -1]

    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        center_i = centers[i:i+1]
        remaining_centers = centers[order[1:]]

        distances = np.sqrt(np.square(center_i - remaining_centers).sum(axis=1))

        inds = np.where(distances > dist_thresh)[0] + 1
        order = order[inds]
    
    return keep


def nms_angle(boxes_list, iou_thres):
    """ 非极大值抑制 """
    boxes = np.array(boxes_list)
    scores = np.array([i[-1] for i in boxes_list])
    keep = []

    # 按置信度进行排序
    index = np.argsort(scores)[::-1]

    while(index.size):
        # 置信度最高的框
        indx_now = index[0]
        keep.append(indx_now)

        if(index.size == 1): # 如果只剩一个框，直接返回
            break
        
        index_other_iou = []
        for indx_other in index[1:]:
            iou = bb_intersection_over_union_angle(boxes_list[indx_now],boxes_list[indx_other])
            if iou <= iou_thres:
                index_other_iou.append(indx_other)
        index = np.array(index_other_iou)
    return boxes[keep],keep


def nms_points(points_list, distance_thres):
    """ 非极大值抑制 """
    points = np.array(points_list)
    scores = np.array([i[-1] for i in points_list])
    keep = []

    # 按置信度进行排序
    index = np.argsort(scores)[::-1]

    while(index.size):
        # 置信度最高的框
        indx_now = index[0]
        keep.append(indx_now)

        if(index.size == 1): # 如果只剩一个框，直接返回
            break
        
        index_other_iou = []
        for indx_other in index[1:]:
            distance = np.sqrt((points_list[indx_now][0]-points_list[indx_other][0])**2 + (points_list[indx_now][1]-points_list[indx_other][1])**2)
            if distance > distance_thres:
                index_other_iou.append(indx_other)
        index = np.array(index_other_iou)
    return points[keep],keep


def nms_by_iou(boxes, iou_thresh=0.5):
    """_summary_

    Args:
        boxes (ndarray): (n, 8), ordered (x, y, z, w, l, h, yaw, score)
        iou_thresh (float, optional): iou threshold. Defaults to 0.5.

    Returns:
        _type_: _description_
    """

    # centers = boxes[:, :2]  # in x-y
    scores = boxes[:, -1]

    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = np.array(
            [calculate_bev_iou(boxes[i], boxes[j]) for j in order[1:]]
        )
        inds = np.where(ious <= iou_thresh)[0] + 1
        order = order[inds]
    
    return keep


def mixed_nms(boxes, label_names, iou_thresh=0.5, dist_thresh=1.0):
    """_summary_

    Args:
        boxes (ndarray): (n, 8), ordered (x, y, z, w, l, h, yaw, score)
        label_names (list): (n, ), bbox label names corresponds to boxes.
        iou_thresh (float, optional): iou threshold. Defaults to 0.5.

    Returns:
        _type_: _description_
    """

    apply_dist_names = ["car", "truck", "bus"]
    scores = boxes[:, -1]

    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break
        
        remaining_order = []
        for j in order[1:]:
            if (
                label_names[i] in apply_dist_names and
                label_names[j] in apply_dist_names
            ):
                dist = np.sqrt(np.square(boxes[i, :2] - boxes[j, :2]).sum())
                if dist >= dist_thresh:
                    remaining_order.append(j)
            else:
                iou = calculate_bev_iou(boxes[i], boxes[j])
                if iou <= iou_thresh:
                    remaining_order.append(j)
        order = np.array(remaining_order)         
    
    return keep

if __name__ == '__main__':
    r1 = (-15.48, 13.79, 4.90, 1.84, -80.74)
    r2 = (-16.47, 11.63, 4.72, 1.82, -80.78)
    print(intersection_area(r1, r2))
