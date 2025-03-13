import numpy as np

from math import pi, cos, sin


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

if __name__ == '__main__':
    r1 = (-15.48, 13.79, 4.90, 1.84, -80.74)
    r2 = (-16.47, 11.63, 4.72, 1.82, -80.78)
    print(intersection_area(r1, r2))
