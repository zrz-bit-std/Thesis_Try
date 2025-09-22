import os
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def bev_polygon(x, z, w, l, ry):
    """生成BEV旋转矩形的多边形"""
    corners = np.array([
        [ l/2,  w/2],
        [ l/2, -w/2],
        [-l/2, -w/2],
        [-l/2,  w/2]
    ])
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, -s], [s, c]])
    corners = corners @ R.T
    corners[:, 0] += x
    corners[:, 1] += z
    return Polygon(corners)

def polygon_iou(poly1, poly2):
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0

def bev_nms(objects, iou_thr=0.5):
    """对一个文件中的所有目标做BEV NMS"""
    objects = sorted(objects, key=lambda x: x[1], reverse=True)
    keep = []
    while objects:
        cur = objects.pop(0)
        keep.append(cur)
        objects = [obj for obj in objects if polygon_iou(cur[2], obj[2]) < iou_thr]
    return keep

def process_file(in_path, out_path, iou_thr=0.5):
    objects = []
    with open(in_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 16:  # KITTI 格式至少16列
                continue
            cls, h, w, l, x, y, z, ry, score = (
                parts[0],float(parts[8]), float(parts[9]), float(parts[10]),
                float(parts[11]), float(parts[12]), float(parts[13]),
                float(parts[14]), float(parts[15])
            )
            # motor类别阈值过滤
            if cls=='motor' and score<=0.5:
                continue
            if cls=='truck' and score<=0.4:
                continue
            
            poly = bev_polygon(x, y, w, l, ry)
            objects.append((line.strip(), score, poly))

    kept = bev_nms(objects, iou_thr)

    with open(out_path, "w") as f:

        for line, _, _ in kept:
            f.write(line + "\n")

    return len(objects), len(kept)  # 返回统计结果

def process_labels(label_dir, output_dir, iou_thr=0.5, num_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    total_orig, total_kept = 0, 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_file,
                os.path.join(label_dir, file),
                os.path.join(output_dir, file),
                iou_thr
            ): file for file in label_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="NMS Processing"):
            orig, kept = future.result()
            total_orig += orig
            total_kept += kept

    print(f"\n✅ NMS Done! Results saved to {output_dir}")
    print(f"📊 Total stats: {total_orig} → {total_kept} kept, {total_orig - total_kept} removed")

if __name__ == "__main__":
    label_dir = "/adga/lushiyong/vis_anno/dataset/hm_i7_0830/kitti_result/3d_url"      # 输入文件夹
    output_dir = "/adga/lushiyong/vis_anno/dataset/hm_i7_0830/kitti_result/3d_url_nms" # 输出文件夹
    process_labels(label_dir, output_dir, iou_thr=0.1, num_workers=20)