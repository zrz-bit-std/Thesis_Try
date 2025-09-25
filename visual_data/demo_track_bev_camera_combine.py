# -*- coding: utf-8 -*-
import os, json, cv2, glob, argparse, math, pickle, numpy as np, subprocess
from tqdm import tqdm
from collections import defaultdict, Counter

import sys
sys.path.insert(0, "/rss/zhanghui/4D_label")
from visual_data.utils.visualize import _COLORS  # 备用

os.environ["DISPLAY"] = ":0"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# ============================= 基础工具 =============================

def images_to_video_ffmpeg(image_folder, output_video, fps=10, limit_first_n=None, slow_factor=1.0):
    files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
    if limit_first_n is not None:
        files = files[:limit_first_n]
    if not files:
        print(f"[WARN] No images in {image_folder}, skip making video.")
        return

    filelist_txt = os.path.join(image_folder, "__filelist.txt")
    with open(filelist_txt, "w") as f:
        for file in files:
            abs_path = os.path.join(image_folder, file)
            f.write(f"file '{abs_path}'\n")

    vf_parts = []
    if slow_factor and slow_factor > 1.0:
        vf_parts.append(f"setpts={float(slow_factor):.3f}*PTS")
    vf_parts += [
        "scale=1280:1280:force_original_aspect_ratio=decrease",
        "pad=1280:1280:(ow-iw)/2:(oh-ih)/2",
        "setsar=1"
    ]
    vf_arg = ",".join(vf_parts)

    cmd = [
        "/usr/bin/ffmpeg","-v","error","-f","concat","-safe","0","-i",filelist_txt,
        "-r",str(fps), "-c:v","libx264","-preset","slow","-crf","23",
        "-pix_fmt","yuv420p","-movflags","+faststart","-vf",vf_arg,"-y",output_video
    ]
    print("[CMD]"," ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("[ERROR] ffmpeg failed:", e)
    finally:
        try: os.remove(filelist_txt)
        except Exception: pass

def rotate_points(points, yaw):
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw),  np.cos(yaw), 0],
                  [0,            0,           1]])
    return R @ points

def project_3d_to_pixel(points, image_shape, K, T, distort=None, filter_z_camera=True, depth=None):
    x,y,z = points
    P = np.array([[x],[y],[z],[1]])
    Pc = np.dot(T[:3,:], P)

    if distort is not None:
        k1,k2,k3,k4 = distort
        xc,yc,zc = Pc.flatten()[:3]
        if filter_z_camera and zc < 0: return None, None, False
        if depth is not None and zc < depth: return None, None, False
        xn, yn = xc/max(zc,1e-6), yc/max(zc,1e-6)
        r = np.hypot(xn, yn)
        if r < 1e-8:
            proj = np.dot(K, np.array([[0],[0],[1]]))
            u,v = proj[0,0], proj[1,0]
        else:
            theta = np.arctan(r)
            theta_d = theta*(1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
            scale = theta_d/max(r,1e-6)
            xd, yd = scale*xn, scale*yn
            proj = np.dot(K, np.array([[xd],[yd],[1]]))
            u,v = proj[0,0], proj[1,0]
    else:
        if Pc[2,0] < 0: return None, None, False
        pix_h = K @ Pc
        u, v = pix_h[0,0]/max(pix_h[2,0],1e-6), pix_h[1,0]/max(pix_h[2,0],1e-6)

    if not (0 <= u < image_shape[1] and 0 <= v < image_shape[0]):
        return u, v, False
    return u, v, True

def _rgb_to_bgr_255(color):
    r,g,b = float(color[0]), float(color[1]), float(color[2])
    if max(r,g,b) <= 1.0:
        return (int(b*255), int(g*255), int(r*255))
    return (int(b), int(g), int(r))

def draw_3d_box_multi(image, corners_2ds, colors, thicknesses=None):
    if thicknesses is None: thicknesses = [4]*len(corners_2ds)
    for corners_2d, color, thick in zip(corners_2ds, colors, thicknesses):
        c = _rgb_to_bgr_255(color)
        for i in range(4):
            cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[(i+1)%4].astype(int)), c, thick)
        for i in range(4,8):
            cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[(i+1)%4+4].astype(int)), c, thick)
        for i in range(4):
            cv2.line(image, tuple(corners_2d[i].astype(int)), tuple(corners_2d[i+4].astype(int)), c, thick)
    return image

def draw_box_on_img_per(params, image, cam2pixel, lidar2cam, distort=None):
    draw_pts, colors, thicknesses = [], [], []
    for item in params:
        if len(item) == 2: param, color; thickness = 4
        else: param, color, thickness = item
        x,y,z,l,w,h,yaw = param
        corners = np.array([
            [-l/2,-w/2,-h/2],[ l/2,-w/2,-h/2],[ l/2, w/2,-h/2],[-l/2, w/2,-h/2],
            [-l/2,-w/2, h/2],[ l/2,-w/2, h/2],[ l/2, w/2, h/2],[-l/2, w/2, h/2]
        ])
        corners_rot = rotate_points(corners.T, yaw)
        corners_world = np.array([[x,y,z]]).T + corners_rot

        corners_2d, valid = [], []
        H,W = image.shape[:2]
        for pt in corners_world.T:
            u,v,ok = project_3d_to_pixel(pt, (H,W), cam2pixel, lidar2cam, distort)
            if u is None and v is None: break
            if ok: valid.append(1)
            corners_2d.append([u,v])
        if len(valid)==8 and len(corners_2d)==8:
            draw_pts.append(np.array(corners_2d))
            colors.append(color)
            thicknesses.append(int(thickness))
    if not draw_pts: return image
    return draw_3d_box_multi(image, draw_pts, colors, thicknesses)

# ============================= 解析与匹配 =============================

def _collect_all_files(root_dir, pattern="*.txt", track_mode=False):
    files = []
    if not os.path.isdir(root_dir): return files
    for d in sorted(os.listdir(root_dir)):
        p1 = os.path.join(root_dir, d)
        if os.path.isdir(p1):
            if track_mode:
                spl = os.path.join(p1, "splited")
                if os.path.isdir(spl):
                    files.extend(sorted(glob.glob(os.path.join(spl, pattern))))
            else:
                files.extend(sorted(glob.glob(os.path.join(p1, pattern))))
                p2 = os.path.join(p1, "splited")
                if os.path.isdir(p2):
                    files.extend(sorted(glob.glob(os.path.join(p2, pattern))))
                for sub_d in sorted(os.listdir(p1)):
                    sub_p = os.path.join(p1, sub_d)
                    if os.path.isdir(sub_p):
                        files.extend(sorted(glob.glob(os.path.join(sub_p, pattern))))
    return files

def _parse_result_files(result_files, color_rgb=(0,1,0), tag="DET", line_thickness=4):
    frame2objs = defaultdict(list)
    frame_ids = []
    for res_file in result_files:
        frame_id = os.path.basename(res_file).replace(".txt","")
        frame_ids.append(frame_id)
        try:
            lines = open(res_file,"r").readlines()
        except Exception as e:
            print(f"[WARN] {tag}: read fail {res_file}: {e}")
            continue
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue
            base = 1 if len(parts) >= 10 else 0
            if len(parts) < base+9: continue
            try:
                h,w,l,x,y,z = map(float, parts[base+1: base+7])
                yaw_deg = float(parts[base+7])
            except ValueError:
                continue
            yaw = math.radians(yaw_deg)
            params = np.array([x,y,z,l,w,h,yaw], dtype=np.float32)
            frame2objs[frame_id].append((params, color_rgb, line_thickness))
    print(f"[INFO-{tag}] 解析完成：{len(frame2objs)} 帧")
    return frame2objs, frame_ids

def _load_test_json(test_json_path):
    jd = json.load(open(test_json_path,"r"))
    if isinstance(jd, list): jd = jd[0]
    cam2pixels = dict(jd.get('cam2img', {}))
    cam2pixels.update(jd.get('cam2img_fisheye', {}))
    lidar2cams = dict(jd.get('lidar2cam', {}))
    lidar2cams.update(jd.get('lidar2cam_fisheye', {}))
    distort_params = jd.get('distort_fisheye', {})

    ts_from_json = set()
    for key in ['frames','samples','timestamps']:
        if key in jd and isinstance(jd[key], list):
            for it in jd[key]:
                if isinstance(it, dict):
                    for k in ['timestamp','ts','time','name']:
                        if k in it: ts_from_json.add(str(it[k]))
                else:
                    ts_from_json.add(str(it))
    return cam2pixels, lidar2cams, distort_params, ts_from_json

# ===== 新增：动态发现相机 + 与标定取交集 =====

def _detect_available_cameras(samples_root):
    if not os.path.isdir(samples_root): return []
    cams = [d for d in os.listdir(samples_root)
            if d.startswith("camera_") and os.path.isdir(os.path.join(samples_root, d))]
    cams = sorted(cams)
    print("[INFO] detected cameras (fs):", cams)
    return cams

def _cam_type(name):
    return "pinhole" if name.endswith("_0") else ("fisheye" if name.endswith("_8") else "unknown")

def _choose_cameras_by_calib(samples_root, cam2pixels, lidar2cams, distort_params):
    fs_cams = set(_detect_available_cameras(samples_root))
    calib_cams = set(cam2pixels.keys()) | set(lidar2cams.keys()) | set(distort_params.keys())
    cams = sorted(fs_cams & calib_cams) if calib_cams else sorted(fs_cams)
    miss_fs = sorted(calib_cams - fs_cams)
    miss_calib = sorted(fs_cams - calib_cams)
    if miss_fs: print("[WARN] calib exists but NO images for:", miss_fs)
    if miss_calib: print("[WARN] images exist but NO calib for:", miss_calib, "→ will ignore these cams")
    print("[INFO] using cameras (fs∩calib):", cams)
    pinhole = [c for c in cams if _cam_type(c)=="pinhole"]
    fisheye = [c for c in cams if _cam_type(c)=="fisheye"]
    unknown = [c for c in cams if _cam_type(c)=="unknown"]
    if unknown: print("[WARN] unknown camera types (neither _0 nor _8):", unknown)
    return cams, pinhole, fisheye

def _pick_reference_cam(samples_root, cams):
    """选一个存在的相机作为参考相机来遍历时间戳（优先针孔）。"""
    pinhole = [c for c in cams if _cam_type(c)=="pinhole"]
    prefer = pinhole[0] if pinhole else (cams[0] if cams else None)
    if prefer is None:
        return None, []
    ref_dir = os.path.join(samples_root, prefer)
    img_files = sorted(glob.glob(os.path.join(ref_dir, "*.jpg")))
    print(f"[INFO] reference camera: {prefer}, frames={len(img_files)}")
    return prefer, img_files

# ===== 拼接辅助（有几路就拼几路） =====

def _hstack_images(img_list):
    if not img_list: return None
    # 针对不同宽/高做对齐：统一高度为最小高度
    hs = [im.shape[0] for im in img_list]
    target_h = min(hs)
    resized = [cv2.resize(im, (int(im.shape[1]*target_h/im.shape[0]), target_h)) for im in img_list]
    return cv2.hconcat(resized) if len(resized) > 1 else resized[0]

def _vstack_images(img_top, img_bottom):
    if img_top is None: return img_bottom
    if img_bottom is None: return img_top
    # 统一宽度为较小宽
    w = min(img_top.shape[1], img_bottom.shape[1])
    top = cv2.resize(img_top, (w, int(img_top.shape[0]*w/img_top.shape[1])))
    bot = cv2.resize(img_bottom, (w, int(img_bottom.shape[0]*w/img_bottom.shape[1])))
    return cv2.vconcat([top, bot])

# ============================= 可视化主流程 =============================

def _render_sequence(sequence, mode="overlay", make_video=True):
    root_path = "/rss/zhanghui/4D_label/dataset_track"
    samples_root = os.path.join(root_path, sequence, "samples")

    test_json = os.path.join(root_path, sequence, "scences/test.json")
    print(f"[INFO] Loading test.json: {test_json}")
    try:
        cam2pixels, lidar2cams, distort_params, ts_from_json = _load_test_json(test_json)
    except Exception as e:
        print(f"[ERROR] load test.json fail: {e}")
        return

    det_root = os.path.join(root_path, "merged", sequence)
    trk_root = os.path.join(root_path, "offline_tracked", sequence)
    det_files = _collect_all_files(det_root, "*.txt", track_mode=False)
    trk_files = _collect_all_files(trk_root, "*.txt", track_mode=True)
    print(f"[INFO] DET files: {len(det_files)} | TRK files: {len(trk_files)}")

    det_map, det_ids = defaultdict(list), []
    trk_map, trk_ids = defaultdict(list), []
    if mode in ("detection","overlay"):
        det_map, det_ids = _parse_result_files(det_files, color_rgb=(0,1,0), tag="DET", line_thickness=8)
    if mode in ("tracking","overlay"):
        trk_map, trk_ids = _parse_result_files(trk_files, color_rgb=(1,0,0), tag="TRK", line_thickness=2)

    # 动态相机
    cams_all, pinhole_cams, fisheye_cams = _choose_cameras_by_calib(samples_root, cam2pixels, lidar2cams, distort_params)
    if not cams_all:
        print("[ERROR] no usable cameras under samples/."); return

    # 选参考相机（不再强制 camera_0_0）
    ref_cam, img_files = _pick_reference_cam(samples_root, cams_all)
    if ref_cam is None or not img_files:
        print("[ERROR] no images found in any camera folder."); return

    out_root = {
        "detection": os.path.join("/rss/zhanghui/4D_label/visual_combine/detection_results", sequence),
        "tracking":  os.path.join("/rss/zhanghui/4D_label/visual_combine/tracking_results",  sequence),
        "overlay":   os.path.join("/rss/zhanghui/4D_label/visual_combine/det_trk_results",   sequence),
    }[mode]
    if os.path.exists(out_root):
        import shutil; shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    processed_frames, saved_images = 0, 0

    for img_ref in tqdm(img_files):
        frame_id = os.path.basename(img_ref).replace(".jpg","")

        # 聚合该帧要画的框
        obj_frame = []
        if mode in ("detection","overlay") and frame_id in det_map: obj_frame += det_map[frame_id]
        if mode in ("tracking","overlay")  and frame_id in trk_map: obj_frame += trk_map[frame_id]
        if not obj_frame:
            processed_frames += 1
            continue

        # —— 针孔行 ——（有几路画几路，>=1 就输出；若为 0 则整行省略）
        pinhole_imgs = []
        for cam in pinhole_cams:
            p = os.path.join(samples_root, cam, frame_id + ".jpg")
            img = cv2.imread(p)
            if img is None:
                # 降噪：只在很少量时打印；这里简单打印
                # print(f"[WARN] missing image: {p}")
                continue
            if cam not in cam2pixels or cam not in lidar2cams:
                # 没有标定就不能投影，直接跳过该路
                # print(f"[WARN] missing calib for {cam}, skip")
                continue
            K = np.array(cam2pixels[cam])[:3,:3]
            T = np.array(lidar2cams[cam])
            img_drawn = draw_box_on_img_per(obj_frame, img.copy(), K, T, distort=None)
            # 尺度统一：高度 800
            img_drawn = cv2.resize(img_drawn, (int(img_drawn.shape[1]*800/img_drawn.shape[0]), 800))
            pinhole_imgs.append(img_drawn)
        pinhole_row = _hstack_images(pinhole_imgs)  # None 或 一张/多张

        # —— 鱼眼行 ——（同理）
        fisheye_imgs = []
        for cam in fisheye_cams:
            p = os.path.join(samples_root, cam, frame_id + ".jpg")
            img = cv2.imread(p)
            if img is None:
                # print(f"[WARN] missing image: {p}")
                continue
            if cam not in cam2pixels or cam not in lidar2cams:
                # print(f"[WARN] missing calib for {cam}, skip")
                continue
            K = np.array(cam2pixels[cam])[:3,:3]
            T = np.array(lidar2cams[cam])
            distort = np.array(distort_params.get(cam, [0,0,0,0])) if cam in distort_params else None
            img_drawn = draw_box_on_img_per(obj_frame, img.copy(), K, T, distort=distort)
            img_drawn = cv2.resize(img_drawn, (int(img_drawn.shape[1]*800/img_drawn.shape[0]), 800))
            fisheye_imgs.append(img_drawn)
        fisheye_row = _hstack_images(fisheye_imgs)

        if pinhole_row is None and fisheye_row is None:
            # 这一帧没有任何路可画（例如该帧只在别的相机存在）
            processed_frames += 1
            continue

        final_img = _vstack_images(pinhole_row, fisheye_row)

        save_path = os.path.join(out_root, frame_id + ".jpg")
        cv2.imwrite(save_path, final_img)
        processed_frames += 1
        saved_images += 1

    print(f"[OK] Processed {processed_frames} frames, saved {saved_images} images to {out_root}")

    if make_video and saved_images > 0:
        videos_root = {
            "detection": "/rss/zhanghui/4D_label/visual_combine/detection_videos",
            "tracking":  "/rss/zhanghui/4D_label/visual_combine/tracking_videos",
            "overlay":   "/rss/zhanghui/4D_label/visual_combine/overlay_videos",
        }[mode]
        video_sequence_path = os.path.join(videos_root, sequence)
        if os.path.exists(video_sequence_path):
            import shutil; shutil.rmtree(video_sequence_path)
        os.makedirs(video_sequence_path, exist_ok=True)
        out_video = os.path.join(videos_root, sequence, f"{mode}.mp4")
        images_to_video_ffmpeg(out_root, out_video, fps=10, limit_first_n=None, slow_factor=1.0)
        print(f"[OK] {mode} video saved: {out_video}")
    else:
        print("[INFO] No video generated (either disabled or no images).")

# ============================= 兼容旧接口 =============================

def detection_results_on_multi_cams(sequence):
    _render_sequence(sequence, mode="detection", make_video=True)

def tracking_results_on_multi_cams(sequence):
    _render_sequence(sequence, mode="tracking", make_video=True)

def detection_and_tracking_results_on_multi_cams(sequence):
    _render_sequence(sequence, mode="overlay", make_video=True)

def _has_encoder(name="libx264"):
    try:
        subprocess.run(["ffmpeg","-hide_banner","-h",f"encoder={name}"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def combine_detection_and_tracking_videos(sequence):
    import shutil
    detection_videos_root = os.path.join("/rss/zhanghui/4D_label/visual_combine/detection_videos", sequence)
    tracking_videos_root  = os.path.join("/rss/zhanghui/4D_label/visual_combine/tracking_videos",  sequence)
    combined_videos_root  = os.path.join("/rss/zhanghui/4D_label/visual_combine/combined_videos", sequence)

    if os.path.exists(combined_videos_root): shutil.rmtree(combined_videos_root)
    os.makedirs(combined_videos_root, exist_ok=True)

    detection_video = os.path.join(detection_videos_root, "detection.mp4")
    tracking_video  = os.path.join(tracking_videos_root,  "tracking.mp4")
    if not os.path.exists(detection_video):
        print(f"[ERROR] Detection video not found: {detection_video}"); return
    if not os.path.exists(tracking_video):
        print(f"[ERROR] Tracking video not found: {tracking_video}"); return

    combined_video = os.path.join(combined_videos_root, "combined.mp4")
    filter_complex = (
        "[0:v]fps=10,scale=-2:1080:flags=bicubic[v0];"
        "[1:v]fps=10,scale=-2:1080:flags=bicubic[v1];"
        "[v0][v1]hstack=inputs=2[v]"
    )
    use_x264 = _has_encoder("libx264")
    if use_x264:
        cmd = ["/usr/bin/ffmpeg","-y","-v","error","-i",detection_video,"-i",tracking_video,
               "-filter_complex",filter_complex,"-map","[v]",
               "-c:v","libx264","-crf","23","-preset","slow",
               "-pix_fmt","yuv420p","-movflags","+faststart",combined_video]
    else:
        cmd = ["/usr/bin/ffmpeg","-y","-v","error","-i",detection_video,"-i",tracking_video,
               "-filter_complex",filter_complex,"-map","[v]",
               "-c:v","libopenh264","-profile:v","main","-level","3.1","-bf","0",
               "-b:v","6M","-maxrate","6M","-bufsize","12M",
               "-pix_fmt","yuv420p","-movflags","+faststart",combined_video]
    print("[CMD]"," ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] Combined video saved: {combined_video}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] combine videos failed: {e}")

# ============================= CLI =============================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True,
                        help="例如 train_sh_3d_road_2_20250531_5000_lx")
    parser.add_argument('--root-path', type=str, default='/rss/zhanghui/4D_label/origin',
                        help='保留以兼容旧参数')
    parser.add_argument('--vis-type', type=str, default='detection',
                        choices=['detection','tracking','both','overlay','combine'],
                        help="detection=只画检测; tracking=只画跟踪; both=先各自出视频再拼接; overlay=同图叠加; combine=只拼接已生成视频")
    args = parser.parse_args()

    if args.vis_type == 'detection':
        detection_results_on_multi_cams(args.sequence)
    elif args.vis_type == 'tracking':
        tracking_results_on_multi_cams(args.sequence)
    elif args.vis_type == 'overlay':
        detection_and_tracking_results_on_multi_cams(args.sequence)
    elif args.vis_type == 'both':
        detection_results_on_multi_cams(args.sequence)
        tracking_results_on_multi_cams(args.sequence)
        combine_detection_and_tracking_videos(args.sequence)
    elif args.vis_type == 'combine':
        combine_detection_and_tracking_videos(args.sequence)
