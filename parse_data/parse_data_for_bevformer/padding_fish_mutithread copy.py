import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

fish_root = '/adga/lushiyong/bev_data/dataset/sh_i3_0501/images/'
txt_path = '/adga/lushiyong/bev_data/dataset/sh_i3_0501/sh_i3_0501.txt'

fish_root = '/adga/lushiyong/bev_data/adp_data/sh_i2_3199/kitti_result/images/'
txt_path = '/adga/lushiyong/bev_data/adp_data/sh_i2_3199/kitti_result/sh_i2_5744_noinfo.txt'

fish_root = '/adga/lushiyong/bev_data/dataset/sh_i2_0501/images/'
txt_path = '/adga/lushiyong/bev_data/dataset/sh_i2_0501/sh_i2_0501.txt'

fish_root = '/adga/lushiyong/bev_data/adp_data/sh_i3_6010/kitti_result/images/'
txt_path = '/adga/lushiyong/bev_data/adp_data/sh_i3_6010/kitti_result/sh_i3_6010.txt'

fish_root = '/adga/lushiyong/bev_data/dataset/sh_i4_0501/images/'
txt_path = '/adga/lushiyong/bev_data/dataset/sh_i4_0501/sh_i4_0501.txt'

fish_root = '/adga/lushiyong/bev_data/dataset/sh_i6_0501/images/'
txt_path = '/adga/lushiyong/bev_data/dataset/sh_i6_0501/sh_i6_0501.txt'

fish_root = '/adga/lushiyong/bev_data/dataset/sh_i5_0523_5w/'
txt_path = '/adga/lushiyong/bev_data/dataset/sh_i5_0523_5w/sh_i5_0523_5w.txt'

fish_root = '/adga/lushiyong/bev_data/adp_data/sh_i4_9013/kitti_result/images/'
txt_path = '/adga/lushiyong/bev_data/adp_data/sh_i4_9013/kitti_result/sh_i4_9013.txt'


fish_root = '/adga/lushiyong/bev_data/dataset/sh_i5_0526_5w/images'
txt_path = '/adga/lushiyong/bev_data/dataset/sh_i5_0526_5w/sh_i5_0526_5w.txt'

fish_root = '/adga/lushiyong/bev_data/dataset/sh_i5_0528_5w/images'
txt_path = '/adga/lushiyong/bev_data/dataset/sh_i5_0528_5w/sh_i5_0528_5w.txt'

fish_root = '/adga/lushiyong/bev_data/adp_data/sh_i6_8000/kitti_result/images/'
txt_path = '/adga/lushiyong/bev_data/adp_data/sh_i6_8000/kitti_result/sh_i6_8000.txt'

fish_root = '/adga/lushiyong/bev_data/adp_data/sh_i5_4d_3500/road_sh5/images'
txt_path = '/adga/lushiyong/bev_data/adp_data/sh_i5_4d_3500/road_sh5/sh_i5_4d_3500.txt'

fish_root = '/adga/lushiyong/bev_data/adp_data/sh_i6_4d_1900/images'
txt_path = '/adga/lushiyong/bev_data/adp_data/sh_i6_4d_1900/sh_i6_4d_1900.txt'

fish_root = '/adga/lushiyong/bev_data/test_dataset/sh_i5_4d/images'
txt_path = '/adga/lushiyong/bev_data/adp_data/sh_i5_4d/sh_i5_4d_test.txt'

fish_dirs = [
    "camera_0_fisheye",
    "camera_1_fisheye",
    "camera_2_fisheye",
    "camera_3_fisheye",
]

# with open(txt_path, 'r') as f:
#     file_list = [line.strip() + '.jpg' for line in f.readlines()]

file_list = os.listdir(os.path.join(fish_root, 'camera_0_8'))


def process_image(fish_dir, img_name):
    try:
        root_fish_dir = fish_dir[:9] + '8'
        img_path = os.path.join(fish_root, root_fish_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Failed to read image {img_path}")
            return

        if img.shape[0] != 720:
            resized_img = cv2.resize(img, (720, 720))

            left = (1280 - 720) // 2
            right = 1280 - 720 - left
            top = bottom = 0

            padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            padded_img = img

        save_path = os.path.join(fish_root, fish_dir, img_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, padded_img)

    except Exception as e:
        print(f"Error processing {img_name} in {fish_dir}: {e}")

# 多线程处理
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for fish_dir in fish_dirs:
        for img_name in file_list:
            futures.append(executor.submit(process_image, fish_dir, img_name))

    # 可选：进度条显示
    for _ in tqdm(futures):
        _.result()