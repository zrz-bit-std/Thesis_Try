import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# fish_root = '/adga/lushiyong/vis_anno/dataset/sh_i2/sh_i2_4d_val_set_sample_copy/send/1'
# fish_root = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/images'
# txt_path = '/adga/lushiyong/vis_anno/dataset/hm_i7_0830/hm_i7_0830_5s.txt'

fish_root = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/images'
txt_path = '/adga/lushiyong/vis_anno/dataset/wh_i4_0830/wh_i4_0830_5s.txt'

with open(txt_path, 'r') as f:
    # i10_sift_filename = [line.strip() for line in f]
    txt_file_names = [line.strip() for line in f]


fish_dirs = [
    # "camera_0_fisheye",
    "camera_1_fisheye",
    "camera_2_fisheye",
    "camera_3_fisheye",
]

file_list = os.listdir(os.path.join(fish_root, 'camera_3_8'))

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
        for img_name in txt_file_names:
            img_name = img_name+'.jpg'
            futures.append(executor.submit(process_image, fish_dir, img_name))

    # 可选：进度条显示
    for _ in tqdm(futures):
        _.result()