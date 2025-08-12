#!/bin/bash

export PYTHONPATH=./

# # 单帧数据:0721生产
# data_root=/data1/turbo_data/4D_label_dataset_one_frame/Intersection
# # dataset_name=train_sh_3d_road_6_20250528_5_one_frame
# # dataset_name=train_sh_3d_road_6_20250529_5_one_frame
# # dataset_name=train_sh_3d_road_6_20250530_5_one_frame
# # dataset_name=train_sh_3d_road_6_20250531_5_one_frame
# dataset_name=train_sh_3d_road_6_20250601_5_one_frame
# save_root=/data1/turbo_data/4D_label_dataset/prelabel/Intersection
# one_frame=1

# 单帧数据:0729生产
data_root=/data1/turbo_data/4D_label_dataset_one_frame/Intersection
# dataset_name=train_sh_3d_road_5_20250712_5_one_frame
dataset_name=train_sh_3d_road_5_20250713_5_one_frame
save_root=/data1/turbo_data/4D_label_dataset/prelabel/Intersection
one_frame=1

echo "data_root: ${data_root}"
echo "dataset_name: ${dataset_name}"
echo "save_root: ${save_root}"
echo "one_frame: ${one_frame}"

if [ "$one_frame" == "1" ]; then
    python parse_data/tools/generate_semi_autolabeling.py \
        --data_root $data_root \
        --dataset_name $dataset_name \
        --new_format \
        --strict \
        --save_root $save_root \
        --one_frame
else
    python parse_data/tools/generate_semi_autolabeling.py \
        --data_root $data_root \
        --dataset_name $dataset_name \
        --new_format \
        --strict \
        --save_root $save_root
fi
