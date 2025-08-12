#!/bin/bash

export PYTHONPATH=./

# # 注意：注意保存数据，不要覆盖之前的数据！！！！

intersection_config_file=/data1/turbo_data/huben/projects/BEVFUSION/configs/camera+lidar+fisheye/raw_bevfusion_plus_fisheye_4D_prelabel_v2_200x200m.yaml

# 半自动标注数据下载位置
data_root=/data1/turbo_data/RALG/data/3.0_pro/4d/semi_autolabel/Intersection/label/shanghai_road_6/p2_one_frame
dataset_names=(
    "train_sh_3d_road_6_20250601_5_one_frame_vehicle"
    "train_sh_3d_road_6_20250531_5_one_frame_vehicle"
    "train_sh_3d_road_6_20250530_5_one_frame_vehicle"
    "train_sh_3d_road_6_20250529_5_one_frame_vehicle"
    "train_sh_3d_road_6_20250528_5_one_frame_vehicle"
)
# 注意：首先生成dataset_names对应的pkl数据, 参见BEVFUSION中的pkl真值数据生成过程！！！
ann_file=/data1/turbo_data/RALG/data/3.0_pro/4d/semi_autolabel/Intersection/version/train/v2_one_frame/shanghai_6/mogo_infos_train.pkl
location=sh
road=6
mode=intersection
save_root=/data1/turbo_data/RALG/data/3.0_lite/Intersection/semi_autolabel/p2_one_frame
mkdir -p $save_root

echo "ann_file: $ann_file"
echo "road_id: ${location}_${road}"
echo "mode: $mode"
echo "dataset_name: $dataset_name"
echo "save_root: $save_root"


params_length=${#dataset_names[@]}

# processing
for ((i=0; i<params_length; i++)); do
    echo "Processing $((i+1)) dataset: ${dataset_names[$i]} ..."
    python parse_data/tools/prepare_data_for_bevlite_from_semiautolabel.py $intersection_config_file \
        --data_root $data_root \
        --ann_file $ann_file \
        --location $location \
        --road $road \
        --mode $mode \
        --dataset_name ${dataset_names[$i]} \
        --extra_filter_roi \
        --save_root $save_root
done

