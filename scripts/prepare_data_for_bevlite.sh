#!/bin/bash

export PYTHONPATH=./

raw_4d_label_dataset_root=/data1/turbo_data/4D_label_dataset_one_frame
location=sh
road=2
mode=Intersection
# dataset_name=train_sh_3d_road_2_20250713_5_one_frame
# dataset_name=train_sh_3d_road_2_20250714_5_one_frame
# dataset_name=train_sh_3d_road_2_20250718_5_one_frame
dataset_name=train_sh_3d_road_2_20250712_5_one_frame

# # ensamble数据：
# location=sh
# road=5
# mode=Intersection
# # dataset_name=train_sh_3d_road_5_20250714_5_one_frame
# dataset_name=train_sh_3d_road_5_20250718_5_one_frame
save_root=/data1/turbo_data/RALG/data/3.0_lite/Intersection/4d_label/p3
one_frame=1
# 目前暂只有6号路口设置了roi范围过滤
extra_filter_roi=0

mkdir -p $save_root


echo "road: ${location}_${road}"
echo "mode: ${mode}"
echo "dataset_name: ${dataset_name}"
echo "one_frame: ${one_frame}"
echo "extra_filter_roi: ${extra_filter_roi}"


# 新版本：进行大车通过（用于半自动，去除了全类别通过）、全类别通过（连续）、全类别通过-单帧
# 注意：one_frame模式，在没有时序数据时进行单帧！！！！！！
if [ "$one_frame" == "1" ]; then
    if [ "$extra_filter_roi" == "1" ]; then
        python  parse_data/tools/prepare_data_for_bevlite.py \
            --raw_4d_label_dataset_root $raw_4d_label_dataset_root \
            --location $location \
            --road $road \
            --mode $mode \
            --dataset_name $dataset_name \
            --new_format \
            --extra_filter_roi \
            --save_root $save_root \
            --strict \
            --one_frame
    else
        python  parse_data/tools/prepare_data_for_bevlite.py \
            --raw_4d_label_dataset_root $raw_4d_label_dataset_root \
            --location $location \
            --road $road \
            --mode $mode \
            --dataset_name $dataset_name \
            --new_format \
            --save_root $save_root \
            --strict \
            --one_frame
    fi
else
    if [ "$extra_filter_roi" == "1" ]; then
        python  parse_data/tools/prepare_data_for_bevlite.py \
            --raw_4d_label_dataset_root $raw_4d_label_dataset_root \
            --location $location \
            --road $road \
            --mode $mode \
            --dataset_name $dataset_name \
            --new_format \
            --extra_filter_roi \
            --save_root $save_root \
            --strict
    else
        python  parse_data/tools/prepare_data_for_bevlite.py \
            --raw_4d_label_dataset_root $raw_4d_label_dataset_root \
            --location $location \
            --road $road \
            --mode $mode \
            --dataset_name $dataset_name \
            --new_format \
            --save_root $save_root \
            --strict
    fi
fi