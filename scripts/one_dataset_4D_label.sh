#!/bin/bash

export PYTHONPATH=./

echo "*************** one data 4D Label begin time: $(date '+%Y-%m-%d %H:%M:%S')"

src_dir="$1"
dataset_name="$2"
location="$3"
road="$4"
calib="$5"
calib_version="$6"
data_type="$7"

# 模型推理参数
config_file="$8"
checkpoint="$9"
bbox_score="${10}"
mode="${11}"  # e.g, intersection
pole="${12}"

# # 1. prepare data
# src_dir=/data1/turbo_data/huben/data/test_4D_label
origin_data_root=$src_dir/origin/

if [[ "$pole" == "None" ]]; then
    model_res_root=$src_dir/$data_type/model_res
else
    model_res_root=$src_dir/$data_type/$pole/model_res
fi

det_save_dir=$model_res_root/bevpro
# # dataset_name=train_hy_4d_road_7_20250304_1x_tmp
# # location=hy
# # road=7
# # calib=/data1/turbo_data/huben/projects/BevCalib
# # calib_version=v20250218

# dataset_name=train_tx_4d_road_1_20250308_lx_tmp
# location=tx
# road=1
# calib=/data1/turbo_data/huben/projects/BevCalib
# calib_version=v20250228

echo "src_dir: $src_dir"
echo "model_res_root: $model_res_root"
echo "det_save_dir: $det_save_dir"
echo "dataset_name: $dataset_name"
echo "calib_version: $calib_version"


echo "============== Start make data for bevpro..."
# # 临时跳过，已经生成!!!
python parse_data/parse_data_for_bevpro/create_data_for_bevpro_v2.py \
    --data_root $origin_data_root \
    --save_dir $det_save_dir \
    --dataset_name $dataset_name \
    --location $location \
    --road $road \
    --calib $calib \
    --calib_version $calib_version \
    --mode $mode \
    --pole $pole


# # 2. prepare pkl data for model inference
det_pkl_data_root=$det_save_dir/$dataset_name

# # 临时跳过，已经生成!!!
echo "============== Start make pkl for model inference..."

if [[ "$mode" == "intersection" ]]; then
    python parse_data/parse_data_for_bevpro/create_data_mogo_wrh.py \
        --data_root $det_pkl_data_root \
        --use_fisheye  # 路口
else
    python parse_data/parse_data_for_bevpro/create_data_mogo_wrh.py \
        --data_root $det_pkl_data_root
fi


# # 3. model inference
np=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)  # equal gpu num
echo "np(gpu_num): $np"

echo "============== Start model inference..."
# # 临时跳过，已经生成!!!
torchpack dist-run -np $np python BEVFUSION/tools/visualize/get_infer_res.py $config_file \
    --checkpoint $checkpoint \
    --bbox-score $bbox_score \
    --dataset_root $det_pkl_data_root


# # 4. prepare pkl data for tracking
track_pkl_merged_save_dir=$model_res_root/merged
track_dir=$model_res_root/offline_tracked

# 临时跳过，已经生成!!!
echo "============== Start prepare data for tracking..."
python parse_data/parse_data_for_bevpro/make_data_for_track.py \
    --save_dir $track_pkl_merged_save_dir \
    --origin_dir $origin_data_root \
    --det_dir $det_save_dir \
    --track_dir $track_dir \
    --dataset_name $dataset_name


# # 5. tracking and split tracking results
track_model_cfg_file=tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track_mogo_241216.yaml

echo "============== Start tracking and split tracking results..."
# 临时跳过，已经生成!!!
python tracking/tools/run_track_20250224.py \
    --track_dir $track_dir \
    --dataset_name $dataset_name \
    --cfg_file $track_model_cfg_file

# 临时修复splited中txt的score
python tracking/utils_track/tracking_data_split.py \
    --track_dir $track_dir \
    --merged_dir $track_pkl_merged_save_dir \
    --dataset_name $dataset_name


# # 6. visualize data
if [[ "$pole" == "None" ]]; then
    result_label_dir=$src_dir/$data_type/labels
else
   result_label_dir=$src_dir/$data_type/$pole/labels 
fi

vis_pnum=20  #

echo "============== Start visualize tracking results..."

# 临时跳过，已经生成!!!
if [[ "$mode" == "intersection" ]]; then
    python visual_data/data_show_multiprocess.py $config_file \
        --location $location \
        --road $road \
        --origin_dir $origin_data_root \
        --label_dir $result_label_dir \
        --merged_dir $track_pkl_merged_save_dir \
        --bev_pro_dir $det_save_dir \
        --tracked_dir $track_dir \
        --dataset_name $dataset_name \
        --mode $mode \
        --use_fisheye \
        --pole $pole \
        --pnum $vis_pnum
else
    python visual_data/data_show_multiprocess.py $config_file \
        --location $location \
        --road $road \
        --origin_dir $origin_data_root \
        --label_dir $result_label_dir \
        --merged_dir $track_pkl_merged_save_dir \
        --bev_pro_dir $det_save_dir \
        --tracked_dir $track_dir \
        --dataset_name $dataset_name \
        --mode $mode \
        --pole $pole \
        --pnum $vis_pnum
fi

echo "*************** ${dataset_name} 4D Label finished time: $(date '+%Y-%m-%d %H:%M:%S')"
