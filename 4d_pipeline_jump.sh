#!/bin/bash
# set -e
# export PYTHONPATH=$(pwd):$PYTHONPATH

# # ========== 环境变量 ==========
data_name=train_sh_3d_road_5_20250524_5000_lx
# src_dir=/rss/RALG/data/3.0_pro/Inter+Road/label/shanghai_road_5/p1_lx
dst_dir=/data1/turbo_data/lishuiayin/tracking/4D_label/tracking/detzero_track
# bev_config=/rss/liumengyuan/4D_label/BEVFUSION/configs/camera+lidar+fisheye/raw_bevfusion_plus_fisheye_4D_prelabel_v2_200x200m.yaml
# ckpt_path=/rss/liumengyuan/BEVFUSION/tmp_output/bevfusion_lmy_0723/epoch_19.pth
# calib=/rss/huben/projects/BevCalib
track_cfg=/data1/turbo_data/lishuiayin/tracking/4D_label/tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track_mogo_241216.yaml
mode=intersection
calib_version=v20250605
# 使用IFS和read配合下划线分割字符串
IFS='_' read -r part1 location part3 part4 road_id part6 part7 <<< "$data_name"

# ===========================================================
# ========== Step 1: 数据解析：把采集好的数据进行解析 ==========
# ===========================================================
# echo "============== 🚩 Generate test.json =========="
# python ./parse_data/parse_data_for_bevpro/create_data_for_bevpro_v2.py \
#   --data_root ${src_dir} \
#   --save_dir ${dst_dir} \
#   --dataset_name ${data_name} \
#   --location ${location} \
#   --road ${road_id} \
#   --calib ${calib} \
#   --calib_version ${calib_version} \
#   --mode ${mode} \


# echo "========== 🚩 Generate pkl =========="
python ./parse_data/parse_data_for_bevpro/create_data_mogo_wrh.py \
  --data_root ${dst_dir}/${data_name} \
  --use_fisheye

# =====================================
# ========== Step 2: 执行推理 ==========
# =====================================
# echo "========== 🚩 Run BEVFusion inference =========="
# torchpack dist-run -np 8 python ./BEVFUSION/tools/visualize/get_infer_res.py ${bev_config} \
#   --checkpoint ${ckpt_path} \
#   --bbox-score 0.3 \
  # --dataset_root ${dst_dir}/${data_name}
# torchpack dist-run -np 8 python ./BEVFUSION/tools/visualize/get_infer_res.py ${bev_config} \
#   --checkpoint ${ckpt_path} \
#   --bbox-score 0.3 \
#   --dataset_root /rss/RALG/data/3.0_pro/Inter+Road/version/val/v3/shanghai_5/

# echo "========== 🚩 Split and synthesize the required pkl for tracking input =========="
# python ./parse_data/parse_data_for_bevpro/make_data_for_track.py \
#     --save_dir ${dst_dir}/merged \
#     --origin_dir ${src_dir} \
#     --det_dir ${dst_dir} \
#     --track_dir ${dst_dir}/offline_tracked \
#     --dataset_name ${data_name} \

# =====================================
# ======== Step 3: 执行4D融合跟踪=======
# =====================================

# echo "========== 🚩 Perform 4D tracking =========="
# cd tracking
# pip install -e .
# cd ../utils
# python setup.py develop
# cd ../
# python ./tracking/tools/run_track_20250224.py \
#   --track_dir ${dst_dir}/offline_tracked \
#   --dataset_name ${data_name} \
#   --cfg_file ${track_cfg} \

# TODO：轨迹曲线拟合功能开发


# echo "========== 🚩 Tracking result split =========="
# python ./tracking/utils_track/tracking_data_split.py \
#   --track_dir ${dst_dir}/offline_tracked \
#   --merged_dir ${dst_dir}/merged \
#   --dataset_name ${data_name} \

# =====================================
# ======== Step 4: 生成可视化结果=======
# =====================================
# echo "========== 🚩 Generate a visualization demo =========="
# python ./visual_data/data_show.py \
#   --origin_dir ${src_dir} \
#   --label_dir ${dst_dir}/labels \
#   --merged_dir ${dst_dir}/merged \
#   --bev_pro_dir ${dst_dir} \
#   --tracked_dir ${dst_dir}/offline_tracked  \
#   --dataset_name ${data_name} \

