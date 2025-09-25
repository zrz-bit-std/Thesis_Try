#!/bin/bash
set -e
export PYTHONPATH=$(pwd):$PYTHONPATH

# # ========== 环境变量 ==========
data_name=train_sh_3d_road_6_20250517_5000_lx
src_dir=/rss/zhanghui/4D_label/origin
dst_dir=/rss/zhanghui/4D_label/dataset_track
bev_config=/rss/yuanqingwen/4D_label/BEVFUSION/configs/camera+lidar+fisheye/raw_bevfusion_plus_fisheye_4D_prelabel_v2_200x200m_lmy.yaml
ckpt_path=/rss/liumengyuan/BEVFUSION/tmp_output/bevfusion_lmy_0723/epoch_19.pth
# bev_config=/data1/turbo_data/RALG/models/version/4D_prelabel/Intersection/v0728/raw_bevfusion_plus_fisheye_4D_prelabel_v2_200x200m.yaml
# ckpt_path=/data1/turbo_data/RALG/models/version/4D_prelabel/Intersection/v0728/epoch_19.pth

calib=/rss/huben/projects/BevCalib
track_cfg=/rss/lishuaiyin/4D_label/tracking/tools/cfgs/tk_model_cfgs/waymo_detzero_track_mogo_241216.yaml
mode=intersection
calib_version=v20250605
# 使用IFS和read配合下划线分割字符串
IFS='_' read -r part1 location part3 part4 road_id part6 part7 <<< "$data_name"

pkl_dir="/adga/lushiyong/vis_anno/dataset/4d/train_sh_3d_road_6_20250517_5000_lx/kitti_result/query_feats"
dst_pkl_dir="${dst_dir}/${data_name}/model_pred_pkl"


# ===========================================================
# ========== Step 1: 数据解析：把采集好的数据进行解析 ==========
# ===========================================================
# echo "============== 🚩 Generate test.json =========="
# python ./parse_data/parse_data_for_bevpro/create_data_for_bevpro_v2_lisy.py \
#   --data_root ${src_dir} \
#   --save_dir ${dst_dir} \
#   --dataset_name ${data_name} \
#   --location ${location} \
#   --road ${road_id} \
#   --calib ${calib} \
#   --calib_version ${calib_version} \
#   --mode ${mode} \


# ===========================================================
# ========== Step 2: 数据解析：将数据放到指定路径下 ==========
# ===========================================================
# echo "============== 🚩 执行cp命令 =========="
# # # # # # # # # 检查源目录是否存在
# if [ ! -d "$pkl_dir" ]; then
#     echo "源目录不存在: $pkl_dir"
#     exit 1
# fi
# # 确保目标目录存在
# mkdir -p "$dst_pkl_dir"
# # 拷贝所有pkl文件到目标目录
# cp "$pkl_dir"/*.pkl "$dst_pkl_dir"/

# echo "已成功将 $pkl_dir 目录下的所有pkl文件拷贝到 $dst_pkl_dir"

# # # ===========================================================
# # # ========== Step 2: pkl文件重命名 ==========
# # # ===========================================================
# python ./visual_data/rename_pkls.py \
#   --src_dir ${dst_pkl_dir} \
#   --origin_data_dir ${src_dir}/${data_name}/original_data \
#   --output_dir ${dst_pkl_dir}

# # # # # # # # ============================将pkl文件转化为txt文件 ==================
# python ./visual_data/process_pkl_to_txt.py \
#   --pkl_dir ${dst_pkl_dir} \
#   --output_dir ${dst_dir}/${data_name}/model_pred


python ./parse_data/parse_data_for_bevpro/make_data_for_track_pkl.py \
    --save_dir ${dst_dir}/merged \
    --origin_dir ${src_dir} \
    --det_dir ${dst_dir} \
    --track_dir ${dst_dir}/offline_tracked \
    --dataset_name ${data_name} \

# # # =====================================
# # # ======== Step 3: 执行4D融合跟踪=======
# # # =====================================

# # # echo "========== 🚩 Perform 4D tracking =========="
# # # cd tracking
# # # pip install -e .
# # # cd ../utils
# # # python setup.py develop
# # # cd ../
# python ./tracking/tools/run_track_pkl.py \
#   --track_dir ${dst_dir}/offline_tracked \
#   --dataset_name ${data_name} \
#   --cfg_file ${track_cfg} \


# echo "========== 🚩 Tracking result split =========="
# python ./tracking/utils_track/tracking_data_split_pkl.py \
#   --track_dir ${dst_dir}/offline_tracked \
#   --merged_dir ${dst_dir}/merged \
#   --dataset_name ${data_name} \



# # # # # # =====================================
# # # # # ======== Step 4: 生成drop_box.pkl文件和 track_box.pkl文件=======

# python ./visual_data/drop_box_pkl.py \
#   --pred ${dst_dir}/merged/${data_name} \
#   --track ${dst_dir}/offline_tracked/${data_name} \
#   --droped ${dst_dir}/offline_tracked/${data_name} \
#   --clear_droped



# # # # # ==================================================
# # # # # ======== Step 5: 将跟踪结果可视化为视频_Lidar=======
# # # # # ==================================================
# #  echo "========== 🚩 Generate a bev mp4 demo =========="
# # python ./visual_data/demo_track_pre_lsy.py \
# #   --sequence ${data_name} \



# # # # # ===================================================
# # # # # ======== Step 5: 将跟踪结果可视化为视频_Camera=======
# # # # # ===================================================
# 两者都做 & 合并视频
# python ./visual_data/demo_track_bev_camera_combine.py \
#     --sequence ${data_name} \
#     --vis-type both

# 生成覆盖视频
# python ./visual_data/demo_track_bev_camera_combine.py \
#     --sequence ${data_name} \
#     --vis-type overlay



# =====================================
# ======== Step 6: 生成评测指标=======
# =====================================
# # echo "========== 🚩 Generate a metric  =========="

# python ./visual_data/mogo_dataset_track_lisy_pkl.py
