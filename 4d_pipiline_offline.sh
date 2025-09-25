#!/bin/bash
set -e
export PYTHONPATH=$(pwd):$PYTHONPATH

# ========== 环境变量 ==========
data_name=train_sh_3d_road_2_20250531_5000_lx
src_dir=/rss/yuanqingwen/data/4D_label/dataset_track
track_dir=/rss/yuanqingwen/data/4D_label/dataset_track/offline_tracked

val_tag=shanghai2
save_combine_tag=shanghai2_combine
grm_config=cfgs/ref_model_cfgs/vehicle_grm_model.yaml
# grm_config=cfgs/ref_model_cfgs/truck_grm_model.yaml
# grm_config=cfgs/ref_model_cfgs/cyclist_grm_model.yaml
grm_vehicle_ckpt=/rss/yuanqingwen/git_version/DetZero/refining/output/ref_model_cfgs/vehicle_grm_model/new_default2/ckpt/checkpoint_epoch_30.pth
# grm_truck_ckpt=/rss/yuanqingwen/git_version/DetZero/refining/output/ref_model_cfgs/truck_grm_model/new_default2_truck/ckpt/checkpoint_epoch_30.pth
# grm_cyclist_ckpt=/rss/yuanqingwen/git_version/DetZero/refining/output/ref_model_cfgs/cyclist_grm_model/default/ckpt/checkpoint_epoch_500.pth

prm_config=cfgs/ref_model_cfgs/vehicle_prm_model.yaml
# prm_config=cfgs/ref_model_cfgs/truck_prm_model.yaml
# prm_config=cfgs/ref_model_cfgs/cyclist_prm_model.yaml
prm_vehicle_ckpt=/rss/yuanqingwen/git_version/DetZero/refining/output/ref_model_cfgs/vehicle_prm_model/new_default2/ckpt/checkpoint_epoch_50.pth
# prm_truck_ckpt=/rss/yuanqingwen/git_version/DetZero/refining/output/ref_model_cfgs/truck_prm_model/new_default2_truck/ckpt/checkpoint_epoch_50.pth
# prm_cyclist_ckpt=/rss/yuanqingwen/git_version/DetZero/refining/output/ref_model_cfgs/cyclist_prm_model/new_default/ckpt/checkpoint_epoch_30.pth


# 使用IFS和read配合下划线分割字符串
IFS='_' read -r part1 location part3 part4 road_id part6 part7 <<< "$data_name"

# ==========================================================================
# ========== Step 1: 数据生成：将tracking的结果转换成refine的输入pkl ==========
# ==========================================================================
echo "============== 🚩 Generate multi-class data.pkl =========="
python ./daemon/prepare_object_data_camera_sparse.py \
  --track-root ${track_dir} \
  --lidar-root ${src_dir} \
  --data-name ${data_name} \


# ===============================================================
# ========== Step 2: grm model inference：多类别分类别infer ============
# ===============================================================
echo "============== 🚩 Infer Vehicle GRM =========="
python ./refining/tools/test.py \
  --cfg_file ${grm_config} \
  --ckpt ${grm_vehicle_ckpt} \
  --extra_tag ${val_tag} \
  --save_to_file


# ===============================================================
# ========== Step 3: prm model inference：多类别分类别infer ============
# ===============================================================
echo "============== 🚩 Infer Vehicle PRM =========="
python ./refining/tools/test.py \
  --cfg_file ${prm_config} \
  --ckpt ${prm_vehicle_ckpt} \
  --extra_tag ${val_tag} \
  --save_to_file


# ===============================================================
# ========== Step 4: 多类别多模型infer结果合并 ============
# ===============================================================
echo "============== 🚩 Combine multi-class GRM+PRM results =========="
python ./daemon/combine_output.py \
  --val-tag ${val_tag} \
  --save-combine-tag ${save_combine_tag} \
