CONFIG_PATH=/adga/lushiyong/bevformer_road_8cam/projects/configs/bevformer_road_mogo_lsy/test_hm_i7_mini_bevformer_r101_200m_200_scale_0.7.py
CHECKPOINT_PATH=/adga/lushiyong/bevformer_road_8cam/work_dirs/mogobev/bevformer_base_200m_300_mix_all_77187_sampler/epoch_50.pth
TXT_SAVE_PATH=/adga/lushiyong/bev_8cam_compare_result/hm_i7_mini_bevformer_base_200m_200_scale_0.7_trainon_allmix_77187/conf0.3_ep50_0909
DST_DIR=/adga/lushiyong/vis_anno/dataset/hm_i7_0830/kitti_result/

cd /adga/lushiyong/bevformer_road_8cam
python tools/test/4d_inference_pipeline.py \
--config ${CONFIG_PATH} \
--checkpoint ${CHECKPOINT_PATH} \
--save-path ${TXT_SAVE_PATH}

cd -
cp -r ${TXT_SAVE_PATH}/3d_url ${DST_DIR}