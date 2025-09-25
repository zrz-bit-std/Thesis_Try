
# echo "========== 🚩 Run Vehicle PRM inference =========="
# python test.py --cfg_file cfgs/ref_model_cfgs/vehicle_prm_model.yaml \
#     --ckpt ../output/ref_model_cfgs/vehicle_prm_model/fusion_0924_3w5/ckpt/checkpoint_epoch_50.pth \
#     --extra_tag val_fusion_0924_3w5 --save_to_file

# echo "========== 🚩 Run Vehicle GRM inference =========="
# python test.py --cfg_file cfgs/ref_model_cfgs/vehicle_grm_model.yaml \
#     --ckpt ../output/ref_model_cfgs/vehicle_grm_model/fusion_0924_3w5/ckpt/checkpoint_epoch_30.pth \
#     --extra_tag val_fusion_0924_3w5 --save_to_file

# # ==========================================================================
# # ==========================================================================

# echo "========== 🚩 Run Truck GRM inference =========="
# python test.py --cfg_file cfgs/ref_model_cfgs/truck_grm_model.yaml \
#     --ckpt ../output/ref_model_cfgs/truck_grm_model/fusion_0924_3w5/ckpt/checkpoint_epoch_30.pth \
#     --extra_tag val_fusion_0924_3w5 --save_to_file

# echo "========== 🚩 Run Truck PRM inference =========="
# python test.py --cfg_file cfgs/ref_model_cfgs/truck_prm_model.yaml \
#     --ckpt ../output/ref_model_cfgs/truck_prm_model/fusion_0924_3w5/ckpt/checkpoint_epoch_50.pth \
#     --extra_tag val_fusion_0924_3w5 --save_to_file

# # ==========================================================================
# # ==========================================================================

# echo "========== 🚩 Run Vehicle GRM training =========="
# python train.py --cfg_file cfgs/ref_model_cfgs/vehicle_grm_model.yaml --extra_tag camera_0924_1w5

# echo "========== 🚩 Run Truck GRM training =========="
# python train.py --cfg_file cfgs/ref_model_cfgs/truck_grm_model.yaml --extra_tag fusion_0924_3w5


echo "========== 🚩 Run Vehicle PRM training =========="
python train.py --cfg_file cfgs/ref_model_cfgs/vehicle_prm_model.yaml --extra_tag camera_0924_1w5

# echo "========== 🚩 Run Truck PRM training =========="
# python train.py --cfg_file cfgs/ref_model_cfgs/truck_prm_model.yaml --extra_tag fusion_0924_3w5

