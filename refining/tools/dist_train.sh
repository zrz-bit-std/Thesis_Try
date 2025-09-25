

CUDA_VISIBLE_DEVICES=0,1,2,3 
export CUDA_LAUNCH_BLOCKING=1
# os.environ["OMP_NUM_THREADS"] = "4"
torchrun --nproc_per_node=4 --master_port=28680 train.py \
    --cfg_file cfgs/ref_model_cfgs/vehicle_prm_model.yaml \
    --extra_tag fusion_0924_3w5 \
    --launcher pytorch \
    --batch_size 96 \
    --workers 4 

torchrun --nproc_per_node=4 --master_port=28690 train.py \
    --cfg_file cfgs/ref_model_cfgs/truck_prm_model.yaml \
    --extra_tag fusion_0924_3w5 \
    --launcher pytorch \
    --batch_size 96 \
    --workers 4 


# torchrun --nproc_per_node=4 --master_port=28670 train.py \
#     --cfg_file cfgs/ref_model_cfgs/cyclist_prm_model.yaml \
#     --extra_tag shanghai5_camera \
#     --launcher pytorch \
#     --batch_size 96 \
    # --workers 4 