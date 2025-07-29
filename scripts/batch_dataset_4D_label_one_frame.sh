#!/bin/bash

echo "*************** 4D Label begin time: $(date '+%Y-%m-%d %H:%M:%S')"

# 注意：修改相应的数据，以及对应的模型配置文件及权重文件！！！！！
# 注意：注意生产时的路径，不要覆盖原来生产的数据！！！！！！


#========================= 调试数据路口数据集 ============================================
# 上海5号路口-单帧数据生产调试：
src_dir=/data1/turbo_data/huben/data/test_4D_label_one_frame
dataset_names=(
    "train_sh_3d_road_5_20250713_5_one_frame_tmp"
)
locations=(
    "sh"
)
roads=(
    "5"
)
calib_versions=(
    "v20250605"  # sh5
)
data_type=Intersection  # [Intersection, Roadsection, IntersectionRoadsection]
mode=intersection


#========================== 正式数据生产 ================================================    
# # 路口单帧生产:
# src_dir=/data1/turbo_data/4D_label_dataset_one_frame
# dataset_names=(
#     "train_sh_3d_road_2_20250718_5_one_frame"
#     "train_sh_3d_road_2_20250719_5_one_frame"
#     "train_sh_3d_road_4_20250718_5_one_frame"
#     "train_sh_3d_road_5_20250719_5_one_frame"
# )
# locations=(
#     "sh"
#     "sh"
#     "sh"
#     "sh"
# )
# roads=(
#     "2"
#     "2"
#     "4"
#     "5"
# )
# calib_versions=(
#     "v20250605"  # sh2
#     "v20250605"  # sh2
#     "v20250109"  # sh4
#     "v20250605"  # sh5
# )
# data_type=Intersection  # [Intersection, Roadsection, IntersectionRoadsection]
# mode=intersection


# 公共参数部分！！
calib=/data1/turbo_data/huben/projects/BevCalib
bbox_score=0.05
intersection_pole="None" # 对于路口为"None"
if [[ "$mode" == "intersection" ]]; then
    config_file=/data1/turbo_data/RALG/models/version/4D_prelabel/Intersection/v0613/raw_bevfusion_plus_fisheye_4D_prelabel_v2_200x200m.yaml
    checkpoint=/data1/turbo_data/RALG/models/version/4D_prelabel/Intersection/v0613/epoch_11.pth
else
    config_file=/data1/turbo_data/RALG/models/version/4D_prelabel/Roadsection/v0519/bevfusion_roadsection_4d_prelabel.yaml
    checkpoint=/data1/turbo_data/RALG/models/version/4D_prelabel/Roadsection/v0519/epoch_15.pth
fi

echo "bbox_score: $bbox_score"

# check list with identical length
params_length=${#dataset_names[@]}
if [[ ${#locations[@]} -ne $params_length || ${#roads[@]} -ne $params_length || ${#calib_versions[@]} -ne $params_length ]]; then
    echo "Error: All param list must have identical length!" >&2
    exit 1
fi
num_poles=${#poles[@]}

echo "num_poles: $num_poles"

# 启动所有后台任务并记录PID
completed_tasks=0
total_tasks=${#dataset_names[@]}
pids=()

# processing
for ((i=0; i<params_length; i++)); do
    if [[ "$mode" == "intersection" ]]; then
        log_file="tmp_output/logs/${dataset_names[$i]}.log"
        echo "Processing $((i+1)) dataset, log_file: $log_file"
        bash scripts/one_dataset_4D_label_one_frame.sh "${src_dir}" "${dataset_names[$i]}" "${locations[$i]}" "${roads[$i]}" "${calib}" "${calib_versions[$i]}" "${data_type}" "${config_file}" "${checkpoint}" "${bbox_score}" "${mode}" "${intersection_pole}"
        pids+=($!)  # 保存后台进程PID
    else
        for ((j=0; j<num_poles; j++)); do
            log_file="tmp_output/logs/${dataset_names[$i]}_${poles[$j]}.log"
            echo "Processing $((i+1)) dataset, log_file: $log_file"
            bash scripts/one_dataset_4D_label_one_frame.sh "${src_dir}" "${dataset_names[$i]}" "${locations[$i]}" "${roads[$i]}" "${calib}" "${calib_versions[$i]}" "${data_type}" "${config_file}" "${checkpoint}" "${bbox_score}" "${mode}" "${poles[$j]}"
            pids+=($!)  # 保存后台进程PID
        done
    fi
done

# 等待每个任务完成并更新进度
for pid in "${pids[@]}"; do
    wait $pid
    ((completed_tasks++))
    progress=$((completed_tasks * 100 / total_tasks))
    echo -ne "总进度：${progress}% (${completed_tasks}/${total_tasks})\r"
done

echo -e "\n Finish all datasets!"
echo "*************** ${mode} 4D Label finished time: $(date '+%Y-%m-%d %H:%M:%S')"

