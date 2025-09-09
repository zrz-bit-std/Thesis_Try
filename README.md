# 4D 标注整体pipline


## Intro
   整个流程分为：采集 -> 上传 -> 数据解析 -> 模型推理 -> 跟踪融合 -> 拆分插值 -> 可视化 -> 人工筛选 -> 数据回流


## Structure Tree of Tracking Module
```
|-- config
|-- merge_model
|-- parse_data
|   |-- parse_data_for_bevlite
|   |-- parse_data_for_bevpro
|   |-- parse_data_for_calib
|   `-- parse_data_for_fisheye
|-- tracking
|   |-- README.md
|   |-- detzero_track
|   |-- detzero_track.egg-info
|   |-- results
|   |-- setup.py
|   |-- tools
|   `-- utils_track
|-- utils
|   |-- build
|   |-- detzero_utils
|   |-- detzero_utils.egg-info
|   `-- setup.py
`-- visual_data
    |-- data_show.py
    `-- utils

``` 

### 环境依赖
4D整体pipeline，依赖两个代码仓库：BevCalib(标定参数)、BEVFUSION(检测模型)
- 1. 在4D_label工作目录下，git clone git@gitlab.zhidaoauto.com:dingchunqiu/BevCalib.git; 然后进入BevCalib目录，切换到特定分支，git checkout percept_transform
- 2. 在4D_label工作目录下，git clone git@gitlab.zhidaoauto.com:rss/perception_engine/BEVFUSION.git; 然后进入BEVFUSION目录，切换特定分支，git checkout master_wangruihao_4d_data, 按该仓库的readme进行编译安装；


### Running
- 1. 数据采集 & 上传： 由 @ 由朱嘉伟 采集，数据存放路径为：/data1/turbo_data/4D_label_dataset/origin
  
- 2. 数据解析 ：把采集好的数据进行解析，以 train_hy_4d_road_7_20250209_lx 为例子
  - 2.1 7号路口的执行如下：
    - 第一步 生成推理所需要的test.json：  执行 parse_data/parse_data_for_bevpro/create_data_for_bevpro.py （注意修改main函数中的地址） 执行完成后会生成 /data1/turbo_data/4D_label_dataset/models_res/bevpro/train_hy_4d_road_7_20250209_lx/scences/test.json 的路径
    - 第二步 生成推理所需要的 pkl： 执行 /data1/turbo_data/wangruihao/code/4D_label/parse_data/parse_data_for_bevpro/create_data_mogo_wrh.py （注意修改main函数中的地址）执行完成后会生成 /data1/turbo_data/4D_label_dataset/models_res/bevpro/train_hy_4d_road_7_20250209_lx 的路径
- 3. 执行推理：
  - 3.1 7号路口推理 下载BEVFusion的镜像，并执行 ，注意模型和config 是配置好的，只需要更改 --dataset_root 的路径即可。执行完成后会生成 /data1/turbo_data/4D_label_dataset/models_res/bevpro/train_hy_4d_road_7_20250209_lx/model_pred 的路径
    ```
    torchpack dist-run -np 1 python tools/visualize/get_infer_res.py configs/camera+lidar+fisheye/res50_DepthLSS+pillar_Intersection_WRH_0213_debug.yaml --checkpoint /data1/turbo_data/zhangchengyue/code/BEV/work/BEVFUSION/runs/ft1200_0213/epoch_20.pth --bbox-score 0.35 --dataset_root /data1/turbo_data/4D_label_dataset/models_res/bevpro/train_hy_4d_road_7_20250209_lx/

    ```
  - 3.2 推理结果进行拆分，并合成跟踪所需要的pkl ：执行 parse_data/parse_data_for_bevpro/make_data_for_track.py （注意修改main函数中的地址）拆分完成后会生成 /data1/turbo_data/4D_label_dataset/models_res/offline_tracked/train_hy_4d_road_7_20250209_lx 的地址
- 4. 执行4D融合跟踪：
  - 4.1 安装detzero所需要的detzero_util: 
   ```
   cd  /data1/turbo_data/wangruihao/code/4D_label/tracking
   pip install -e .
   ```
  - 4.2 执行跟踪： 执行 tracking/tools/run_track_20250224.py （注意修改main函数中的地址）
  - 4.3 跟踪结果拆分：执行 tracking/utils_track/tracking_data_split.py  （注意修改main函数中的地址）
- 5. 生成可视化地址：
  - 根据跟踪结果和原始数据地址，生成可视化demo： 执行 visual_data/data_show.py （（注意修改main函数中的地址） 执行完成后就会生成 /data1/turbo_data/4D_label_dataset/labels/train_hy_4d_road_7_20250209_lx/selected 的地址，
- 6. 结果进行筛选 


### 数据批量生产
- 3D单帧数据，针对单帧采集数据(因采集数据量的问题，暂时也走单帧生产路线，最终需要切换到时序生产链路)
  ```
  bash scripts/batch_dataset_4D_label_one_frame.sh
  ```

- 4D时序数据，使用时序采集数据
  ```
  bash scripts/batch_dataset_4D_label.sh
  ```

### 准备半自动清洗数据到标注平台
- 4D清洗数据返回后，挑选出大车(car/truck/bus)通过数据，送标人工标注进行小目标（rider/pedestrian/bicycle/motorcycle)标注, 
- **注意**：运行脚本时，需要修改相应的数据、保存数据路径，避免数据覆盖！！！
  ```
  bash scripts/prepare_semilabel_to_anno_platform.sh
  ```

### 准备bevlite数据格式
- 半自动标注数据转bevlite数据格式：[待半自动标注完成后，参见BEVFUSION pkl数据生产线，先从cos桶下载数据，然后生成pkl数据]，然后使用如下脚本，例如：
  - **注意**：运行脚本时，需要修改相应的数据、保存数据路径，避免数据覆盖！！！
  ```
  # 下载cos桶数据至本地目录，例如：将上海6号路口单帧半自动标注数据下载至，/data1/turbo_data/RALG/data/3.0_pro/4d/semi_autolabel/Intersection/label/shanghai_road_6/p2_one_frame
  
  bash scripts/prepare_data_for_bevlite_from_semiautolabel.sh
  ```
- 全自动标注数据转bevlite数据格式：待清洗数据返回后（主要返回清理数据对应的json文件），json文件放入送筛可视化数据的同级目录（如下所示），例如： /data1/turbo_data/4D_label_dataset_one_frame/Intersection/labels/train_sh_3d_road_6_20250725_5_one_frame/raw_anno_info/，

  |-- raw_anno_info

  |-- selected

  |-- useful.txt

  |-- useful_vehicle.txt
    - **注意**：运行脚本时，需要修改相应的数据、保存数据路径，避免数据覆盖！！！

  ```
  bash scripts/prepare_data_for_bevlite.sh
  ```

tips:如果报错某个模块未找到，则加上
set -e
export PYTHONPATH=$(pwd):$PYTHONPATH 
会解决BEVFUSION模块找不到的错误。
首先需要将工程分支切换到master_gen_4D_label分支。
1.执行 create_data_for_bevpro_v2.py脚本，生成的samples文件夹和test.json文件没有camera_0_0和camera_3_0是因为/data1/turbo_data/lishuaiyin/4D_label/BEVFUSION/tools/data_process/sensor_modules.py脚本文件中"sh_5": {
        "intersection": [
            "camera_0_1",  # S0 camera顺向/逆向槽位命名反了
            "camera_1_0",  
            "camera_2_0",
            "camera_3_1",  # S3 camera顺向/逆向槽位命名反了
            "camera_0_8",
            "camera_1_8",
            "camera_2_8",
            "camera_3_8",
            "lidar_0_12",
            "lidar_1_12",
            "lidar_2_12",
            "lidar_3_12",
        ]
    },确定的。

2.执行/data1/turbo_data/lishuaiyin/4D_label/parse_data/parse_data_for_bevpro/create_data_mogo_wrh.py生成[text](dataset_track/train_sh_3d_road_5_20250524_5000_lx/mogo_infos_test.pkl)。

3.执行
torchpack dist-run -np 4 python ./BEVFUSION/tools/visualize/get_infer_res.py ${bev_config} \
  --checkpoint ${ckpt_path} \
  --bbox-score 0.3 \
  --dataset_root ${dst_dir}/${data_name}
其中torchpack dist-run -np 4，默认的只有4张卡，因此这儿需要改成4，最后执行生成的结果会保存在/data1/turbo_data/lishuaiyin/4D_label/dataset_track/train_sh_3d_road_5_20250524_5000_lx/model_pred，这是最后生成的检测结果。
4.执行
python ./parse_data/parse_data_for_bevpro/make_data_for_track.py \
    --save_dir ${dst_dir}/merged \
    --origin_dir ${src_dir} \
    --det_dir ${dst_dir} \
    --track_dir ${dst_dir}/offline_tracked \
    --dataset_name ${data_name} \ 
会生成merged文件夹和offline_tracked文件夹，merged文件夹是经过筛选和整理的模型推理结果集合，只包含完整且有效的推理数据片段。offline_tracked文件夹是为离线跟踪算法准备的输入数据目录，包含转换为跟踪算法所需格式的数据track.pkl。

5.执行4D融合跟踪：
python ./tracking/tools/run_track_20250224.py \
  --track_dir ${dst_dir}/offline_tracked \
  --dataset_name ${data_name} \
  --cfg_file ${track_cfg} \

会生成/data1/turbo_data/lishuaiyin/4D_label/dataset_track/offline_tracked/train_sh_3d_road_5_20250524_5000_lx/20250524070526/tracking路径下的两个pkl文件。

6.执行
python ./tracking/utils_track/tracking_data_split.py \
  --track_dir ${dst_dir}/offline_tracked \
  --merged_dir ${dst_dir}/merged \
  --dataset_name ${data_name} \

tracking_data_split.py文件的主要作用是转化为txt形式的数据。最后的生成路径在/data1/turbo_data/lishuaiyin/4D_label/dataset_track/offline_tracked/train_sh_3d_road_5_20250524_5000_lx/20250524070526/splited下。

7.执行
python ./visual_data/data_show.py \
  --origin_dir ${src_dir} \
  --label_dir ${dst_dir}/labels \
  --merged_dir ${dst_dir}/merged \
  --bev_pro_dir ${dst_dir} \
  --tracked_dir ${dst_dir}/offline_tracked  \
  --dataset_name ${data_name} \

最后会将跟踪结果可视化展示图片中。/data1/turbo_data/lishuaiyin/4D_label/dataset_track/labels/train_sh_3d_road_5_20250524_5000_lx/selected

8.生成视频展示运行
python ./visual_data/demo_track_lsy.py \
  --sequence ${data_name} \

可以将tracking的结果在点云bev视角下进行视频展示。
tips其中的报错：
ffmpeg version 4.3 Copyright (c) 2000-2020 the FFmpeg developers
  built with gcc 7.3.0 (crosstool-NG 1.23.0.449-a04d0)
  configuration: --prefix=/opt/conda --cc=/opt/conda/conda-bld/ffmpeg_1597178665428/_build_env/bin/x86_64-conda_cos6-linux-gnu-cc --disable-doc --disable-openssl --enable-avresample --enable-gnutls --enable-hardcoded-tables --enable-libfreetype --enable-libopenh264 --enable-pic --enable-pthreads --enable-shared --disable-static --enable-version3 --enable-zlib --enable-libmp3lame
  libavutil      56. 51.100 / 56. 51.100
  libavcodec     58. 91.100 / 58. 91.100
  libavformat    58. 45.100 / 58. 45.100
  libavdevice    58. 10.100 / 58. 10.100
  libavfilter     7. 85.100 /  7. 85.100
  libavresample   4.  0.  0 /  4.  0.  0
  libswscale      5.  7.100 /  5.  7.100
  libswresample   3.  7.100 /  3.  7.100
Unrecognized option 'preset'.
Error splitting the argument list: Option not found
安装ffmpeg的库版本。
sudo apt-get update
sudo apt-get install ffmpeg libx264-dev
 cmd = [
           "/usr/bin/ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", "filelist.txt", 
            # "-vf", "setpts=2.0*PTS",  ##
            "-r", str(fps),
            "-c:v", "libx264",          # 
            "-preset", "slow",          # 
            "-crf", "23",               # 
            "-pix_fmt", "yuv420p",      # 
            "-movflags", "+faststart",  # 
            "-vf", "scale=iw:ih",       # 
            "-y",
            output_video                # 
        ]
验证ffmpeg是否支持libx264:
/usr/bin/ffmpeg -encoders | grep libx264   若输出包含 libx264，则说明支持 -preset 参数，修改后即可正常运行。
9.生成map指标：
python metric_eval_track_lisy.py ,需要事前用create_json_map()函数生成json文件。