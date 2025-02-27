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

### Running
- 1. 数据采集 & 上传： 由 @ 由朱嘉伟 采集，数据存放路径为：/data1/turbo_data/4D_label_dataset/origin
  
- 2. 数据解析 ：把采集好的数据进行解析，以 train_hy_4d_road_7_20250209_lx 为例子
  - 2.1 7号路口的执行如下：
    - 第一步 生成推理所需要的test.json：  执行 parse_data/parse_data_for_bevpro/create_data_for_bevpro.py （注意修改main函数中的地址） 执行完成后会生成 /data1/turbo_data/4D_label_dataset/models_res/bevpro/train_hy_4d_road_7_20250209_lx/scences/test.json 的路径
    - 第二步 生成推理所需要的 pkl： 执行 /data1/turbo_data/wangruihao/code/4D_label/parse_data/parse_data_for_bevpro/create_data_mogo_wrh.py （注意修改main函数中的地址）执行完成后会生成 /data1/turbo_data/4D_label_dataset/models_res/bevpro/train_hy_4d_road_7_20250209_lx 的路径
- 3. 执行推理：
  - 3.1 7号路口推理 下载BEVFusion的镜像，并执行 ，注意模型和config 是配置好的，只需要更改 --dataset_root 的路径即可。执行完成后会生成 /data1/turbo_data/4D_label_dataset/models_res/bevpro/train_hy_4d_road_7_20250209_lx/model_pred 的路径
    ```
    torchpack dist-run -np 1 python tools/visualize/get_infer_res.py configs/camera+lidar+fisheye/res50_DepthLSS+pillar_Intersection_WRH_0213_debug.yaml --checkpoint /data1/turbo_data/wangruihao/code/BEVFUSION/save_model/20250213/epoch_18.pth --bbox-score 0.35 --dataset_root /data1/turbo_data/4D_label_dataset/models_res/bevpro/train_hy_4d_road_7_20250209_lx/

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
