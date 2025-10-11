# DetZero - Vision Detection Module

## 简介

这是DetZero框架的视觉检测模块，使用纯视觉方案（BEVFormer）进行3D目标检测，替代原有的激光雷达CenterPoint方法。

### 主要特点

- **纯视觉方案**: 使用多视角相机图像进行3D目标检测
- **BEVFormer架构**: 采用先进的BEV（鸟瞰图）表征学习方法
- **时序建模**: 支持多帧时序信息融合
- **Waymo适配**: 专门针对Waymo数据集优化

## 架构说明

本模块直接使用BEVFormer的核心组件：

```
detection_vis/
├── detzero_det_vis/           # 检测模块代码
│   ├── models/                # 模型定义
│   │   ├── bevformer.py      # BEVFormer模型封装
│   │   └── bevformer_modules/ # BEVFormer模块
│   ├── datasets/              # 数据集
│   │   ├── dataset.py        # 数据集基类
│   │   ├── waymo/            # Waymo数据集
│   │   └── builder.py        # 数据加载器构建
│   └── utils/                 # 工具函数
├── tools/                     # 训练和测试工具
│   ├── cfgs/                  # 配置文件
│   │   ├── det_dataset_cfgs/ # 数据集配置
│   │   └── det_model_cfgs/   # 模型配置
│   ├── train.py              # 训练脚本
│   ├── test.py               # 测试脚本
│   └── train_utils.py        # 训练工具
└── setup.py                   # 安装脚本
```

## 安装

### 前置依赖

1. 安装BEVFormer依赖（参考BEVFormer官方文档）
2. 确保BEVFormer在路径 `/rss/zhangruizhe/thesis_writing/BevFormer/BEVFormer`

### 安装vision检测模块

```bash
cd DetZero/detection_vis
python setup.py develop
```

## 数据准备

### Waymo数据集

本模块需要Waymo数据集的多视角图像数据：

1. 下载Waymo开放数据集
2. 提取相机图像到指定目录
3. 准备标注文件

数据目录结构：
```
data/waymo/
├── images/
│   └── {sequence_name}/
│       ├── FRONT_{frame_id}.jpg
│       ├── FRONT_LEFT_{frame_id}.jpg
│       ├── FRONT_RIGHT_{frame_id}.jpg
│       ├── SIDE_LEFT_{frame_id}.jpg
│       └── SIDE_RIGHT_{frame_id}.jpg
├── annotations/
│   └── {sequence_name}/
│       └── {frame_id}.pkl
└── ImageSets/
    ├── train.txt
    ├── val.txt
    ├── train_small.txt
    └── val_small.txt
```

## 运行

### 训练

```bash
cd DetZero/detection_vis/tools
python train.py --cfg_file cfgs/det_model_cfgs/bevformer_waymo_small.yaml
```

### 测试

```bash
cd DetZero/detection_vis/tools
python test.py --cfg_file cfgs/det_model_cfgs/bevformer_waymo_small.yaml --ckpt <PATH_TO_CKPT>
```

### 分布式训练

```bash
cd DetZero/detection_vis/tools
bash scripts/dist_train.sh <NUM_GPUS> --cfg_file cfgs/det_model_cfgs/bevformer_waymo_small.yaml
```

## 配置说明

### 数据集配置 (det_dataset_cfgs)

- `waymo_vision_base.yaml`: Waymo完整数据集配置
- `waymo_vision_small.yaml`: Waymo小数据集配置（用于快速实验）

主要参数：
- `QUEUE_LENGTH`: 时序队列长度，用于多帧融合
- `IMG_NORM_CFG`: 图像归一化配置
- `NUM_CAMERAS`: 相机数量（Waymo为5个）

### 模型配置 (det_model_cfgs)

- `bevformer_waymo_small.yaml`: BEVFormer在Waymo小数据集上的配置

主要参数：
- `IMG_BACKBONE`: 图像骨干网络（ResNet50）
- `IMG_NECK`: 特征金字塔网络（FPN）
- `PTS_BBOX_HEAD`: BEVFormer检测头
- `TRANSFORMER`: Transformer配置（编码器和解码器）

## 与原DetZero的对比

| 特性 | 原DetZero (detection) | 新模块 (detection_vis) |
|------|---------------------|----------------------|
| 输入模态 | 激光雷达点云 | 多视角相机图像 |
| 检测器 | CenterPoint | BEVFormer |
| 特征提取 | 3D体素卷积 | 图像CNN + BEV Transformer |
| 时序建模 | 点云拼接 | BEV特征融合 |

## 性能优化建议

1. **混合精度训练**: 使用FP16可以减少显存占用
2. **梯度累积**: batch size受限时使用梯度累积
3. **数据预加载**: 增加workers数量加快数据加载

## 相关工作

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer): 原始BEVFormer实现
- [DetZero](https://github.com/PJLab-ADG/DetZero): DetZero框架
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): 3D检测工具箱

## 引用

如果使用本模块，请引用：

```bibtex
@inproceedings{li2022bevformer,
  title={BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  booktitle={ECCV},
  year={2022}
}

@inproceedings{ma2023detzero,
  title={DetZero: Rethinking Offboard 3D Object Detection with Long-term Sequential Point Clouds},
  author={Ma, Tao and Yang, Xuemeng and Zhou, Hongbin and Li, Xin and Shi, Botian and Liu, Junjie and Yang, Yuchen and Liu, Zhizheng and He, Liang and Qiao, Yu and Li, Yikang and Li, Hongsheng},
  booktitle={ICCV},
  year={2023}
}
```

## 许可证

Apache License 2.0

## 联系方式

如有问题，请联系项目维护者。

