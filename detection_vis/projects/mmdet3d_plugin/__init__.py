from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
# 避免在全局导入时牵连 datasets 与 dd3d，可在使用处按需导入pipelines
# from .datasets.pipelines import (
#   PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
#   NormalizeMultiviewImage,  CustomCollect3D)
from .models.utils import *
from .models.opt.adamw import AdamW2
from .bevformer import *

# 避免在导入时牵连 datasets/dd3d 等可选依赖
# 如需使用 dd3d 或 datasets，请在具体用到的脚本内按需导入
