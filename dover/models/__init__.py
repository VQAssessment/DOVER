from .swin_backbone import SwinTransformer3D as VQABackbone
from .swin_backbone import SwinTransformer2D as IQABackbone
from .head import VQAHead, IQAHead, VARHead
from .swin_backbone import swin_3d_tiny, swin_3d_small
from .conv_backbone import convnext_3d_tiny, convnext_3d_small
from .evaluator import BaseEvaluator, BaseImageEvaluator, DOVER
__all__ = [
    "VQABackbone",
    "IQABackbone",
    "VQAHead",
    "IQAHead",
    "VARHead",
    "BaseEvaluator",
    "BaseImageEvaluator",
    "DOVER",
]
