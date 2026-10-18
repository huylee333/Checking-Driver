import os
import sys

# Mở rộng search path của package 'models' để kế thừa toàn bộ models gốc từ parent project:
# - SwinTransformer_cls
# - SwinTransformer_fusion
# - video_swin_transformer
_PARENT_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
if os.path.exists(_PARENT_MODELS_DIR) and _PARENT_MODELS_DIR not in __path__:
    __path__.append(_PARENT_MODELS_DIR)

from .dinov3_video_encoder import DINOv3VideoEncoder

__all__ = ['DINOv3VideoEncoder']
