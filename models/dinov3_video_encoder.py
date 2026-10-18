"""
DINOv3 Video Encoder - thay thế Video Swin Transformer 3D (models/video_swin_transformer.py)

Mặc định dùng bản DINOv3 NHỎ NHẤT: ViT-S/16 (timm: 'vit_small_patch16_dinov3.lvd1689m',
~21.6M params, hidden_size = 384).

Interface giữ nguyên tương thích với SwinTransformer3D gốc:
  - forward(x)          -> logits          [B, num_classes]
  - get_vid_feature(x)  -> feature vector  [B, 1024]
  - Input x shape (B, C, D, H, W) = (B, 3, 77, 224, 224)

Khác biệt so với Video Swin 3D gốc (chỉ ở phần encoder, pipeline giữ nguyên):
  1. Spatial: 3D W-MSA/SW-MSA  ->  DINOv3 ViT-S/16 chạy trên từng frame (frozen), lấy CLS token
  2. Temporal: 3D window attention + patch merging  ->  Temporal TransformerEncoder + CLS token
  3. Projection: Linear(384 -> 1024) để khớp chiều video feature 1024 của fusion head gốc
     (models/SwinTransformer_fusion.py: Linear(num_features + 1024, num_classes))
"""

import os
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


DEFAULT_DINOV3_MODEL = 'vit_small_patch16_dinov3.lvd1689m'

# Ánh xạ tên timm -> Hugging Face model id (dùng cho nhánh fallback transformers).
# Lưu ý: repo facebook/dinov3-* trên Hugging Face là gated, cần đăng nhập + chấp nhận license.
TIMM_TO_HF = {
    'vit_small_patch16_dinov3': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'vit_small_plus_patch16_dinov3': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
    'vit_base_patch16_dinov3': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'vit_large_patch16_dinov3': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
}


def _hf_model_id(model_name):
    base = model_name.split('.')[0]
    return TIMM_TO_HF.get(base, model_name)


class DINOv3VideoEncoder(nn.Module):
    """DINOv3-based Video Encoder thay thế Video Swin Transformer 3D.

    Kiến trúc:
        Video (B, C, D, H, W)
          -> DINOv3 ViT-S/16 CLS token mỗi frame   (B, D, 384)
          -> + temporal pos-embed, prepend temporal CLS
          -> Temporal TransformerEncoder            (B, 1+D, 384)
          -> temporal CLS -> LayerNorm -> Linear    (B, 1024)        = get_vid_feature
          -> head                                   (B, num_classes) = forward

    Args:
        num_classes (int): số class. Default 4 (giống project gốc).
        num_frames (int): số frame video. Default 77.
        model_name (str): tên DINOv3 trong timm (hoặc HF id). Default: ViT-S/16 (nhỏ nhất).
        output_dim (int): chiều video feature. Default 1024 (khớp fusion head gốc).
        temporal_layers (int): số tầng temporal encoder. Default 2.
        temporal_heads (int|None): số head; None -> hidden_size // 64 (384 -> 6).
        temporal_dropout (float): dropout temporal encoder. Default 0.1.
        chunk_size (int): số frame đưa qua DINOv3 mỗi lần (tránh OOM). Default 16.
        freeze_backbone (bool): đóng băng DINOv3. Default True.
        pretrained (bool): tải trọng số pretrained DINOv3. Default True.
        weights_path (str|None): file trọng số local (vd. model.safetensors tải từ
            timm/vit_small_patch16_dinov3.lvd1689m). Nếu có và pretrained=True thì đọc
            file này thay vì tải qua mạng. Chỉ áp dụng cho backend timm.
    """

    def __init__(self,
                 num_classes=4,
                 num_frames=77,
                 model_name=DEFAULT_DINOV3_MODEL,
                 output_dim=1024,
                 temporal_layers=2,
                 temporal_heads=None,
                 temporal_dropout=0.1,
                 chunk_size=16,
                 freeze_backbone=True,
                 pretrained=True,
                 weights_path=None):
        super().__init__()

        self.num_frames = num_frames
        self.chunk_size = chunk_size
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        self.backend = None

        # ----------------------------------------------------------------
        # 1. DINOv3 backbone (thay patch embed + 4 stage của Swin 3D)
        #    Ưu tiên timm, fallback sang Hugging Face transformers.
        # ----------------------------------------------------------------
        if weights_path is not None and pretrained and not os.path.isfile(weights_path):
            raise FileNotFoundError(f'Không tìm thấy file trọng số DINOv3: {weights_path}')

        try:
            import timm
            extra = {}
            if weights_path is not None and pretrained:
                # Đọc trọng số từ file local thay vì tải từ Hugging Face Hub
                extra['pretrained_cfg_overlay'] = dict(file=os.path.abspath(weights_path))
            try:
                # global_pool='token' -> output chính là CLS token (timm mặc định 'avg' cho DINOv3)
                self.dino = timm.create_model(model_name, pretrained=pretrained,
                                              num_classes=0, global_pool='token', **extra)
            except TypeError:
                self.dino = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **extra)
            self.hidden_size = self.dino.num_features
            self.backend = 'timm'
        except Exception as e_timm:
            if weights_path is not None and pretrained:
                # Không fallback sang HF: file local là trọng số định dạng timm
                raise RuntimeError(f"Không load được '{weights_path}' vào timm model '{model_name}': {e_timm}")
            hf_id = _hf_model_id(model_name)
            try:
                from transformers import AutoModel, AutoConfig
                if pretrained:
                    self.dino = AutoModel.from_pretrained(hf_id)
                else:
                    self.dino = AutoModel.from_config(AutoConfig.from_pretrained(hf_id))
                self.hidden_size = self.dino.config.hidden_size
                self.backend = 'transformers'
            except Exception as e_hf:
                raise RuntimeError(
                    f"Không thể khởi tạo DINOv3 '{model_name}' (HF id: '{hf_id}'). "
                    f"Lỗi timm: {e_timm}. Lỗi transformers: {e_hf}"
                )

        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()

        # ----------------------------------------------------------------
        # 2. Temporal Transformer Encoder (thay vai trò temporal của 3D window attention)
        # ----------------------------------------------------------------
        if temporal_heads is None:
            temporal_heads = max(1, self.hidden_size // 64)  # 384 -> 6, 768 -> 12
        if self.hidden_size % temporal_heads != 0:
            raise ValueError(f'hidden_size={self.hidden_size} không chia hết cho temporal_heads={temporal_heads}')

        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, self.hidden_size))
        trunc_normal_(self.temporal_pos_embed, std=0.02)

        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        trunc_normal_(self.temporal_cls, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=temporal_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=temporal_dropout,
            activation='gelu',
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.temporal_norm = nn.LayerNorm(self.hidden_size)

        # ----------------------------------------------------------------
        # 3. Projection hidden_size -> output_dim (384 -> 1024)
        # ----------------------------------------------------------------
        self.proj = nn.Linear(self.hidden_size, output_dim)

        # ----------------------------------------------------------------
        # 4. Classification head (giống self.head của SwinTransformer3D)
        # ----------------------------------------------------------------
        self.head = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_new_weights()

    def _init_new_weights(self):
        """Khởi tạo weights cho temporal encoder, projection, head (không đụng DINOv3)."""
        for module in [self.temporal_encoder, self.temporal_norm, self.proj, self.head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def _backbone_cls(self, chunk):
        """CLS token của DINOv3 cho một chunk frame: (n, C, H, W) -> (n, hidden)."""
        if self.backend == 'timm':
            return self.dino(chunk)
        outputs = self.dino(pixel_values=chunk)
        return outputs.last_hidden_state[:, 0]

    def _extract_frame_features(self, video):
        """(B, C, D, H, W) -> (B, D, hidden): CLS token từng frame, xử lý theo chunk."""
        B, C, D, H, W = video.shape
        frames = video.permute(0, 2, 1, 3, 4).contiguous().reshape(B * D, C, H, W)

        all_features = []
        for i in range(0, B * D, self.chunk_size):
            chunk = frames[i:i + self.chunk_size]
            if self.freeze_backbone:
                with torch.no_grad():
                    cls_tokens = self._backbone_cls(chunk)
            else:
                cls_tokens = self._backbone_cls(chunk)
            all_features.append(cls_tokens)

        frame_features = torch.cat(all_features, dim=0)
        return frame_features.view(B, D, self.hidden_size)

    def _temporal_aggregate(self, frame_features):
        """(B, D, hidden) -> (B, output_dim)."""
        B, D, _ = frame_features.shape
        x = frame_features

        if D <= self.num_frames:
            x = x + self.temporal_pos_embed[:, :D]
        else:
            pos_emb = nn.functional.interpolate(
                self.temporal_pos_embed.permute(0, 2, 1), size=D, mode='linear'
            ).permute(0, 2, 1)
            x = x + pos_emb

        cls = self.temporal_cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)            # (B, 1+D, hidden)
        x = self.temporal_encoder(x)

        video_feat = self.temporal_norm(x[:, 0])  # thay AdaptiveAvgPool3d + flatten
        return self.proj(video_feat)              # (B, 1024)

    def get_vid_feature(self, x):
        """Giống SwinTransformer3D.get_vid_feature(): (B, C, D, H, W) -> (B, 1024)."""
        return self._temporal_aggregate(self._extract_frame_features(x))

    def forward(self, x):
        """Giống SwinTransformer3D.forward(): (B, C, D, H, W) -> (B, num_classes)."""
        return self.head(self.get_vid_feature(x))

    def train(self, mode=True):
        """Giữ DINOv3 ở eval mode khi bị frozen."""
        super().train(mode)
        if self.freeze_backbone and hasattr(self, 'dino'):
            self.dino.eval()
        return self
