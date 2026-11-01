"""Shape contract of the three models, on tiny random-weight instances (no pretrained weights, no GPU).

The Stage 4 fusion model consumes the 1024-d vector that the video encoder exports, so the two
shapes below are the interface between the stages. A refactor that breaks them must fail here.
"""
import numpy as np
import pytest
import torch

from dinov3_encoder import DINOv3VideoEncoder
from inference import _softmax3
from swin1d import SwinTransformer_1D
from swin_fusion import SwinTransformer_1D as SwinFusion

# Same architecture family as production (embed_dim 256, 16 heads, depths 2-2-6-2), shrunk for speed.
SMALL = dict(seq_len=51, in_chans=3, num_classes=4, window_size=8, patch_size=1,
             num_heads=[4, 4, 4, 4], depths=[1, 1, 1, 1], embed_dim=32)


def test_1d_swin_outputs_four_logits():
    m = SwinTransformer_1D(**SMALL).eval()
    with torch.no_grad():
        assert tuple(m(torch.randn(2, 3, 51)).shape) == (2, 4)


def test_fusion_model_takes_kinematics_plus_1024_video_feature():
    m = SwinFusion(**SMALL).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 51), torch.randn(2, 1024))
    assert tuple(out.shape) == (2, 4)


def test_softmax3_ignores_the_unused_fourth_output_and_sums_to_one():
    p = _softmax3(torch.tensor([[1.0, 2.0, 3.0, 100.0]]))
    assert p.shape == (1, 3)
    assert np.isclose(p.sum(), 1.0)
    assert p.argmax() == 2


@pytest.fixture(scope='module')
def video_encoder():
    return DINOv3VideoEncoder(num_classes=4, num_frames=4, output_dim=1024,
                              freeze_backbone=True, pretrained=False).eval()


def test_video_encoder_exports_a_1024_feature_and_four_logits(video_encoder):
    x = torch.randn(1, 3, 4, 224, 224)
    with torch.no_grad():
        assert tuple(video_encoder.get_vid_feature(x).shape) == (1, 1024)
        assert tuple(video_encoder(x).shape) == (1, 4)


def test_video_temporal_head_takes_cls_tokens_of_the_backbone(video_encoder):
    """The web precomputes CLS tokens (N, frames, 384) and runs only the temporal part."""
    with torch.no_grad():
        assert tuple(video_encoder._temporal_aggregate(torch.randn(2, 4, 384)).shape) == (2, 1024)


def test_frozen_backbone_stays_frozen_and_in_eval_mode(video_encoder):
    video_encoder.train()
    assert not video_encoder.dino.training
    assert not any(p.requires_grad for p in video_encoder.dino.parameters())
    assert any(p.requires_grad for p in video_encoder.temporal_encoder.parameters())
