"""Whole-image colour conditioning plus native local residual reconstruction."""
import math

import torch
from torch import nn

from .rgb import ResidualUNet

ARCHITECTURE = 'rgb-scene-affine-unet-v1'


def scene_statistics(rgb):
    """Six input-only native-image moments, before crops or padding.

    Channel means then population standard deviations. These are invariant to
    training's right-angle rotations and flips; no target statistics are used.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point() or min(rgb.shape[-2:]) < 1:
        raise ValueError('Expected nonempty floating RGB NCHW input.')
    if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
        raise ValueError('Scene RGB must be finite and in [0, 1].')
    return torch.cat([rgb.mean((-2, -1)), rgb.std((-2, -1), correction=0)], dim=1)


class SceneAffineUNet(nn.Module):
    """Identity-initialized gain/offset MLP and independent local U-Net.

    The local branch sees the original input. The global branch contributes
    (gain-1)*input + offset; no invented detail is introduced by this branch.
    Learned local corrections still require the full native artifact screen.
    """
    scale = 8

    def __init__(self, base_channels=16, residual_scale=0.5,
                 scene_hidden=32, gain_scale=0.5, offset_scale=0.25):
        super().__init__()
        if type(scene_hidden) is not int or not 4 <= scene_hidden <= 128:
            raise ValueError('scene_hidden must be an integer in [4, 128].')
        if not all(math.isfinite(s) and 0 < s <= 1 for s in [gain_scale, offset_scale]):
            raise ValueError('Colour scales must be finite and in (0, 1].')
        self.local = ResidualUNet(base_channels, residual_scale)
        self.scene = nn.Sequential(nn.Linear(6, scene_hidden), nn.ReLU(), nn.Linear(scene_hidden, 6))
        nn.init.zeros_(self.scene[-1].weight)
        nn.init.zeros_(self.scene[-1].bias)
        self.settings = {**self.local.settings, 'scene_hidden': scene_hidden,
                         'gain_scale': gain_scale, 'offset_scale': offset_scale}

    def forward_unclamped(self, rgb, *, scene_stats=None):
        if scene_stats is None:
            scene_stats = scene_statistics(rgb)
        if (scene_stats.shape != (rgb.shape[0], 6) or not scene_stats.is_floating_point()
                or scene_stats.dtype != rgb.dtype or scene_stats.device != rgb.device
                or not torch.isfinite(scene_stats).all()
                or (scene_stats < 0).any() or (scene_stats > 1).any()):
            raise ValueError('Expected finite matching N x 6 input scene statistics in [0, 1].')
        parameters = torch.tanh(self.scene(scene_stats))
        gain_delta = self.settings['gain_scale'] * parameters[:, :3, None, None]
        offset = self.settings['offset_scale'] * parameters[:, 3:, None, None]
        return self.local.forward_unclamped(rgb) + gain_delta * rgb + offset

    def forward(self, rgb, *, scene_stats=None):
        return self.forward_unclamped(rgb, scene_stats=scene_stats).clamp(0, 1)
