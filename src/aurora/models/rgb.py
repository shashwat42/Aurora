"""Compact four-scale RGB U-Net with bounded residual prediction.

No spatial normalization: crop statistics must not change full-scene colors.
The four encoder scales require a multiple of eight; inference pads and unpads.
"""
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "rgb-residual-unet-v1"


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class ResidualUNet(nn.Module):
    scale = 8

    def __init__(self, base_channels=16, residual_scale=0.5):
        super().__init__()
        if type(base_channels) is not int or not 4 <= base_channels <= 64:
            raise ValueError("base_channels must be an integer between 4 and 64.")
        if not 0 < residual_scale <= 1:
            raise ValueError("residual_scale must be in (0, 1].")
        self.settings = {"base_channels": base_channels, "residual_scale": residual_scale}
        widths = [base_channels * 2 ** i for i in range(4)]
        self.encoders = nn.ModuleList([block(3, widths[0])] +
                                     [block(a, b) for a, b in zip(widths, widths[1:])])
        self.decoders = nn.ModuleList([block(widths[i + 1] + widths[i], widths[i]) for i in [2, 1, 0]])
        self.head = nn.Conv2d(widths[0], 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_unclamped(self, rgb):
        """Training access to residual RGB before the final range projection.

        Parameter layout and native-size decoding are identical to version 1.
        Inference must call forward(), which always returns bounded RGB.
        """
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("Expected floating RGB NCHW input.")
        if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
            raise ValueError("RGB input must be finite and in [0, 1].")
        h, w = rgb.shape[-2:]
        if min(h, w) < 1:
            raise ValueError("RGB dimensions must be positive.")
        # Replication also supports one-pixel inputs; no resize or scene cropping.
        padded = F.pad(rgb, (0, (-w) % self.scale, 0, (-h) % self.scale), mode="replicate")
        x, skips = padded, []
        for i, encoder in enumerate(self.encoders):
            if i:
                x = F.avg_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        result = padded + self.settings["residual_scale"] * torch.tanh(self.head(x))
        return result[..., :h, :w]

    def forward(self, rgb):
        return self.forward_unclamped(rgb).clamp(0, 1)
