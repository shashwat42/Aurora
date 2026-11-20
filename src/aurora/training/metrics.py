"""Native RGB metric v1: Gaussian SSIM, population moments, valid windows.

11x11 window, sigma=1.5, K1=.01, K2=.03, range=1. RGB channels
and valid spatial positions receive equal weight. Images smaller than 11
pixels on either axis are rejected, consistent with the frozen data audit.
The same differentiable SSIM definition is used by reconstruction training.
"""
import torch
from torch.nn import functional as F

METRIC_VERSION = "rgb-psnr-ssim-valid11-v1"


def _check(x, y):
    if x.shape != y.shape or x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 11:
        raise ValueError("Metrics require aligned RGB NCHW images at least 11x11.")
    if not x.is_floating_point() or not y.is_floating_point():
        raise ValueError("Metrics require floating-point images in [0, 1].")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Metrics require finite images.")
    if (x < 0).any() or (x > 1).any() or (y < 0).any() or (y > 1).any():
        raise ValueError("Metric images must be in [0, 1].")


def ssim(x, y):
    _check(x, y)
    positions = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-positions.square() / (2 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    values = torch.cat([x, y, x.square(), y.square(), x * y])
    values = F.conv2d(values, kernel.reshape(1, 1, 1, 11).expand(3, 1, 1, 11), groups=3)
    values = F.conv2d(values, kernel.reshape(1, 1, 11, 1).expand(3, 1, 11, 1), groups=3)
    mx, my, xx, yy, xy = values.chunk(5)
    vx, vy = (xx - mx.square()).clamp_min(0), (yy - my.square()).clamp_min(0)
    covariance = xy - mx * my
    score = ((2 * mx * my + .01 ** 2) * (2 * covariance + .03 ** 2) /
             ((mx.square() + my.square() + .01 ** 2) * (vx + vy + .03 ** 2)))
    return score.mean(dim=(1, 2, 3))


def psnr(x, y):
    _check(x, y)
    mse = (x - y).square().mean(dim=(1, 2, 3))
    return -10 * torch.log10(mse)  # Identity is mathematically +infinity.
