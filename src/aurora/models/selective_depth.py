"""RGB/depth-conditioned spatial and channel modulation of depth fusion."""
import torch
from torch import nn
from .scene_depth import DepthLocalUNet, SceneDepthUNet

ARCHITECTURE='rgb-scene-depth-selective-unet-v1'


class SelectiveDepthLocal(DepthLocalUNet):
    def __init__(self,base_channels=16,residual_scale=.5):
        super().__init__(base_channels,residual_scale)
        self.depth_selectors=nn.ModuleList()
        for i in range(4):
            width=base_channels*2**i
            selector=nn.Sequential(nn.Conv2d(2*width,8,1),nn.ReLU(),nn.Conv2d(8,width,1))
            nn.init.zeros_(selector[-1].weight);nn.init.zeros_(selector[-1].bias)
            self.depth_selectors.append(selector)

    def selection_map(self,scale,rgb_features,projected_depth):
        return torch.sigmoid(self.depth_selectors[scale](torch.cat([rgb_features,projected_depth],dim=1)))

    def fuse_depth(self,scale,rgb_features,projected_depth):
        selection=self.selection_map(scale,rgb_features,projected_depth)
        return rgb_features+torch.tanh(self.depth_gates[scale])*selection*projected_depth


class SelectiveSceneDepthUNet(SceneDepthUNet):
    local_class=SelectiveDepthLocal
