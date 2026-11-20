"""V3 RGB backbone with aligned four-scale learned auxiliary-depth fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from .rgb import ResidualUNet, block
from .scene_rgb import SceneAffineUNet, scene_statistics

ARCHITECTURE = 'rgb-scene-depth-gated-unet-v1'


class DepthLocalUNet(ResidualUNet):
    def __init__(self, base_channels=16, residual_scale=.5):
        super().__init__(base_channels, residual_scale)
        self.depth_encoders = nn.ModuleList([block(1 if i == 0 else 4*2**(i-1), 4*2**i) for i in range(4)])
        self.depth_projections = nn.ModuleList([nn.Conv2d(4*2**i, base_channels*2**i, 1) for i in range(4)])
        self.depth_gates = nn.Parameter(torch.zeros(4))

    def fuse_depth(self, scale, rgb_features, projected_depth):
        return rgb_features + torch.tanh(self.depth_gates[scale])*projected_depth

    def forward_unclamped(self, rgb, *, depth):
        scene_statistics(rgb)  # Same finite/range/input validation as RGB context.
        if (depth.shape != (rgb.shape[0], 1, *rgb.shape[-2:]) or depth.dtype != rgb.dtype
                or depth.device != rgb.device or not torch.isfinite(depth).all()
                or (depth < 0).any() or (depth > 1).any()):
            raise ValueError('Depth must be aligned finite N1HW in [0,1], matching RGB dtype/device.')
        h,w = rgb.shape[-2:]
        pad = (0, (-w)%8, 0, (-h)%8)
        padded = F.pad(rgb, pad, mode='replicate')
        x,d = padded,F.pad(depth, pad, mode='replicate')
        skips=[]
        for i,(encoder,depth_encoder,projection) in enumerate(zip(self.encoders,self.depth_encoders,self.depth_projections)):
            if i:
                x,d=F.avg_pool2d(x,2),F.avg_pool2d(d,2)
            d=depth_encoder(d)
            x=self.fuse_depth(i,encoder(x),projection(d))
            skips.append(x)
        for decoder,skip in zip(self.decoders,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=decoder(torch.cat([x,skip],dim=1))
        return (padded+self.settings['residual_scale']*torch.tanh(self.head(x)))[...,:h,:w]


class SceneDepthUNet(SceneAffineUNet):
    local_class = DepthLocalUNet
    def __init__(self, **settings):
        super().__init__(**settings)
        # Preserve exactly the seeded V3 RGB/scene parameters. Depth is additional.
        original=self.local
        self.local=self.local_class(original.settings['base_channels'],original.settings['residual_scale'])
        self.local.load_state_dict(original.state_dict(),strict=False)

    def forward_unclamped(self,rgb,*,depth,scene_stats=None):
        if scene_stats is None:scene_stats=scene_statistics(rgb)
        if (scene_stats.shape != (rgb.shape[0],6) or scene_stats.dtype != rgb.dtype
                or scene_stats.device != rgb.device or not torch.isfinite(scene_stats).all()
                or (scene_stats < 0).any() or (scene_stats > 1).any()):
            raise ValueError('Invalid scene statistics.')
        parameters=torch.tanh(self.scene(scene_stats))
        return (self.local.forward_unclamped(rgb,depth=depth)
                +self.settings['gain_scale']*parameters[:,:3,None,None]*rgb
                +self.settings['offset_scale']*parameters[:,3:,None,None])

    def forward(self,rgb,*,depth,scene_stats=None):
        return self.forward_unclamped(rgb,depth=depth,scene_stats=scene_stats).clamp(0,1)
