"""Versioned generator architectures."""
from .rgb import ResidualUNet
from .rgb import ARCHITECTURE as LOCAL_ARCHITECTURE
from .scene_rgb import SceneAffineUNet, ARCHITECTURE as SCENE_ARCHITECTURE
from .scene_depth import SceneDepthUNet, ARCHITECTURE as DEPTH_ARCHITECTURE
from .selective_depth import SelectiveSceneDepthUNet, ARCHITECTURE as SELECTIVE_DEPTH_ARCHITECTURE

ARCHITECTURES = {LOCAL_ARCHITECTURE: ResidualUNet, SCENE_ARCHITECTURE: SceneAffineUNet}
ARCHITECTURES[DEPTH_ARCHITECTURE] = SceneDepthUNet
ARCHITECTURES[SELECTIVE_DEPTH_ARCHITECTURE] = SelectiveSceneDepthUNet
DEPTH_ARCHITECTURES = {DEPTH_ARCHITECTURE, SELECTIVE_DEPTH_ARCHITECTURE}


def build_rgb(architecture, settings):
    if architecture not in ARCHITECTURES:
        raise ValueError('Unsupported RGB architecture.')
    return ARCHITECTURES[architecture](**settings)


__all__ = ['ResidualUNet', 'SceneAffineUNet', 'build_rgb']
