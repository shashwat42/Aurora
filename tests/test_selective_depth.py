import unittest
import torch
import test_scene_depth as fixtures
from aurora.models.scene_depth import SceneDepthUNet
from aurora.models.selective_depth import SelectiveSceneDepthUNet, ARCHITECTURE


class SelectiveDepthTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_shared_initialization_identity_and_selector_range(self):
        torch.manual_seed(17);control=SceneDepthUNet(base_channels=4)
        torch.manual_seed(17);treatment=SelectiveSceneDepthUNet(base_channels=4)
        for name,value in control.state_dict().items():
            self.assertTrue(torch.equal(value,treatment.state_dict()[name]))
        for h,w in [(1,1),(19,27)]:
            x=torch.rand(1,3,h,w);d=torch.rand(1,1,h,w)
            self.assertTrue(torch.equal(treatment(x,depth=d),x))
        a,b=torch.rand(1,4,11,17),torch.rand(1,4,11,17)
        selector=treatment.local.selection_map(0,a,b)
        self.assertTrue(torch.equal(selector,torch.full_like(selector,.5)))

    def test_spatial_modulation_gradients_and_suppression(self):
        torch.manual_seed(0);model=SelectiveSceneDepthUNet(base_channels=4)
        with torch.no_grad():
            model.local.depth_gates.fill_(.4)
            model.local.head.weight.fill_(.03)
            model.local.depth_selectors[0][-1].weight.fill_(.05)
        x=torch.rand(1,3,24,32)*.4+.2;depth=torch.rand(1,1,24,32)
        a=torch.rand(1,4,8,8);b=torch.rand(1,4,8,8)
        selection=model.local.selection_map(0,a,b)
        self.assertGreater(selection.std().item(),0)
        y=model.forward_unclamped(x,depth=depth);y.mean().backward()
        self.assertGreater(model.local.depth_selectors[0][-1].weight.grad.abs().sum().item(),0)
        with torch.no_grad():
            for s in model.local.depth_selectors:s[-1].bias.fill_(-30)
            suppressed=model(x,depth=depth)
            model.local.depth_gates.zero_();off=model(x,depth=depth)
        torch.testing.assert_close(suppressed,off,atol=1e-6,rtol=0)

    def test_tiny_selective_training_reload_and_corrupt_depth_rejection(self):
        fixtures.SceneDepthTests.run_tiny_depth_fixture(self,ARCHITECTURE)
