import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from aurora.models.scene_rgb import SceneAffineUNet
from aurora.models.scene_depth import SceneDepthUNet
from aurora.training.run import train
from aurora.inference import RGBInference
from aurora.image import validate_image
from aurora.manifest import file_hash

ROOT=Path(__file__).resolve().parents[1]


class SceneDepthTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_rgb_initialization_parity_and_real_depth_dependence(self):
        torch.manual_seed(20261101);rgb=SceneAffineUNet(base_channels=4)
        torch.manual_seed(20261101);model=SceneDepthUNet(base_channels=4)
        for name,value in rgb.state_dict().items():
            self.assertTrue(torch.equal(value,model.state_dict()[name]))
        x=torch.rand(1,3,19,27);d=torch.rand(1,1,19,27)
        self.assertTrue(torch.equal(model(x,depth=d),rgb(x)))
        self.assertTrue(torch.equal(model(x,depth=torch.zeros_like(d)),x))
        with self.assertRaises(TypeError):model(x)
        with self.assertRaises(ValueError):model(x,depth=d[...,:18,:])
        with self.assertRaises(ValueError):model(x,depth=d*float('nan'))
        with torch.no_grad():
            model.local.head.weight.fill_(.03)
            model.local.depth_gates.fill_(.4)
        a=model.forward_unclamped(x,depth=d)
        b=model.forward_unclamped(x,depth=torch.zeros_like(d))
        self.assertGreater((a-b).abs().max().item(),0)
        a.mean().backward()
        self.assertGreater(model.local.depth_gates.grad.abs().sum().item(),0)
        self.assertGreater(model.local.depth_encoders[0][0].weight.grad.abs().sum().item(),0)

    def test_tiny_depth_training_reload_and_inference_requires_depth(self):
        self.run_tiny_depth_fixture('rgb-scene-depth-gated-unet-v1')

    def run_tiny_depth_fixture(self, architecture):
        (ROOT/'artifacts/tests').mkdir(exist_ok=True,parents=True)
        with TemporaryDirectory(dir=ROOT/'artifacts/tests') as temp:
            root=Path(temp);rows=[]
            for i,split in enumerate(['train','diagnostic']):
                row={'id':str(i),'scene_id':str(i),'split':split}
                for key,color in [('raw',(70,90,120)),('reference',(80,100,130))]:
                    path=root/f'{i}-{key}.png';Image.new('RGB',(23,19),color).save(path)
                    row[key+'_path']=path.name;row[key+'_sha256']=file_hash(path)
                rows.append(row)
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'status':'frozen','pairs':rows,'challenging':[]}))
            manifest.with_suffix('.json.sha256').write_text(file_hash(manifest))
            depth_config=root/'depth.json';depth_config.write_text('{}')
            class FixtureDepth:
                fingerprint='fixture'
                def __init__(self,*args):pass
                def load_cached(self,valid):return np.linspace(0,1,valid.image.width*valid.image.height,dtype=np.float32).reshape(valid.image.height,valid.image.width)
                def _paths(self,valid):return (depth_config,depth_config)
            config=json.loads((ROOT/'configs/depth_baseline.json').read_text())|{
                'architecture':architecture,
                'manifest':str(manifest),'manifest_sha256':file_hash(manifest),'data_root':str(root),
                'depth_config':str(depth_config),'depth_config_sha256':file_hash(depth_config),'depth_fingerprint':'fixture',
                'model':{'base_channels':4,'residual_scale':.5},'epochs':2,'candidate_epochs':[1,2],
                'crop_size':16,'batch_size':1,'threads':2,'output':str(root/'run')}
            path=root/'config.json';path.write_text(json.dumps(config))
            # Saved model settings include all constructor defaults, as real configs do.
            config['model'].update(scene_hidden=32,gain_scale=.5,offset_scale=.25)
            path.write_text(json.dumps(config))
            with patch('aurora.depth.DepthEstimator',FixtureDepth),contextlib.redirect_stdout(io.StringIO()):
                report=train(path)
                self.assertTrue(report['checkpoint_reload_exact'])
                service=RGBInference(root/'run/last/bundle.json')
                result=service.predict(validate_image(root/'1-raw.png'))
                self.assertEqual(tuple(result.shape),(1,3,19,23))
                with patch.object(service.depth,'load_cached',side_effect=ValueError('corrupt depth')):
                    with self.assertRaisesRegex(ValueError,'corrupt depth'):
                        service.predict(validate_image(root/'1-raw.png'))
                service.depth_policy='live'
                phases=[]
                with patch.object(service.depth,'predict',side_effect=ValueError('live depth failed'),create=True):
                    with self.assertRaisesRegex(ValueError,'live depth failed'):
                        service.predict(validate_image(root/'1-raw.png'),stage_callback=phases.append)
                self.assertEqual(phases,['depth'])
