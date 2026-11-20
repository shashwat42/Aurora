"""Small preregistered reconstruction-only scene-context probe."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aurora.image import validate_image
from aurora.inference import RGBInference, export_png
from aurora.manifest import read_manifest,file_hash,safe_path
from aurora.training.metrics import psnr,ssim


def tensor(image):return torch.from_numpy(np.asarray(image,dtype=np.float32).transpose(2,0,1).copy()/255).unsqueeze(0)
def pixels(t):return Image.fromarray(np.rint(t[0].permute(1,2,0).numpy()*255).astype(np.uint8))
def resize(x,size,down=False):return F.interpolate(x,size=size,mode='bilinear',align_corners=False,antialias=down)
def difference(a,b):
    d=(a-b).abs()
    return {'mean_abs':d.mean().item(),'max_abs':d.max().item(),'fraction_channels_gt_one_level':(d>1/255).float().mean().item()}


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    config_path=ROOT/'configs/diagnostic_scene_context.json';cfg=json.loads(config_path.read_text())
    source=RGBInference(ROOT/cfg['bundle']);bundle=source.bundle
    assert bundle['weights_sha256']==cfg['weights_sha256']
    manifest_path=ROOT/bundle['configuration']['manifest'];manifest=read_manifest(manifest_path)
    assert file_hash(manifest_path)==bundle['configuration']['manifest_sha256']
    rows={r['id']:r for r in manifest['pairs'] if r['split']=='diagnostic'}
    baseline=ROOT/'artifacts/phase2/identity_pilot/evaluation/control'
    prior={r['id']:r for r in json.loads((baseline/'summary.json').read_text())['records']}
    out=ROOT/cfg['output'];out.mkdir(exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    report={'protocol_sha256':file_hash(config_path),'weights_sha256':bundle['weights_sha256'],'manifest_sha256':file_hash(manifest_path),
            'model_forwards':0,'training_steps':0,'selection_images_used':0,'test_images_used':0,'challenging_images_used':0,'cases':{}}
    for identifier in cfg['ids']:
        start=time.perf_counter();row=rows[identifier];images=[]
        for key in ['raw','reference']:
            valid=validate_image(safe_path(ROOT/bundle['configuration']['data_root'],row[key+'_path']))
            assert valid.source_sha256==row[key+'_sha256'];images.append(valid.image)
        x,target=map(tensor,images);h,w=x.shape[-2:];assert min(h,w)>=cfg['crop_size']
        y0=((h-256)//2)//8*8;x0=((w-256)//2)//8*8
        scale=cfg['long_edge']/max(h,w);small=tuple(max(16,round(d*scale/8)*8) for d in [h,w])
        with torch.inference_mode():
            native_raw=source.model.forward_unclamped(x)
            crop=source.model(x[...,y0:y0+256,x0:x0+256])
            low_x=resize(x,small,True);low_raw=source.model.forward_unclamped(low_x)
            residual=native_raw-x;scaled=resize(low_raw-low_x,(h,w))
            filtered=resize(resize(residual,small,True),(h,w))
            corrected_mean=scaled+residual.mean((-2,-1),keepdim=True)-scaled.mean((-2,-1),keepdim=True)
            variants={'native':native_raw,'resampled_native_residual':x+filtered,'scene_scaled_residual':x+scaled,'scene_scaled_meanmatched':x+corrected_mean}
            original=pixels(native_raw.clamp(0,1));cached=baseline/'images'/(identifier+'.png')
            assert file_hash(cached)==prior[identifier]['png_sha256']
            with Image.open(cached) as old:assert old.tobytes()==original.tobytes()
            m=cfg['interior_margin'];native_crop=native_raw.clamp(0,1)[...,y0:y0+256,x0:x0+256]
            comparisons={'entire_crop':difference(crop,native_crop),'interior':difference(crop[...,m:-m,m:-m],native_crop[...,m:-m,m:-m])}
            metrics={};case=out/identifier;case.mkdir()
            overview=Image.new('RGB',(6*280,250),'white');draw=ImageDraw.Draw(overview)
            views=[('Input',images[0])]
            for name,raw in variants.items():
                pred=raw.clamp(0,1);error=pred-target;bias=error.mean((-2,-1),keepdim=True)
                metrics[name]={'psnr':psnr(pred,target).item(),'ssim':ssim(pred,target).item(),
                    'rgb_mean_error':bias.flatten().tolist(),'mean_bias_mse':bias.square().mean().item(),
                    'centered_error_mse':(error-bias).square().mean().item(),
                    'out_of_range_channel_fraction':((raw<0)|(raw>1)).float().mean().item()}
                png=pixels(pred);export_png(png,case/(name+'.png'));views.append((name,png))
            assert abs(metrics['native']['psnr']-prior[identifier]['psnr'])<1e-5
            assert abs(metrics['native']['ssim']-prior[identifier]['ssim'])<1e-6
            views.append(('Reference',images[1]))
            for col,(name,image) in enumerate(views):
                overview.paste(ImageOps.contain(image,(280,220)),(col*280,25));draw.text((col*280+2,4),name,fill='black')
            overview.save(case/'overview.png')
            np.savez_compressed(case/'native_residual.npz',residual=residual[0].permute(1,2,0).numpy())
        report['model_forwards']+=3
        report['cases'][identifier]={'source_sha256':row['raw_sha256'],'reference_sha256':row['reference_sha256'],
            'native_hw':[h,w],'small_hw':small,'aligned_crop_xywh':[x0,y0,256,256],'crop_comparison':comparisons,
            'metrics':metrics,'seconds':time.perf_counter()-start}
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'id':identifier,'crop_comparison':comparisons,'metrics':metrics}),flush=True)
    print('Completed exactly nine forwards on three diagnostic cases. No training.')

if __name__=='__main__':main()
