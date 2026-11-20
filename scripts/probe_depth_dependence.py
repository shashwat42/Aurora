"""Fixed-three diagnostic follow-up: cache parity, mean maps and fusion removal."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aurora.inference import RGBInference,export_png
from aurora.image import validate_image
from aurora.manifest import read_manifest,safe_path,file_hash
from aurora.training.metrics import psnr,ssim


def main():
    torch.set_num_threads(4)
    bundle=ROOT/'artifacts/phase3/depth_baseline/epoch-020/bundle.json'
    service=RGBInference(bundle);cfg=service.bundle['configuration']
    manifest=read_manifest(ROOT/cfg['manifest'])
    rows={r['id']:r for r in manifest['pairs'] if r['split']=='diagnostic'}
    ids=['14_img_','380_img_','729_img_']
    out=ROOT/'artifacts/phase3/dependence_probe';out.mkdir(exist_ok=False)
    protocol={'ids':ids,'checkpoint_sha256':file_hash(bundle),'weights_sha256':service.bundle['weights_sha256'],
              'variants':['native','constant_native_mean','fusion_gates_zero'],
              'budget':'Nine generator forwards and one equipment live depth prediction; no training, selection or full-dataset evaluation.',
              'limitations':['Mean map preserves the scalar mean but removes variance and spatial structure together.',
                             'Zero gates removes all depth contributions including encoder/projection biases; it does not recreate separately trained RGB weights.',
                             'All interventions are exploratory out-of-training-distribution tests, not candidates.']}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rois={r['id']:r['native_crop_xywh'] for r in json.loads((ROOT/'artifacts/phase2/diagnosis_v2/identity_regressions/report.json').read_text())['findings']}
    records=[];cache_checks=[]
    for identifier in ids:
        row=rows[identifier]
        raw=validate_image(safe_path(ROOT/cfg['data_root'],row['raw_path']))
        ref=validate_image(safe_path(ROOT/cfg['data_root'],row['reference_path']))
        assert raw.source_sha256==row['raw_sha256'] and ref.source_sha256==row['reference_sha256']
        cached=service.depth.load_cached(raw);paths=service.depth._paths(raw)
        cache_record={'id':identifier,'input_sha256':raw.source_sha256,'map_sha256':file_hash(paths[0]),
                      'metadata_sha256':file_hash(paths[1]),'shape':list(cached.shape),
                      'depth_min':float(cached.min()),'depth_max':float(cached.max()),'depth_mean':float(cached.mean())}
        if identifier=='729_img_':
            live,meta=service.depth.predict(raw)
            cache_record.update(live_max_abs_difference=float(np.max(np.abs(live-cached))),
                                live_exact=np.array_equal(live,cached),live_geometry=meta['native_hw'])
        cache_checks.append(cache_record)
        x=torch.from_numpy(np.asarray(raw.image,dtype=np.float32).transpose(2,0,1).copy()/255).unsqueeze(0)
        target=torch.from_numpy(np.asarray(ref.image,dtype=np.float32).transpose(2,0,1).copy()/255).unsqueeze(0)
        d=torch.from_numpy(cached.copy()).unsqueeze(0).unsqueeze(0)
        images=[]
        for variant in protocol['variants']:
            original_gates=service.model.local.depth_gates.detach().clone()
            try:
                if variant=='fusion_gates_zero':
                    with torch.no_grad():service.model.local.depth_gates.zero_()
                depth=torch.full_like(d,float(d.mean())) if variant=='constant_native_mean' else d
                with torch.inference_mode():prediction=service.model(x,depth=depth)
            finally:
                with torch.no_grad():service.model.local.depth_gates.copy_(original_gates)
            a=prediction[0].permute(1,2,0).numpy();pixels=np.rint(a*255).astype(np.uint8)
            if variant=='native':
                with Image.open(ROOT/'artifacts/phase3/evaluation/epoch-020/diagnostic/images'/(identifier+'.png')) as im:
                    assert np.array_equal(pixels,np.asarray(im))
            error=a.astype(np.float64)-np.asarray(ref.image,dtype=np.float64)/255
            bias=error.mean((0,1));mse=float(np.mean(error**2));bias_mse=float(np.mean(bias**2));centered=float(np.mean((error-bias)**2))
            assert np.isclose(mse,bias_mse+centered,rtol=0,atol=1e-12)
            im=Image.fromarray(pixels);images.append(im);path=out/(identifier+'-'+variant+'.png');export_png(im,path)
            records.append({'id':identifier,'variant':variant,'psnr':float(psnr(prediction,target).item()),
                            'ssim':float(ssim(prediction,target).item()),'mean_rgb_error':bias.tolist(),
                            'mse':mse,'mean_bias_mse':bias_mse,'centered_error_mse':centered,
                            'clipped_endpoint_fraction':float(np.mean((a==0)|(a==1))),
                            'output_sha256':file_hash(path)})
        images.append(ref.image)
        overview=Image.new('RGB',(4*280,240),'white');od=ImageDraw.Draw(overview)
        detail=Image.new('RGB',(4*272,285),'white');dd=ImageDraw.Draw(detail)
        x0,y0,w,h=rois[identifier]
        for col,(label,im) in enumerate(zip(protocol['variants']+['reference'],images)):
            overview.paste(ImageOps.contain(im,(280,210)),(col*280,25));od.text((col*280+2,3),identifier+' '+label,fill='black')
            detail.paste(im.crop((x0,y0,x0+w,y0+h)),(col*272,25));dd.text((col*272+2,3),label,fill='black')
        overview.save(out/(identifier+'-overview.png'));detail.save(out/(identifier+'-native.png'))
        Image.fromarray(np.rint(cached*255).astype(np.uint8)).save(out/(identifier+'-depth.png'))
    report={'status':'probe_complete_interpretation_pending','protocol_sha256':file_hash(out/'protocol.json'),
            'script_sha256':file_hash(Path(__file__)),'records':records,'cache_checks':cache_checks,
            'model_forwards':9,'depth_estimator_forwards':1,'training_steps':0,
            'selection_images_used':0,'test_images_used':0,'challenging_images_used':0,
            'selected':False,'stage3_complete':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
