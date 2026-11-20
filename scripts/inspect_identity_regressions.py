"""Post-hoc inspection of cached diagnostic pilot regressions; no model calls."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from aurora.image import validate_image
from aurora.manifest import file_hash, read_manifest, safe_path


def mse_components(prediction, target):
    error=prediction-target
    bias=error.mean(axis=(0,1))
    mse=float(np.mean(error**2))
    mean_mse=float(np.mean(bias**2))
    centered=float(np.mean((error-bias)**2))
    if not np.isclose(mse,mean_mse+centered,atol=1e-12,rtol=1e-10):
        raise ValueError('MSE decomposition failed.')
    return {'mse':mse,'rgb_mean_error':bias.tolist(),'mean_bias_mse':mean_mse,'centered_error_mse':centered}


def main():
    pilot=ROOT/'artifacts/phase2/identity_pilot'
    previous=ROOT/'artifacts/phase2/diagnosis_v2/identity_pilot_review/comparison.json'
    prior=json.loads(previous.read_text())
    worst=sorted(prior['paired_deltas'],key=lambda r:r['psnr_delta'])[:5]
    ids=[r['id'] for r in worst]+['380_img_','729_img_','851_img_']
    assert len(set(ids))==len(ids)
    protocol={'scope':'Post-hoc diagnostic inspection, not independent validation',
              'selection':'Five largest native-float PSNR losses of identity versus matched control, plus three prior failure cases',
              'ids':ids,'pixel_analysis':'Decoded 8-bit RGB export comparisons in [0,1], not replacement native-float PSNR/SSIM',
              'native_crops':'Centered 256x256, except prior 380 contour (240,0,256,160), 729 contour (1400,210,256,256)',
              'summary_sha256':file_hash(pilot/'summary.json'),'comparison_sha256':file_hash(previous),
              'model_forwards':0,'training_steps':0,'selection_images_used':0,'test_images_used':0,'challenging_images_used':0}
    out=ROOT/'artifacts/phase2/diagnosis_v2/identity_regressions';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    manifest=read_manifest(ROOT/'data/metadata/manifest.v1.json')
    rows={r['id']:r for r in manifest['pairs'] if r['split']=='diagnostic'}
    summaries={name:json.loads((pilot/'evaluation'/name/'summary.json').read_text()) for name in ['control','identity']}
    records={name:{r['id']:r for r in s['records']} for name,s in summaries.items()}
    overview=Image.new('RGB',(4*320,len(ids)*260),'white');draw=ImageDraw.Draw(overview)
    crops=Image.new('RGB',(4*280,len(ids)*290),'white');cdraw=ImageDraw.Draw(crops)
    findings=[]
    for index,identifier in enumerate(ids):
        row=rows[identifier]
        images=[]
        hashes={}
        for name,key in [('input','raw'),('reference','reference')]:
            path=safe_path(ROOT/'data',row[key+'_path']);valid=validate_image(path)
            if valid.source_sha256!=row[key+'_sha256']:raise ValueError('Source image changed.')
            images.append(valid.image.copy());hashes[name]=valid.source_sha256
        raw,ref=images
        outputs=[]
        for name in ['control','identity']:
            path=pilot/'evaluation'/name/'images'/(identifier+'.png')
            hashes[name]=file_hash(path)
            if hashes[name]!=records[name][identifier]['png_sha256']:raise ValueError('Cached output changed.')
            with Image.open(path) as image:outputs.append(image.convert('RGB'))
        ordered=[raw,*outputs,ref]
        if any(image.size!=raw.size for image in ordered):raise ValueError('Geometry mismatch.')
        arrays=[np.asarray(image,dtype=np.float64)/255 for image in ordered]
        x,c,i,r=arrays;rc=c-x;ri=i-x
        control=mse_components(c,r);identity=mse_components(i,r)
        delta=identity['mse']-control['mse'];bias_delta=identity['mean_bias_mse']-control['mean_bias_mse']
        denominator=float(np.mean(rc*rc))
        projection=float(np.mean(ri*rc)/denominator) if denominator else None
        correction_ratio=float(np.sqrt(np.mean(ri*ri)/denominator)) if denominator else None
        h,w=x.shape[:2];roi=(240,0,256,160) if identifier=='380_img_' else (1400,210,256,256) if identifier=='729_img_' else (max(0,(w-256)//2),max(0,(h-256)//2),min(w,256),min(h,256))
        a,b,width,height=roi
        for column,(name,image) in enumerate(zip(['Input','Control','Identity','Reference'],ordered)):
            overview.paste(ImageOps.contain(image,(320,230)),(column*320,index*260+25));draw.text((column*320+3,index*260+3),identifier+' '+name,fill='black')
            crops.paste(image.crop((a,b,a+width,b+height)),(column*280,index*290+25));cdraw.text((column*280+3,index*290+3),identifier+' '+name,fill='black')
        findings.append({'id':identifier,'sha256':hashes,'native_crop_xywh':roi,
            'native_float_psnr_delta':records['identity'][identifier]['psnr']-records['control'][identifier]['psnr'],
            'native_float_ssim_delta':records['identity'][identifier]['ssim']-records['control'][identifier]['ssim'],
            'png_control':control,'png_identity':identity,'png_mse_delta':delta,
            'png_mean_bias_mse_delta':bias_delta,'png_centered_error_mse_delta':identity['centered_error_mse']-control['centered_error_mse'],
            'png_mean_bias_share_of_mse_regression':bias_delta/delta if delta>0 else None,
            'png_identity_vs_control_correction_rms_ratio':correction_ratio,'png_identity_residual_projection_onto_control':projection})
    overview.save(out/'overview.png');crops.save(out/'native_crops.png')
    report={'protocol':protocol,'findings':findings,'decision':'Pending visual interpretation; descriptive post-hoc diagnostics only.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k in ['id','native_float_psnr_delta','png_mean_bias_share_of_mse_regression','png_identity_vs_control_correction_rms_ratio','png_identity_residual_projection_onto_control']} for r in findings],indent=2))

if __name__=='__main__':main()
