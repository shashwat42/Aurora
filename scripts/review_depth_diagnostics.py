"""Cached run audit and bounded final-checkpoint depth intervention probe."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aurora.manifest import file_hash,read_manifest,safe_path
from aurora.image import validate_image
from aurora.inference import RGBInference,export_png
from aurora.training.metrics import psnr,ssim


def main():
    torch.set_num_threads(4)
    cfg_path=ROOT/'configs/depth_baseline.json';cfg=json.loads(cfg_path.read_text())
    run=ROOT/cfg['output'];evaluation=ROOT/'artifacts/phase3/evaluation'
    rgb_run=ROOT/'artifacts/phase2/rgb_baseline_v3';rgb_eval=ROOT/'artifacts/phase2/evaluation_v3'
    manifest=read_manifest(ROOT/cfg['manifest']);rows={r['id']:r for r in manifest['pairs'] if r['split']=='diagnostic'}
    train={r['id'] for r in manifest['pairs'] if r['split']=='train'}
    assert len(rows)==40 and len(train)==713 and file_hash(ROOT/cfg['manifest'])==cfg['manifest_sha256']
    status=json.loads((run/'status.json').read_text());assert status['status']=='complete' and status['steps']==7140
    assert json.loads((run/'config.json').read_text())==cfg
    provenance=json.loads((run/'provenance.json').read_text())
    assert provenance['config_sha256']==file_hash(cfg_path)
    assert all(file_hash(ROOT/'src/aurora'/p)==h for p,h in provenance['source_sha256'].items())
    logs=[]
    for directory in [run,rgb_run]:
        logs.append([json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()])
    assert len(logs[0])==len(logs[1])==7140
    counts={}
    for a,b in zip(*logs):
        assert all(a[k]==b[k] for k in ['step','epoch','ids'])
        assert all(np.isfinite(a[k]) for k in ['loss','gradient_norm'])
        counts.setdefault(a['epoch'],Counter()).update(a['ids'])
    assert set(counts)==set(range(1,21)) and all(c==Counter({i:1 for i in train}) for c in counts.values())
    summaries={};comparisons=[]
    for epoch in [5,10,20]:
        name=f'epoch-{epoch:03d}';bundle_path=run/name/'bundle.json';bundle=json.loads(bundle_path.read_text())
        assert bundle['configuration']==cfg and bundle['provenance']==provenance
        assert file_hash(bundle_path.parent/'weights.pt')==bundle['weights_sha256']
        assert file_hash(bundle_path.parent/'training_state.pt')==bundle['training_state_sha256']
        assert bundle['training_steps']==epoch*357
        for kind,base in [('depth',evaluation),('rgb',rgb_eval)]:
            folder=base/name/'diagnostic';s=json.loads((folder/'summary.json').read_text())
            assert s['count']==len(s['records'])==40 and {r['id'] for r in s['records']}==set(rows)
            for r in s['records']:
                wrapped=json.loads((folder/'records'/(r['id']+'.json')).read_text())
                assert wrapped['record']==r and hashlib.sha256(json.dumps(r,sort_keys=True,allow_nan=False).encode()).hexdigest()==wrapped['record_sha256']
                assert file_hash(folder/'images'/(r['id']+'.png'))==r['png_sha256']
                assert r['raw_sha256']==rows[r['id']]['raw_sha256'] and r['reference_sha256']==rows[r['id']]['reference_sha256']
            for m in ['psnr','ssim']:assert np.isclose(np.mean([float(r[m]) for r in s['records']]),s['mean_'+m],rtol=0,atol=1e-12)
            if kind=='depth':
                identity=json.loads((folder/'identity.json').read_text());assert identity['bundle_sha256']==file_hash(bundle_path)
                assert identity['weights_sha256']==bundle['weights_sha256'] and identity['split']=='diagnostic'
            summaries[kind,name]=s
        comparisons.append({'epoch':epoch,**{kind+'_'+m:summaries[kind,name]['mean_'+m] for kind in ['rgb','depth'] for m in ['psnr','ssim']},
                            **{m+'_delta':summaries['depth',name]['mean_'+m]-summaries['rgb',name]['mean_'+m] for m in ['psnr','ssim']}})
    assert not (evaluation/'ledger.json').exists() and not any(evaluation.glob('epoch-*/selection'))
    out=ROOT/'artifacts/phase3/diagnostic_review';out.mkdir(exist_ok=False)
    fixed=json.loads((ROOT/'configs/valid_center_pilot.json').read_text())['native_edge_ids']
    rois={r['id']:r['native_crop_xywh'] for r in json.loads((ROOT/'artifacts/phase2/diagnosis_v2/identity_regressions/report.json').read_text())['findings']}
    for start in [0,4]:
        overview=Image.new('RGB',(5*280,4*240),'white');od=ImageDraw.Draw(overview)
        detail=Image.new('RGB',(5*272,4*285),'white');dd=ImageDraw.Draw(detail)
        for index,i in enumerate(fixed[start:start+4]):
            row=rows[i];images=[]
            for key in ['raw','reference']:
                valid=validate_image(safe_path(ROOT/cfg['data_root'],row[key+'_path']));assert valid.source_sha256==row[key+'_sha256'];images.append(valid.image)
            raw,ref=images;ordered=[raw]
            for base,name in [(rgb_eval,'epoch-020'),(evaluation,'epoch-010'),(evaluation,'epoch-020')]:
                with Image.open(base/name/'diagnostic/images'/(i+'.png')) as im:ordered.append(im.convert('RGB'))
            ordered.append(ref);x,y,w,h=rois[i]
            for col,(label,im) in enumerate(zip(['Input','RGB20','Depth10','Depth20','Reference'],ordered)):
                assert im.size==raw.size
                overview.paste(ImageOps.contain(im,(280,210)),(col*280,index*240+25));od.text((col*280+2,index*240+3),i+' '+label,fill='black')
                detail.paste(im.crop((x,y,x+w,y+h)),(col*272,index*285+25));dd.text((col*272+2,index*285+3),i+' '+label,fill='black')
        overview.save(out/f'overview-{start//4+1}.png');detail.save(out/f'native-{start//4+1}.png')
    # Declared fixed three cases, final checkpoint only; exactly nine forwards.
    service=RGBInference(run/'epoch-020/bundle.json');probes=[]
    for i in ['14_img_','380_img_','729_img_']:
        row=rows[i];valid=validate_image(safe_path(ROOT/cfg['data_root'],row['raw_path']))
        ref=validate_image(safe_path(ROOT/cfg['data_root'],row['reference_path']))
        x=torch.from_numpy(np.asarray(valid.image,dtype=np.float32).transpose(2,0,1).copy()/255).unsqueeze(0)
        target=torch.from_numpy(np.asarray(ref.image,dtype=np.float32).transpose(2,0,1).copy()/255).unsqueeze(0)
        d=torch.from_numpy(service.depth.load_cached(valid).copy()).unsqueeze(0).unsqueeze(0)
        sheet=Image.new('RGB',(3*280,240),'white');draw=ImageDraw.Draw(sheet)
        for col,(variant,depth) in enumerate([('native',d),('omitted_zero',torch.zeros_like(d)),('horizontal_flip',d.flip(-1))]):
            with torch.inference_mode():prediction=service.model(x,depth=depth)
            pixels=np.rint(prediction[0].permute(1,2,0).numpy()*255).astype(np.uint8)
            if variant=='native':
                with Image.open(evaluation/'epoch-020/diagnostic/images'/(i+'.png')) as cached:assert np.array_equal(pixels,np.asarray(cached))
            image=Image.fromarray(pixels);export_png(image,out/(i+'-'+variant+'.png'))
            probes.append({'id':i,'variant':variant,'psnr':float(psnr(prediction,target).item()),'ssim':float(ssim(prediction,target).item()),
                           'output_sha256':file_hash(out/(i+'-'+variant+'.png'))})
            sheet.paste(ImageOps.contain(image,(280,210)),(col*280,25));draw.text((col*280+2,3),i+' '+variant,fill='black')
        sheet.save(out/(i+'-interventions.png'))
    report={'status':'verified_native_review_pending','comparisons':comparisons,'training_steps_verified':7140,
            'training_sequence_matches_rgb':True,'config_sha256':file_hash(cfg_path),'probes':probes,
            'probe_scope':'Fixed3 diagnostic images; final epoch20 only; zero depth and horizontally flipped depth are interventions, not retrained models or candidates.',
            'depth_gates_tanh':torch.tanh(service.model.local.depth_gates).detach().tolist(),
            'model_forwards':9,'training_steps_run':0,'full_metrics_recomputed':False,'selection_images_used':0,
            'test_images_used':0,'challenging_images_used':0,'selected':False,'stage3_complete':False,
            'script_sha256':file_hash(Path(__file__))}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
