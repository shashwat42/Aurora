"""Verify completed diagnostic pilot and render cached comparisons; no model forwards."""
import hashlib
import json
from pathlib import Path
import sys
import statistics
from PIL import Image, ImageDraw, ImageOps
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aurora.manifest import file_hash, read_manifest, safe_path
from aurora.training.pilot import sequence_hash


def main():
    pilot = ROOT / 'artifacts/phase2/identity_pilot'
    result = json.loads((pilot/'summary.json').read_text())
    protocol = json.loads((pilot/'protocol.json').read_text())
    assert file_hash(ROOT/'configs/identity_pilot.json') == result['protocol_sha256']
    assert protocol == json.loads((ROOT/'configs/identity_pilot.json').read_text())
    manifest_path = ROOT/'data/metadata/manifest.v1.json'
    manifest = read_manifest(manifest_path)
    rows = {r['id']: r for r in manifest['pairs'] if r['split']=='diagnostic'}
    cached = {}
    for name in ['control','identity']:
        bundle = json.loads((pilot/name/'last/bundle.json').read_text())
        cfg = json.loads((ROOT/protocol['arms'][name]).read_text())
        assert bundle['configuration'] == cfg
        assert bundle['run_kind']=='diagnostic_pilot' and not bundle['selected']
        assert bundle['weights_sha256']==file_hash(pilot/name/'last'/bundle['weights'])==result['results'][name]['training']['weights_sha256']
        assert cfg['manifest_sha256']==file_hash(manifest_path)
        assert cfg['init_weights_sha256']==result['source_weights_sha256']
        assert sequence_hash(pilot/name/'steps.jsonl')==result['results'][name]['training_sequence_sha256']
        summary=json.loads((pilot/'evaluation'/name/'summary.json').read_text())
        assert summary['split']=='diagnostic' and summary['count']==len(rows)==40
        assert summary['weights_sha256']==bundle['weights_sha256']
        for metric in ['psnr','ssim']:
            assert summary[f'mean_{metric}']==result['results'][name][f'diagnostic_mean_{metric}']
        assert {r['id'] for r in summary['records']}==set(rows)
        for record in summary['records']:
            path=pilot/'evaluation'/name/'images'/(record['id']+'.png')
            assert file_hash(path)==record['png_sha256']
            assert record['raw_sha256']==rows[record['id']]['raw_sha256']
            assert record['reference_sha256']==rows[record['id']]['reference_sha256']
        cached[name]=summary
    assert result['training_order_matched'] and result['results']['control']['training_sequence_sha256']==result['results']['identity']['training_sequence_sha256']
    baseline=ROOT/'artifacts/phase2/evaluation_v2/epoch-020/diagnostic'
    cached['v2_epoch20']=json.loads((baseline/'summary.json').read_text())
    per_id={name:{r['id']:r for r in summary['records']} for name,summary in cached.items()}
    deltas=[{'id':identifier, **{f'{metric}_delta':per_id['identity'][identifier][metric]-per_id['control'][identifier][metric] for metric in ['psnr','ssim']}} for identifier in sorted(rows)]
    control=result['results']['control']['mean_identity_change'];identity=result['results']['identity']['mean_identity_change']
    analysis={'status':'verified_diagnostic_pilot_not_selected','scope':'Cached diagnostic records and images only; zero training steps or model forwards',
        'summary_sha256':file_hash(pilot/'summary.json'),'training_order_matched':True,
        'metrics':{n:{'psnr':s['mean_psnr'],'ssim':s['mean_ssim']} for n,s in cached.items()},
        'identity_vs_control':{'mean_psnr_delta':statistics.mean(d['psnr_delta'] for d in deltas),'mean_ssim_delta':statistics.mean(d['ssim_delta'] for d in deltas),'psnr_wins':sum(d['psnr_delta']>0 for d in deltas),'ssim_wins':sum(d['ssim_delta']>0 for d in deltas),'both_wins':sum(d['psnr_delta']>0 and d['ssim_delta']>0 for d in deltas),'mean_reference_identity_change_control':control,'mean_reference_identity_change_treatment':identity,'relative_identity_change_reduction_percent':100*(control-identity)/control},
        'paired_deltas':deltas,'selection_images_used':0,'test_images_used':0,'challenging_images_used':0,'selected':False,'phase2_complete':False}
    out=ROOT/'artifacts/phase2/diagnosis_v2/identity_pilot_review';out.mkdir(exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(analysis,indent=2)+'\n')
    ids=protocol['identity_diagnostic_ids']
    overview=Image.new('RGB',(5*280,6*240),'white');draw=ImageDraw.Draw(overview)
    contour=Image.new('RGB',(5*280,2*310),'white');cdraw=ImageDraw.Draw(contour)
    rois={'380_img_':(240,0,256,160),'729_img_':(1400,210,256,256)}
    for index,identifier in enumerate(ids):
        row=rows[identifier]
        raw=ROOT/'data'/row['raw_path'];ref=ROOT/'data'/row['reference_path']
        assert file_hash(raw)==row['raw_sha256'] and file_hash(ref)==row['reference_sha256']
        old=baseline/'images'/(identifier+'.png');assert file_hash(old)==per_id['v2_epoch20'][identifier]['png_sha256']
        paths=[raw,old,pilot/'evaluation/control/images'/(identifier+'.png'),pilot/'evaluation/identity/images'/(identifier+'.png'),ref]
        for column,(label,path) in enumerate(zip(['Input','V2 epoch20','Control +2','Identity +2','Reference'],paths)):
            with Image.open(path) as opened:image=opened.convert('RGB')
            tile=ImageOps.contain(image,(280,210));overview.paste(tile,(column*280,index*240+25));draw.text((column*280+3,index*240+4),identifier+' '+label,fill='black')
            if identifier in rois:
                x,y,w,h=rois[identifier];r=list(rois).index(identifier)
                contour.paste(image.crop((x,y,x+w,y+h)),(column*280,r*310+30));cdraw.text((column*280+3,r*310+4),identifier+' '+label,fill='black')
    overview.save(out/'overview.png');contour.save(out/'native_contours.png')
    print(json.dumps({k:v for k,v in analysis.items() if k!='paired_deltas'},indent=2))

if __name__=='__main__':main()
