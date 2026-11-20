"""Verify completed spatial identity pilot and inspect fixed cached cases only."""
import json
from pathlib import Path
import statistics
import sys
from PIL import Image, ImageDraw, ImageOps
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aurora.manifest import file_hash, read_manifest, safe_path
from aurora.training.pilot import sequence_hash, reused_diagnostic


def main():
    run=ROOT/'artifacts/phase2/spatial_identity_pilot'
    summary=json.loads((run/'summary.json').read_text())
    protocol_path=ROOT/'configs/identity_pilot_spatial.json'
    protocol=json.loads(protocol_path.read_text())
    assert file_hash(protocol_path)==summary['protocol_sha256']
    assert json.loads((run/'protocol.json').read_text())==protocol
    old=ROOT/'artifacts/phase2/identity_pilot'
    old_summary=json.loads((old/'summary.json').read_text())
    configs={'control':ROOT/protocol['arms']['control'],'centered':ROOT/protocol['arms']['identity'],
             'full':ROOT/'configs/identity_pilot_identity.json'}
    paths={'control':old/'evaluation/control/summary.json','full':old/'evaluation/identity/summary.json',
           'centered':run/'evaluation/identity/summary.json'}
    evaluated={};bundles={};sequences=[]
    for name,cfgpath in configs.items():
        cfg=json.loads(cfgpath.read_text());folder=ROOT/cfg['output']
        bundle=json.loads((folder/'last/bundle.json').read_text());bundles[name]=bundle
        assert bundle['configuration']==cfg and bundle['run_kind']=='diagnostic_pilot' and not bundle['selected']
        assert file_hash(folder/'last'/bundle['weights'])==bundle['weights_sha256']
        expected=old_summary['results']['identity'] if name=='full' else summary['results']['control' if name=='control' else 'identity']
        assert bundle['weights_sha256']==expected['training']['weights_sha256']
        assert cfg['init_weights_sha256']==summary['source_weights_sha256']
        seq=sequence_hash(folder/'steps.jsonl');sequences.append(seq)
        assert seq==expected['training_sequence_sha256']
        evaluated[name]=reused_diagnostic({'summary':str(paths[name]),'summary_sha256':file_hash(paths[name])},bundle)
        assert evaluated[name]['count']==40
        for metric in ['psnr','ssim']:
            assert evaluated[name]['mean_'+metric]==expected['diagnostic_mean_'+metric]
    assert len(set(sequences))==1 and summary['training_order_matched']
    assert summary['results']['control']['diagnostic_evaluation_reused']
    assert not summary['results']['identity']['diagnostic_evaluation_reused']
    a,b=summary['results']['control'],summary['results']['identity']
    records={n:{r['id']:r for r in e['records']} for n,e in evaluated.items()}
    paired=[{'id':i,**{m+'_delta':records['centered'][i][m]-records['control'][i][m] for m in ['psnr','ssim']}} for i in sorted(records['control'])]
    report={'status':'verified_complete_continuation_criterion_failed_not_promoted','summary_sha256':file_hash(run/'summary.json'),
        'protocol_sha256':file_hash(protocol_path),'training_order_matched':True,
        'metrics':{n:{m:e['mean_'+m] for m in ['psnr','ssim']} for n,e in evaluated.items()},
        'centered_vs_control':{'mean_psnr_delta':b['diagnostic_mean_psnr']-a['diagnostic_mean_psnr'],
        'mean_ssim_delta':b['diagnostic_mean_ssim']-a['diagnostic_mean_ssim'],
        'centered_reference_change_reduction_percent':100*(1-b['mean_centered_identity_change']/a['mean_centered_identity_change']),
        'total_reference_change_reduction_percent':100*(1-b['mean_identity_change']/a['mean_identity_change']),
        'psnr_wins':sum(r['psnr_delta']>0 for r in paired),'ssim_wins':sum(r['ssim_delta']>0 for r in paired),
        'both_wins':sum(r['psnr_delta']>0 and r['ssim_delta']>0 for r in paired)},
        'continuation_criteria':{'both_enhancement_means_at_least_control':b['diagnostic_mean_psnr']>=a['diagnostic_mean_psnr'] and b['diagnostic_mean_ssim']>=a['diagnostic_mean_ssim'],
        'centered_reference_change_lower':b['mean_centered_identity_change']<a['mean_centered_identity_change']},
        'paired_deltas':paired,'model_forwards':0,'training_steps':0,'selection_images_used':0,'test_images_used':0,
        'challenging_images_used':0,'selected':False,'phase2_complete':False,
        'decision':'Do not promote. Stop this identity-loss diagnostic cycle; no automatic weight sweep or further training.'}
    out=ROOT/'artifacts/phase2/diagnosis_v2/spatial_identity_review';out.mkdir(exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    previous=json.loads((ROOT/'artifacts/phase2/diagnosis_v2/identity_regressions/protocol.json').read_text())
    ids=previous['ids'];rows={r['id']:r for r in read_manifest(ROOT/'data/metadata/manifest.v1.json')['pairs'] if r['split']=='diagnostic'}
    rois={r['id']:r['native_crop_xywh'] for r in json.loads((ROOT/'artifacts/phase2/diagnosis_v2/identity_regressions/report.json').read_text())['findings']}
    overview=Image.new('RGB',(5*280,len(ids)*235),'white');draw=ImageDraw.Draw(overview)
    crops=Image.new('RGB',(5*280,len(ids)*290),'white');cdraw=ImageDraw.Draw(crops)
    for index,i in enumerate(ids):
        row=rows[i]
        sources=[safe_path(ROOT/'data',row['raw_path']),paths['control'].parent/'images'/(i+'.png'),
                 paths['full'].parent/'images'/(i+'.png'),paths['centered'].parent/'images'/(i+'.png'),safe_path(ROOT/'data',row['reference_path'])]
        for col,(label,path) in enumerate(zip(['Input','Control','Full identity','Centered identity','Reference'],sources)):
            with Image.open(path) as image:image=image.convert('RGB')
            overview.paste(ImageOps.contain(image,(280,210)),(col*280,index*235+25));draw.text((col*280+3,index*235+3),i+' '+label,fill='black')
            x,y,w,h=rois[i];crops.paste(image.crop((x,y,x+w,y+h)),(col*280,index*290+25));cdraw.text((col*280+3,index*290+3),i+' '+label,fill='black')
    overview.save(out/'overview.png');crops.save(out/'native_crops.png')
    print(json.dumps({k:v for k,v in report.items() if k!='paired_deltas'},indent=2))

if __name__=='__main__':main()
