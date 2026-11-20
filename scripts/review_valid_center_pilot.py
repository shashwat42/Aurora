"""Validate completed valid-center pilot and render fixed cached visual cases."""
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aurora.manifest import file_hash, read_manifest, safe_path
from aurora.image import validate_image
from aurora.training.pilot import sequence_hash,reused_diagnostic,native_edge_diagnostics


def main():
    run=ROOT/'artifacts/phase2/valid_center_pilot';summary=json.loads((run/'summary.json').read_text())
    path=ROOT/'configs/valid_center_pilot.json';protocol=json.loads(path.read_text())
    assert file_hash(path)==summary['protocol_sha256'] and json.loads((run/'protocol.json').read_text())==protocol
    evaluated={};paths={};configs={};sequences=[];edge_checks={}
    for name in ['control','treatment']:
        cfg=json.loads((ROOT/protocol['arms'][name]).read_text());configs[name]=cfg
        directory=ROOT/cfg['output'];bundle=json.loads((directory/'last/bundle.json').read_text());reported=summary['results'][name]
        assert bundle['configuration']==cfg and bundle['run_kind']=='diagnostic_pilot' and not bundle['selected']
        assert file_hash(directory/'last'/bundle['weights'])==bundle['weights_sha256']==reported['training']['weights_sha256']
        assert cfg['init_weights_sha256']==protocol['source_weights_sha256']==summary['source_weights_sha256']
        assert cfg.get('loss_margin',0)==reported['loss_margin']
        seq=sequence_hash(directory/'steps.jsonl');sequences.append(seq);assert seq==reported['training_sequence_sha256']
        reuse=protocol.get('reuse_diagnostic_evaluation',{}).get(name)
        paths[name]=ROOT/reuse['summary'] if reuse else run/'evaluation'/name/'summary.json'
        spec={'summary':str(paths[name]),'summary_sha256':reuse['summary_sha256'] if reuse else file_hash(paths[name])}
        evaluated[name]=reused_diagnostic(spec,bundle)
        assert evaluated[name]['count']==reported['diagnostic_count']==40
        for metric in ['psnr','ssim']:assert evaluated[name]['mean_'+metric]==reported['diagnostic_mean_'+metric]
        edges=native_edge_diagnostics(evaluated[name],paths[name].parent,cfg,protocol['native_edge_ids'],protocol['native_edge_band'])
        assert edges==json.loads((run/('native_edges_'+name+'.json')).read_text())
        assert {k:v for k,v in edges.items() if k!='records'}==reported['native_edges']
        edge_checks[name]=edges
    assert len(set(sequences))==1 and summary['training_order_matched']
    indexed={n:{r['id']:r for r in e['records']} for n,e in evaluated.items()}
    paired=[{'id':i,**{m+'_delta':indexed['treatment'][i][m]-indexed['control'][i][m] for m in ['psnr','ssim']}} for i in sorted(indexed['control'])]
    a,b=summary['results']['control'],summary['results']['treatment']
    border_delta=100*(b['native_edges']['mean_border_mse']/a['native_edges']['mean_border_mse']-1)
    interior_delta=100*(b['native_edges']['mean_interior_mse']/a['native_edges']['mean_interior_mse']-1)
    report={'status':'verified_complete_criteria_failed_not_promoted','protocol_sha256':file_hash(path),'summary_sha256':file_hash(run/'summary.json'),
        'training_order_matched':True,'mean_psnr_delta':b['diagnostic_mean_psnr']-a['diagnostic_mean_psnr'],
        'mean_ssim_delta':b['diagnostic_mean_ssim']-a['diagnostic_mean_ssim'],
        'native_border_mse_change_percent':border_delta,'native_interior_mse_change_percent':interior_delta,
        'psnr_wins':sum(r['psnr_delta']>0 for r in paired),'ssim_wins':sum(r['ssim_delta']>0 for r in paired),
        'both_wins':sum(r['psnr_delta']>0 and r['ssim_delta']>0 for r in paired),
        'criteria':{'both_primary_means_at_least_control':b['diagnostic_mean_psnr']>=a['diagnostic_mean_psnr'] and b['diagnostic_mean_ssim']>=a['diagnostic_mean_ssim'],
        'native_border_mse_not_increased':border_delta<=0},'paired_deltas':paired,
        'decision':'Do not promote or automatically sweep crop margins. No further training scheduled.',
        'model_forwards':0,'training_steps':0,'selection_images_used':0,'test_images_used':0,'challenging_images_used':0,'selected':False,'phase2_complete':False}
    out=ROOT/'artifacts/phase2/diagnosis_v2/valid_center_review';out.mkdir(exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    rows={r['id']:r for r in read_manifest(ROOT/configs['control']['manifest'])['pairs'] if r['split']=='diagnostic'}
    rois={r['id']:r['native_crop_xywh'] for r in json.loads((ROOT/'artifacts/phase2/diagnosis_v2/identity_regressions/report.json').read_text())['findings']}
    ids=protocol['native_edge_ids'];overview=Image.new('RGB',(4*280,len(ids)*235),'white');od=ImageDraw.Draw(overview)
    crops=Image.new('RGB',(4*280,len(ids)*290),'white');cd=ImageDraw.Draw(crops)
    edges=Image.new('RGB',(4*160,len(ids)*160),'white');ed=ImageDraw.Draw(edges)
    for index,i in enumerate(ids):
        row=rows[i]
        raw=validate_image(safe_path(ROOT/configs['control']['data_root'],row['raw_path']));ref=validate_image(safe_path(ROOT/configs['control']['data_root'],row['reference_path']))
        assert raw.source_sha256==row['raw_sha256'] and ref.source_sha256==row['reference_sha256']
        images=[raw.image]
        for name in ['control','treatment']:
            with Image.open(paths[name].parent/'images'/(i+'.png')) as image:images.append(image.convert('RGB'))
        images.append(ref.image)
        for col,(name,image) in enumerate(zip(['Input','Control','Valid center','Reference'],images)):
            overview.paste(ImageOps.contain(image,(280,210)),(col*280,index*235+25));od.text((col*280+2,index*235+3),i+' '+name,fill='black')
            x,y,w,h=rois[i];crops.paste(image.crop((x,y,x+w,y+h)),(col*280,index*290+25));cd.text((col*280+2,index*290+3),i+' '+name,fill='black')
            edges.paste(image.crop((0,0,128,128)),(col*160,index*160+25));ed.text((col*160+2,index*160+3),i+' '+name,fill='black')
    overview.save(out/'overview.png');crops.save(out/'native_crops.png');edges.save(out/'native_top_left_edges.png')
    print(json.dumps({k:v for k,v in report.items() if k!='paired_deltas'},indent=2))

if __name__=='__main__':main()
