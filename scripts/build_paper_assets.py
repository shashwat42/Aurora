"""Build reproducible paper figures/tables from verified diagnostic records."""
import csv,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
# Pure-Python ReportLab supplied by the bundled artifact runtime; no second venv.
sys.path.append('/home/shashwatsingh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
from reportlab.graphics.shapes import Drawing,String,Rect,Line,Polygon
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor
from aurora.manifest import read_manifest,safe_path,file_hash
from aurora.image import validate_image

OUT=ROOT/'artifacts/paper_integration';F=OUT/'figures';T=OUT/'tables';E=OUT/'evidence'
COLORS=['#334155','#0891b2','#7c3aed']


def text(d,x,y,s,size=11,color='#334155',anchor='start'):
    d.add(String(x,y,s,fontName='Helvetica',fontSize=size,fillColor=HexColor(color),textAnchor=anchor))


def save(d,name):
    d.contents.insert(0,Rect(0,0,d.width,d.height,fillColor=HexColor('#ffffff'),strokeColor=None))
    renderSVG.drawToFile(d,str(F/(name+'.svg')))


def box(d,x,y,w,h,lines,color='#e0f2fe'):
    d.add(Rect(x,y,w,h,rx=6,ry=6,fillColor=HexColor(color),strokeColor=HexColor('#94a3b8')))
    for i,s in enumerate(lines):text(d,x+w/2,y+h/2+(len(lines)-1)*7-i*14,s,10,anchor='middle')


def arrow(d,x,y,x2,y2):
    d.add(Line(x,y,x2,y2,strokeColor=HexColor('#64748b'),strokeWidth=1.2))
    a=np.arctan2(y2-y,x2-x);pts=[x2,y2,x2-7*np.cos(a-.45),y2-7*np.sin(a-.45),x2-7*np.cos(a+.45),y2-7*np.sin(a+.45)]
    d.add(Polygon(pts,fillColor=HexColor('#64748b'),strokeColor=None))


def main():
    for p in [F,T,E,OUT/'examples']:p.mkdir(parents=True,exist_ok=True)
    metrics=list(csv.DictReader((ROOT/'artifacts/phase3/mvp/paper_metrics.csv').open()))
    for r in metrics:
        assert file_hash(ROOT/r['summary_path'])==r['summary_sha256']
    shutil.copyfile(ROOT/'artifacts/phase3/mvp/paper_metrics.csv',T/'diagnostic_metrics.csv')
    shutil.copyfile(ROOT/'artifacts/phase3/mvp/profile_verification/paper_profile.csv',T/'runtime_profile.csv')
    for name,path in [('profile_verification.json','artifacts/phase3/mvp/profile_verification/report.json'),('workflow.json','artifacts/phase3/mvp/live_workflow.json'),('model_decision.json','artifacts/phase3/mvp/model_decision.json'),('selective_review.json','artifacts/phase3/selective_review/decision.json')]:
        shutil.copyfile(ROOT/path,E/name)
    labels=['RGB V3','Scalar depth','Selective depth']
    for metric,lo,hi,step in [('psnr',20,21.5,.25),('ssim',.84,.89,.01)]:
        d=Drawing(720,330);text(d,28,305,'Diagnostic fidelity: '+('PSNR (dB)' if metric=='psnr' else 'SSIM'),17)
        chart=LinePlot();chart.x=66;chart.y=64;chart.width=610;chart.height=180
        chart.data=[[(int(r['epoch']),float(r[metric])) for r in metrics if r['model']==label] for label in labels]
        chart.xValueAxis.valueMin=5;chart.xValueAxis.valueMax=20;chart.xValueAxis.valueSteps=[5,10,15,20]
        chart.yValueAxis.valueMin=lo;chart.yValueAxis.valueMax=hi;chart.yValueAxis.valueStep=step
        chart.yValueAxis.labels.fontSize=10;chart.xValueAxis.labels.fontSize=10
        for i,color in enumerate(COLORS):chart.lines[i].strokeColor=HexColor(color);chart.lines[i].strokeWidth=2;chart.lines[i].symbol=makeMarker('FilledCircle');chart.lines[i].symbol.fillColor=HexColor(color);chart.lines[i].symbol.size=5
        d.add(chart)
        for i,label in enumerate(labels):
            d.add(Line(72+i*210,275,95+i*210,275,strokeColor=HexColor(COLORS[i]),strokeWidth=3));text(d,102+i*210,271,label)
        text(d,372,34,'Training epoch',11,anchor='middle');text(d,28,12,'Native diagnostic40; equal image weighting; development evidence, not final-test results.',9)
        save(d,'diagnostic_'+metric)
    profile=json.loads((ROOT/'artifacts/phase3/mvp/profile/summary.json').read_text())
    for name,key,title,unit,maxval in [('latency','p95_seconds','Measured worker-pipeline latency','seconds',10),('memory','max_worker_rss_bytes','Maximum observed worker memory','GB (decimal)',3)]:
        d=Drawing(720,330);text(d,28,305,title,17)
        c=VerticalBarChart();c.x=66;c.y=65;c.width=610;c.height=190
        values=[r[key]/(1e9 if name=='memory' else 1) for r in profile['summary']]
        c.data=[values];c.categoryAxis.categoryNames=[f"{r['width']} x {r['height']}" for r in profile['summary']]
        c.valueAxis.valueMin=0;c.valueAxis.valueMax=maxval;c.valueAxis.valueStep=1 if name=='latency' else .5
        c.bars[0].fillColor=HexColor('#0891b2');c.bars[0].strokeColor=None
        c.barLabelFormat='%.3f';c.barLabels.fontSize=11;c.barLabels.nudge=8;c.categoryAxis.labels.fontSize=10
        d.add(c);text(d,66,272,'p95, seconds' if name=='latency' else unit,10)
        text(d,28,30,'Intel i7-1255U | n=30 per image after 3 warmups | fresh models + live depth',10)
        text(d,28,12,'Three diagnostic cases only; excludes upload/browser latency and parent-process RSS.',9)
        save(d,'runtime_'+name)
    d=Drawing(760,520);text(d,24,493,'Implemented selective-depth MVP architecture',17)
    box(d,265,413,230,52,['Validated native RGB input','EXIF orientation + sRGB'])
    box(d,25,306,220,66,['Frozen Depth Anything V2 Small','Native bilinear restoration','Per-image min-max normalization'])
    box(d,270,306,220,66,['RGB U-Net encoder','Channels: 16 / 32 / 64 / 128'], '#dcfce7')
    box(d,520,306,215,66,['Full-input RGB mean + std','6 -> 32 -> 6 colour MLP'], '#fef3c7')
    arrow(d,310,413,140,372);arrow(d,380,413,380,372);arrow(d,450,413,620,372)
    box(d,25,198,220,66,['Trainable depth encoder','Channels: 4 / 8 / 16 / 32','1x1 projections'], '#dcfce7')
    box(d,270,198,220,66,['Four aligned fusion scales','RGB/depth-conditioned sigmoid','x learned tanh scalar gates'], '#dcfce7')
    arrow(d,135,306,135,264);arrow(d,245,230,270,230);arrow(d,380,306,380,264)
    box(d,270,100,220,58,['U-Net decoder + skips','Local residual: 0.5 x tanh'], '#dcfce7');arrow(d,380,198,380,158)
    box(d,520,100,215,58,['Channel gain delta +/-0.5','Channel offset +/-0.25'], '#fef3c7');arrow(d,627,306,627,158)
    box(d,270,18,465,50,['Add input + local residual + colour correction','Clip [0,1] -> unpad -> native-size sRGB PNG'])
    arrow(d,380,100,380,68);arrow(d,627,100,627,68)
    text(d,25,125,'Blue: fixed preprocessing',9);text(d,25,110,'Green/yellow: trainable',9)
    text(d,25,72,'517,913 generator parameters',9);text(d,25,57,'Frozen estimator excluded',9)
    text(d,25,18,'Gates are not confidence.',9)
    save(d,'architecture')
    d=Drawing(760,310);text(d,24,284,'Runtime lifecycle and cancellation',17)
    items=[('Upload + validate','RGB / size / sealed-input checks'),('Allocate operation token','Reject stale results'),('Spawn worker','Load pinned models'),('Estimate live depth','Depth failure is explicit'),('Enhance + export','Native PNG + sRGB'),('Compare + download','Reopen or retry')]
    for i,(a,b) in enumerate(items):
        x=20+(i%3)*250;y=172 if i<3 else 65
        box(d,x,y,220,66,[a,b], '#e0f2fe' if i!=3 else '#dcfce7')
        if i%3<2:arrow(d,x+220,y+33,x+250,y+33)
    arrow(d,630,172,630,150);d.add(Line(630,150,130,150,strokeColor=HexColor('#64748b')));arrow(d,130,150,130,131)
    text(d,25,20,'Cancel / replacement upload: invalidate token -> terminate worker -> clear output -> permit retry.',10)
    save(d,'lifecycle')
    manifest=read_manifest(ROOT/'data/metadata/manifest.v1.json');rows={r['id']:r for r in manifest['pairs'] if r['split']=='diagnostic'}
    rgbbase=ROOT/'artifacts/phase2/evaluation_v3/epoch-020/diagnostic';depthbase=ROOT/'artifacts/phase3/selective_evaluation/epoch-020/diagnostic'
    indexes={name:{r['id']:r for r in json.loads((base/'summary.json').read_text())['records']} for name,base in [('RGB',rgbbase),('MVP',depthbase)]}
    rois={r['id']:r['native_crop_xywh'] for r in json.loads((ROOT/'artifacts/phase2/diagnosis_v2/identity_regressions/report.json').read_text())['findings']}
    ids=['14_img_','887_img_','284_img_','380_img_','729_img_','851_img_']
    font=ImageFont.truetype('/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf',17)
    provenance=[]
    for page in range(2):
        whole=Image.new('RGB',(4*360,3*300),'white');wd=ImageDraw.Draw(whole)
        detail=Image.new('RGB',(4*280,3*290),'white');dd=ImageDraw.Draw(detail)
        for n,i in enumerate(ids[page*3:page*3+3]):
            r=rows[i];raw=validate_image(safe_path(ROOT/'data',r['raw_path']));ref=validate_image(safe_path(ROOT/'data',r['reference_path']))
            assert raw.source_sha256==r['raw_sha256'] and ref.source_sha256==r['reference_sha256']
            dst=OUT/'examples'/i;dst.mkdir(exist_ok=True)
            shutil.copyfile(safe_path(ROOT/'data',r['raw_path']),dst/'raw_original.png')
            images=[raw.image];hashes={'raw_original':r['raw_sha256']}
            for label,base in [('RGB',rgbbase),('MVP',depthbase)]:
                p=base/'images'/(i+'.png');assert file_hash(p)==indexes[label][i]['png_sha256']
                with Image.open(p) as im:images.append(im.convert('RGB'))
                shutil.copyfile(p,dst/(label.lower()+'_enhanced.png'));hashes[label]=file_hash(p)
            images.append(ref.image)
            ref.image.save(dst/'reference_display.png')
            x,y,w,h=rois[i]
            for col,(label,im) in enumerate(zip(['Raw input','RGB20 control','Depth MVP20','Curated reference'],images)):
                extra='' if col in [0,3] else f" | {indexes['RGB' if col==1 else 'MVP'][i]['psnr']:.2f} dB"
                wd.text((col*360+4,n*300+3),i+' '+label,font=font,fill='black');wd.text((col*360+4,n*300+25),extra.strip(' |'),font=font,fill='#475569')
                whole.paste(ImageOps.contain(im,(352,245)),(col*360+4,n*300+53))
                dd.text((col*280+4,n*290+3),i+' '+label,font=font,fill='black');detail.paste(im.crop((x,y,x+w,y+h)),(col*280+4,n*290+28))
            provenance.append({'id':i,'split':'diagnostic','raw_path':r['raw_path'],'hashes':hashes,'native_crop_xywh':rois[i],
                               'rgb_scores':{k:indexes['RGB'][i][k] for k in ['psnr','ssim']},'mvp_scores':{k:indexes['MVP'][i][k] for k in ['psnr','ssim']}})
        whole.save(F/f'qualitative_whole_{page+1}.png',dpi=(300,300));detail.save(F/f'qualitative_native_{page+1}.png',dpi=(300,300))
    (E/'qualitative_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    selection=json.loads((ROOT/'artifacts/phase2/evaluation_v3/selection.json').read_text())
    with (T/'rgb_artifact_review.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['checkpoint','minor','major','passed']);w.writeheader()
        for name,v in selection['screens'].items():w.writerow({'checkpoint':name,**{k:v[k] for k in ['minor','major','passed']}})
    for source in [ROOT/'artifacts/phase3/selective_review/report.json',ROOT/'artifacts/phase3/diagnostic_review/report.json']:
        shutil.copyfile(source,E/('selective_sensitivity.json' if 'selective' in str(source) else 'scalar_sensitivity.json'))
    print('Built 6 SVG charts/diagrams, 4 qualitative sheets, 6 native example sets, CSV tables and evidence copies.')


if __name__=='__main__':main()
