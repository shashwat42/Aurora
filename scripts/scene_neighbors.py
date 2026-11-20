"""Pre-freeze administrative scene review using the pinned frozen DINOv2 backbone.
These descriptors are not enhancement scores and never fit to the dataset.
"""
from pathlib import Path
import csv,json,sys
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aurora.depth import DepthEstimator
from aurora.image import validate_image
OUT=ROOT/'artifacts/data_review/completion'
OUT.mkdir(parents=True,exist_ok=True)
records=[r for r in csv.DictReader((ROOT/'data/metadata/files.csv').open()) if r['category'] in {'raw','challenging'} and r['format'] in {'PNG','JPEG'} and r['frames']=='1']
model=DepthEstimator(ROOT/'configs/depth.json');model._load()
features=[]
for i,r in enumerate(records):
 valid=validate_image(ROOT/'data'/r['path'])
 with torch.inference_mode():
  inputs=model.processor(images=valid.image,size={'height':224,'width':224},return_tensors='pt')
  tokens=model.model.backbone(**inputs).feature_maps[-1]
  assert tokens.ndim==3 and tokens.shape[2]==384, tokens.shape
  f=tokens[0,1:,:].mean(dim=0)
  descriptor=torch.nn.functional.normalize(f,dim=0).cpu().numpy()
  assert descriptor.shape==(384,) and np.isfinite(descriptor).all()
  features.append(descriptor)
 if (i+1)%100==0: print(f'{i+1}/{len(records)}',flush=True)
f=np.array(features);assert f.shape==(len(records),384);np.save(OUT/'scene_features.npy',f,allow_pickle=False)
(OUT/'feature_inventory.json').write_text(json.dumps({'records':records,'revision':model.config['revision'],'method':'Frozen DINOv2 Small stage12 patch-token mean (class token excluded), L2 normalized; aspect-preserving 224-pixel processor; scene curation only'},indent=2)+'\n')
s=f@f.T;np.fill_diagonal(s,-1)
candidates=[]
for i in range(len(records)):
 for j in np.where(s[i,i+1:]>=0.93)[0]+i+1:
  candidates.append({'a':records[i]['category']+':'+records[i]['id'],'b':records[j]['category']+':'+records[j]['id'],'cosine':float(s[i,j]),'a_path':records[i]['path'],'b_path':records[j]['path']})
candidates.sort(key=lambda c:-c['cosine'])
(OUT/'feature_candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
# Include every paired image and its closest other image, including cross-set neighbors.
nearest=[]
for i,r in enumerate(records):
 j=int(s[i].argmax());nearest.append({'id':r['category']+':'+r['id'],'neighbor':records[j]['category']+':'+records[j]['id'],'cosine':float(s[i,j]),'path':r['path'],'neighbor_path':records[j]['path']})
(OUT/'nearest_neighbors.json').write_text(json.dumps(nearest,indent=2)+'\n')
def sheets(items,prefix,left,right,label,n=40):
 for start in range(0,len(items),n):
  sheet=Image.new('RGB',(1800,10*125),'#20252b');draw=ImageDraw.Draw(sheet)
  for k,r in enumerate(items[start:start+n]):
   x,y=(k%4)*450,(k//4)*125
   draw.text((x+3,y+2),label(r),fill='white')
   for col,key in enumerate([left,right]):
    with Image.open(ROOT/'data'/r[key]) as im:
     sheet.paste(ImageOps.contain(im.convert('RGB'),(220,102)),(x+col*225,y+22))
  sheet.save(OUT/f'{prefix}_{start//n+1:02d}.jpg',quality=92)
sheets(candidates,'feature_candidates','a_path','b_path',lambda r:f"{r['a']} | {r['b']} {r['cosine']:.3f}")
sheets(nearest,'nearest','path','neighbor_path',lambda r:f"{r['id']} | {r['neighbor']} {r['cosine']:.3f}")
print(json.dumps({'images':len(records),'candidates_above_093':len(candidates),'neighbor_sheets':(len(nearest)+39)//40}),flush=True)
