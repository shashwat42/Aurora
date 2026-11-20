from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/data_review/completion'
m=json.loads((ROOT/'data/metadata/manifest.draft.json').read_text())
parent={r['id']:r['scene_id'] for r in m['pairs']}; par={v:v for v in parent.values()}
def find(x):
 while par[x]!=x:
  par[x]=par[par[x]];x=par[x]
 return x
def scene(key):
 if key.startswith('challenging:'):return key
 i=key[4:]
 if i not in parent:
  i=next(r['canonical_id'] for r in m['excluded_pairs'] if r['id']==i)
 return find(parent[i])
c=json.loads((OUT/'feature_candidates.json').read_text())
for r in c:
 if r['a'].startswith('raw:') and r['b'].startswith('raw:'):
  a,b=scene(r['a']),scene(r['b']);par[a]=b
n=json.loads((OUT/'nearest_neighbors.json').read_text())
focus=[{'a':r['id'],'b':r['neighbor'],'a_path':r['path'],'b_path':r['neighbor_path'],'cosine':r['cosine']} for r in n if scene(r['id'])!=scene(r['neighbor'])]
cross=[r for r in c if r['a'].split(':')[0]!=r['b'].split(':')[0]]
(OUT/'focus_nearest.json').write_text(json.dumps(focus,indent=2)+'\n')
paths=sorted({r[k] for r in focus+cross for k in ['a_path','b_path']})
def thumb(p):
 with Image.open(ROOT/'data'/p) as im:return p,ImageOps.contain(im.convert('RGB'),(220,102))
with ThreadPoolExecutor(max_workers=4) as pool: thumbs=dict(pool.map(thumb,paths))
for prefix,items in [('focus',focus),('crossset',cross)]:
 for start in range(0,len(items),40):
  s=Image.new('RGB',(1800,1250),'#20252b');d=ImageDraw.Draw(s)
  for k,r in enumerate(items[start:start+40]):
   x,y=k%4*450,k//4*125
   d.text((x+3,y+2),f"{r['a']} | {r['b']} {r['cosine']:.3f}",fill='white')
   s.paste(thumbs[r['a_path']],(x,y+22));s.paste(thumbs[r['b_path']],(x+225,y+22))
  s.save(OUT/f'{prefix}_{start//40+1:02d}.jpg',quality=94)
print({'focus_images':len(focus),'crossset_candidates':len(cross)})
