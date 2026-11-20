"""Verify and materialize the internal paper integration deliverable."""
import hashlib,json,zipfile
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT.parent/'Docs/Integration'
ASSETS=ROOT/'artifacts/paper_integration'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
provenance=json.loads((ASSETS/'evidence/qualitative_provenance.json').read_text())
for row in provenance:
    assert row['split']=='diagnostic'
    case=ASSETS/'examples'/row['id']
    for name,key in [('raw_original.png','raw_original'),('rgb_enhanced.png','RGB'),('mvp_enhanced.png','MVP')]:
        assert digest(case/name)==row['hashes'][key],(row['id'],name)
    with Image.open(case/'raw_original.png') as raw,Image.open(case/'mvp_enhanced.png') as out:
        assert raw.size==out.size
for p in (DOCS/'source_papers').glob('*.pdf'):
    assert digest(p)==digest(Path('/home/shashwatsingh/Downloads')/p.name)
files=[]
for p in sorted(DOCS.rglob('*')):
    if p.is_file():files.append((p,'Integration/'+p.relative_to(DOCS).as_posix()))
manifest_path=ASSETS/'evidence/package_manifest.json'
for p in sorted(ASSETS.rglob('*')):
    if p.is_file() and p!=manifest_path:
        assert 'screenshot' not in p.as_posix().lower()
        files.append((p,'Integration/assets/'+p.relative_to(ASSETS).as_posix()))
assert len(files)==len({name for _,name in files})
manifest={'schema_version':1,'status':'verified','input_output_pairs':len(provenance),'ui_screenshots_included':False,'scope':'Internal integration supplement; diagnostic development evidence only','files':[{'path':name,'bytes':p.stat().st_size,'sha256':digest(p)} for p,name in files]}
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
files.append((manifest_path,'Integration/assets/evidence/package_manifest.json'))
archive=ROOT/'artifacts/paper_integration_bundle.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p,name in files:z.write(p,name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)
print(json.dumps({'status':'verified','native_input_output_pairs':len(provenance),'files':len(files),'archive_bytes':archive.stat().st_size,'archive_sha256':digest(archive)},indent=2))
