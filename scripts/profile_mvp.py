"""User-run full worker profiling: 3 warmups + 30 operations per native size."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aurora.app import default_bundle
from aurora.image import validate_image
from aurora.manifest import file_hash,read_manifest,safe_path
from aurora.worker import WorkerManager


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='artifacts/phase3/mvp/profile')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    bundle,approved=default_bundle();meta=json.loads(Path(bundle).read_text())
    if not meta['configuration'].get('depth_config'):raise ValueError('Active MVP must be depth-guided')
    manifest=read_manifest(meta['configuration']['manifest'])
    rows={r['id']:r for r in manifest['pairs'] if r['split']=='diagnostic'}
    ids=['284_img_','14_img_','734_img_']
    cpu=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    cpu_name=next((s.split(':',1)[1].strip() for s in cpu.splitlines() if s.startswith('model name')),platform.processor())
    identity={'bundle_sha256':file_hash(bundle),'weights_sha256':meta['weights_sha256'],
              'manifest_sha256':meta['configuration']['manifest_sha256'],
              'depth_fingerprint':meta['configuration']['depth_fingerprint'],
              'cpu':cpu_name,'platform':platform.platform(),'python':sys.version,
              'protocol':'3 excluded warmup operations then30 recorded operations per native diagnostic image; fresh worker/models and live depth each operation; no upload/network/browser latency',
              'cold_start_scope':'Every worker loads generator and estimator anew; warmups warm OS caches, not a persistent model.',
              'input_max_bytes':20*1024*1024,'input_max_pixels':2500000}
    (out/'protocol.json').write_text(json.dumps(identity,indent=2)+'\n')
    results=[];manager=WorkerManager(bundle,root=out/'jobs')
    try:
        with (out/'operations.jsonl').open('x') as stream:
            for identifier in ids:
                row=rows[identifier];source=safe_path(meta['configuration']['data_root'],row['raw_path'])
                assert file_hash(source)==row['raw_sha256']
                for index in range(33):
                    start=time.perf_counter();session=manager.upload('profile',source)
                    if session.valid is None:raise ValueError(session.message)
                    manager.start('profile',session.token);deadline=time.monotonic()+300
                    while time.monotonic()<deadline:
                        status=manager.poll('profile',session.token)
                        if status['state']=='failed':raise RuntimeError(status['message'])
                        if status['state']=='success':break
                        time.sleep(.05)
                    else:raise TimeoutError('Worker exceeded300seconds')
                    elapsed=time.perf_counter()-start
                    from PIL import Image
                    with Image.open(status['output']) as image:
                        valid=validate_image(source)
                        assert image.size==valid.image.size and image.info.get('icc_profile') and not image.getexif()
                    assert status['report']['depth_mode']=='live'
                    record={'id':identifier,'operation':index,'warmup':index<3,
                            'pipeline_seconds':elapsed,**status['report']}
                    stream.write(json.dumps(record)+'\n');stream.flush()
                    if index>=3:results.append(record)
                    manager.remove('profile')
                    print(f'{identifier} operation {index+1}/33 complete',flush=True)
        summary=[]
        for identifier in ids:
            records=[r for r in results if r['id']==identifier]
            summary.append({'id':identifier,'width':records[0]['width'],'height':records[0]['height'],'count':len(records),
                            'median_seconds':float(np.median([r['pipeline_seconds'] for r in records])),
                            'p95_seconds':float(np.percentile([r['pipeline_seconds'] for r in records],95)),
                            'max_worker_rss_bytes':max(r['worker_peak_rss_bytes'] for r in records)})
        report={'status':'complete','identity':identity,'summary':summary,'recorded_operations':90,
                'warmup_operations':9,'test_images_used':0,'selection_images_used':0,'challenging_images_used':0,
                'limitations':'Three diagnostic images, not a size-class worst-case guarantee; excludes browser/network/upload transport. No quality or depth-benefit claim.'}
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally:manager.close()


if __name__=='__main__':main()
