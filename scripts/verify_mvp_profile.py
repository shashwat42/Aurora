"""Recompute recorded profile aggregates without repeating inference."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aurora.app import default_bundle
from aurora.manifest import file_hash,read_manifest


def main():
    directory=ROOT/'artifacts/phase3/mvp/profile'
    summary=json.loads((directory/'summary.json').read_text())
    protocol=json.loads((directory/'protocol.json').read_text())
    records=[json.loads(line) for line in (directory/'operations.jsonl').read_text().splitlines()]
    assert summary['status']=='complete' and summary['identity']==protocol
    assert len(records)==99 and sum(r['warmup'] for r in records)==9
    assert summary['recorded_operations']==90 and summary['warmup_operations']==9
    bundle,approved=default_bundle();b=json.loads(Path(bundle).read_text())
    assert not approved and file_hash(bundle)==protocol['bundle_sha256']
    assert file_hash(Path(bundle).parent/b['weights'])==protocol['weights_sha256']==b['weights_sha256']
    assert file_hash(ROOT/b['configuration']['manifest'])==protocol['manifest_sha256']
    rows={r['id']:r for r in read_manifest(ROOT/b['configuration']['manifest'])['pairs'] if r['split']=='diagnostic'}
    assert {r['id'] for r in records}=={'284_img_','14_img_','734_img_'}
    computed=[]
    for aggregate in summary['summary']:
        identifier=aggregate['id'];group=[r for r in records if r['id']==identifier]
        assert [r['operation'] for r in group]==list(range(33))
        for record in group:
            assert record['warmup']==(record['operation']<3)
            assert record['depth_mode']=='live' and record['weights_sha256']==protocol['weights_sha256']
            assert record['depth_fingerprint']==protocol['depth_fingerprint']==b['configuration']['depth_fingerprint']
            assert record['source_sha256']==rows[identifier]['raw_sha256']
            assert all(record[k]==rows[identifier][k] for k in ['width','height'])
            assert all(np.isfinite(record[k]) and record[k]>0 for k in ['pipeline_seconds','worker_seconds','model_load_seconds','worker_peak_rss_bytes'])
        measured=[r for r in group if not r['warmup']];assert len(measured)==aggregate['count']==30
        times=[r['pipeline_seconds'] for r in measured]
        calculated={'id':identifier,'width':measured[0]['width'],'height':measured[0]['height'],'count':30,
                    'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),
                    'max_worker_rss_bytes':max(r['worker_peak_rss_bytes'] for r in measured)}
        assert calculated==aggregate
        computed.append(calculated)
    assert len(computed)==3
    assert all(summary[k]==0 for k in ['test_images_used','selection_images_used','challenging_images_used'])
    out=ROOT/'artifacts/phase3/mvp/profile_verification';out.mkdir(exist_ok=False)
    report={'status':'verified_99_operations_90_measured','profile_summary_sha256':file_hash(directory/'summary.json'),
            'protocol_sha256':file_hash(directory/'protocol.json'),'records_sha256':file_hash(directory/'operations.jsonl'),
            'verification_script_sha256':file_hash(Path(__file__)),'identity':protocol,'recomputed_summary':computed,
            'percentile_method':'NumPy percentile default linear interpolation; per-image distribution over30 operations',
            'benchmark_rerun':False,'model_forwards':0,'quality_acceptance_claimed':False,
            'scope':'Measured cases/hardware only. Does not establish worst-case bounds, original resource-contract acceptance, browser latency or depth benefit.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'paper_profile.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(computed[0]));writer.writeheader();writer.writerows(computed)
    previous=json.loads((ROOT/'artifacts/phase3/mvp/paper_evidence.json').read_text())
    previous.update(profile_status='Complete and independently recomputed from recorded operations',profile=report)
    (out/'paper_evidence_profiled.json').write_text(json.dumps(previous,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='identity'},indent=2))


if __name__=='__main__':main()
