"""Audit complete masked review files and immutable V3 evidence; no ratings."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aurora.manifest import file_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    evaluation = ROOT / 'artifacts/phase2/evaluation_v3'
    snapshot_path = ROOT / 'artifacts/phase2/diagnosis_v3/selection_verification/report.json'
    snapshot = json.loads(snapshot_path.read_text())
    require(file_hash(evaluation / 'ledger.json') == snapshot['ledger_sha256'], 'Ledger changed after verification')
    for relative, digest in snapshot['review_packet_hashes'].items():
        require(file_hash(ROOT / relative) == digest, 'Masked image packet changed')
    ledger = json.loads((evaluation / 'ledger.json').read_text())
    people = ['Codex V3 AI reviewer 1', 'Codex V3 AI reviewer 2']
    reports, csv_hashes = {}, {}
    for candidate in ledger['candidates']:
        path = evaluation / candidate['name'] / 'selection/summary.json'
        require(file_hash(path) == candidate['selection_summary_sha256'], 'Selection metrics changed')
        require(file_hash(candidate['bundle']) == candidate['bundle_sha256'], 'Candidate bundle changed')
        ids = {r['id'] for r in json.loads(path.read_text())['records']}
        ratings = []
        counts = {}
        for number, person in enumerate(people, 1):
            csv_path = evaluation / 'review' / candidate['label'] / f'reviewer-{number}.csv'
            with csv_path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                require(reader.fieldnames == ['id', 'reviewer', 'independent', 'severity', 'notes'], 'CSV schema')
                rows = list(reader)
            require(len(rows) == len(ids) == 40 and {r['id'] for r in rows} == ids, 'Review coverage/duplicate IDs')
            require(all(r['reviewer'] == person and r['independent'] == 'yes'
                        and r['severity'] in ['none', 'minor', 'major'] and r['notes'].strip() for r in rows), 'Complete named procedural review and case notes required')
            ratings.append({r['id']: r['severity'] for r in rows})
            counts[person] = dict(Counter(r['severity'] for r in rows))
            csv_hashes[str(csv_path.relative_to(ROOT))] = file_hash(csv_path)
        order = {'none': 0, 'minor': 1, 'major': 2}
        combined = {i: max((r[i] for r in ratings), key=order.get) for i in ids}
        totals = Counter(combined.values())
        reports[candidate['name']] = {'label': candidate['label'], 'reviewer_counts': counts,
                                     'combined_counts': dict(totals),
                                     'passed': totals['major'] == 0 and totals['minor'] <= 4,
                                     'disagreement_count': sum(ratings[0][i] != ratings[1][i] for i in ids),
                                     'combined_severity': combined}
    report = {'status': 'complete_reviews_verified', 'reviewer_ids': people,
              'selection_verification_sha256': file_hash(snapshot_path), 'review_csv_sha256': csv_hashes,
              'candidates': reports, 'rating_count': 240, 'human_review': False,
              'independence_scope': 'Two blinded AI procedures; no statistical independence or human certification claimed',
              'review_script_sha256': file_hash(Path(__file__)),
              'model_forwards': 0, 'training_steps': 0, 'test_images_used': 0, 'challenging_images_used': 0}
    out = ROOT / 'artifacts/phase2/diagnosis_v3/artifact_review_verification'; out.mkdir(exist_ok=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['review_csv_sha256', 'candidates']}, indent=2))
    print(json.dumps({name: {k: v for k, v in record.items() if k != 'combined_severity'}
                      for name, record in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
