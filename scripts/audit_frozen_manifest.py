"""Verify frozen data/review integrity without model inference or test scoring."""
from pathlib import Path
from collections import Counter
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aurora.manifest import read_manifest, file_hash, safe_path, development_rows

path = ROOT / "data/metadata/manifest.v1.json"
manifest = read_manifest(path)
assert file_hash(safe_path(ROOT / "data", manifest["review_policy_path"])) == manifest["review_policy_sha256"]
assert file_hash(safe_path(ROOT / "data", manifest["scene_review_path"])) == manifest["scene_review_sha256"]
assert file_hash(safe_path(ROOT / "data", manifest["review_evidence_path"])) == manifest["review_evidence_sha256"]
policy = json.loads(safe_path(ROOT / "data", manifest["review_policy_path"]).read_text())
assert policy["status"] == "approved"
assert file_hash(safe_path(ROOT / "data", manifest["completion_evidence_path"])) == manifest["completion_evidence_sha256"]
completion = json.loads(safe_path(ROOT / "data", manifest["completion_evidence_path"]).read_text())
for relative, expected in completion["evidence_hashes"].items():
    assert file_hash(safe_path(ROOT, relative)) == expected, relative
queue = list(csv.DictReader(safe_path(ROOT / "data", manifest["scene_review_path"]).open()))
assert all(r["decision"] == "accepted" and r["reviewer"].strip() for r in queue)
assert len(queue) == manifest["scene_clusters"]
queue_members = {r["scene_id"]: set(r["members"].split()) for r in queue}
counts = Counter()
for row in manifest["pairs"]:
    assert row["id"] in queue_members[row["scene_id"]]
    for kind in ["raw", "reference"]:
        assert file_hash(safe_path(ROOT / "data", row[kind + "_path"])) == row[kind + "_sha256"]
    assert min(row["width"], row["height"]) >= 11
    counts[row["split"]] += 1
for row in manifest["challenging"]:
    assert file_hash(safe_path(ROOT / "data", row["raw_path"])) == row["raw_sha256"]
counts["challenging"] = len(manifest["challenging"])
assert dict(counts) == manifest["counts"]
assert len(manifest["pairs"]) + len(manifest["excluded_pairs"]) == 890
assert len(manifest["challenging"]) + len(manifest["excluded_challenging"]) == 60
assert set().union(*queue_members.values()) == {r["id"] for r in manifest["pairs"]}
for split in ["test", "challenging"]:
    try:
        development_rows(manifest, split)
    except ValueError:
        pass
    else:
        raise AssertionError("Holdout access was allowed through the development API")
out = ROOT / "data/metadata/splits/v1"
out.mkdir(parents=True, exist_ok=True)
for split in ["train", "diagnostic", "selection", "test", "challenging"]:
    rows = manifest["challenging"] if split == "challenging" else [r for r in manifest["pairs"] if r["split"] == split]
    (out / (split + ".json")).write_text(json.dumps({"manifest_sha256": file_hash(path), "split": split,
                                                   "images": rows}, indent=2) + "\n")
report = {"status": "passed", "manifest_sha256": file_hash(path), "counts": dict(counts),
          "reviewed_scene_clusters": len(queue), "review_policy_and_evidence_hashes": "matched",
          "all_included_source_file_hashes": "matched", "holdout_development_api_access": "rejected",
          "previously_inspected_ids_in_selection_or_test": 0,
          "note": "Visual scene-group disjointness is verified against the accepted annotations. Original capture-site/dive identifiers are not available, so camera/dive/domain independence is not claimed."}
(ROOT / "artifacts/data_review/frozen_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
