"""Build auditable candidate splits; freeze only after explicit scene-review acceptance.

The default emits a draft. It never overwrites a frozen manifest.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import sys
import shutil
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aurora.manifest import file_hash, read_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="Requires approved scene review in configs/data_review.json")
    args = parser.parse_args()
    target = ROOT / "data/metadata" / ("manifest.v1.json" if args.freeze else "manifest.draft.json")
    if args.freeze and (target.exists() or target.with_suffix(".json.sha256").exists()):
        parser.error("Refusing to overwrite a frozen manifest or checksum; use a versioned protocol change.")
    policy_path = ROOT / "configs/data_review.json"
    policy = json.loads(policy_path.read_text())
    if args.freeze and policy["status"] != "approved":
        parser.error("Scene review is not approved; preserve the draft and complete the documented review first.")
    if args.freeze:
        completion = ROOT / policy["completion_evidence_path"]
        if file_hash(completion) != policy["completion_evidence_sha256"] or json.loads(completion.read_text())["status"] != "accepted":
            parser.error("Scene completion evidence is missing, changed or unaccepted.")
    review_path = ROOT / "artifacts/data_review/review.json"
    review = json.loads(review_path.read_text())
    original = json.loads((ROOT / "data/metadata/uieb_audit.json").read_text())
    excluded = {}
    for group in original["identical_rgb_groups"]["raw"]:
        ids = sorted((Path(n).stem for n in group), key=lambda s: int(s.split("_")[0]))
        for key in ids[1:]:
            excluded[key] = {"id": key, "reason": "Exact decoded-RGB duplicate input", "canonical_id": ids[0]}
    rows = {r["id"]: r for r in review["pairs"] if r["id"] not in excluded}
    protected = set(policy.get("protected_development_ids", []))
    parent = {key: key for key in rows}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(keys):
        keys = [k for k in keys if k in rows]
        for key in keys[1:]:
            a, b = find(keys[0]), find(key)
            parent[max(a, b)] = min(a, b)
    for candidate in review["candidates"]:
        union([candidate["a"], candidate["b"]])
    if policy.get("feature_grouping"):
        evidence = ROOT / policy["feature_grouping"]["candidates_path"]
        if file_hash(evidence) != policy["feature_grouping"]["candidates_sha256"]:
            raise ValueError("Feature review evidence changed.")
        for candidate in json.loads(evidence.read_text()):
            if candidate["cosine"] >= policy["feature_grouping"]["threshold"] and candidate["a"].startswith("raw:") and candidate["b"].startswith("raw:"):
                union([candidate["a"][4:], candidate["b"][4:]])
    for group in policy["visual_img_ranges"]:
        union([key for key in rows if key.endswith("_img_") and group["start"] <= int(key.split("_")[0]) <= group["end"]])
    missing_annotations = []
    for group in policy["visual_id_groups"]:
        missing_annotations += [key for key in group["ids"] if key not in rows and key not in excluded]
        union(group["ids"])
    clusters = defaultdict(list)
    for key in sorted(rows):
        clusters[find(key)].append(key)
    queue_path = ROOT / "data/metadata/scene_review.csv"
    prior = {}
    if queue_path.exists():
        with queue_path.open() as stream:
            prior = {r["scene_id"]: r for r in csv.DictReader(stream)}
    queue = []
    for key, members in sorted(clusters.items()):
        digest = hashlib.sha256(json.dumps(sorted(members)).encode()).hexdigest()
        previous = prior.get("scene_" + key, {})
        unchanged = previous.get("members_sha256") == digest
        queue.append({"scene_id": "scene_" + key, "members": " ".join(members), "member_count": len(members),
                      "members_sha256": digest, "decision": previous.get("decision", "pending") if unchanged else "pending",
                      "reviewer": previous.get("reviewer", "") if unchanged else "",
                      "notes": previous.get("notes", "") if unchanged else ""})
    if args.freeze and any(r["decision"] != "accepted" or not r["reviewer"].strip() for r in queue):
        parser.error("Every current scene cluster needs an accepted decision and reviewer in data/metadata/scene_review.csv before freezing.")
    with queue_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)
    # Targets are proposed membership counts, not benchmark-standard splits.
    rng = random.Random(20261101)
    groups = sorted(clusters.values())
    rng.shuffle(groups)
    membership = {}
    for split, target in [("test", 90), ("diagnostic", 40), ("selection", 40)]:
        # Subset sum selects whole scene groups and exactly reaches the target.
        possible = {0: []}
        for i, group in enumerate(groups):
            if split in {"test", "selection"} and protected.intersection(group):
                continue
            for size, selected in list(possible.items()):
                new_size = size + len(group)
                if new_size <= target and new_size not in possible:
                    possible[new_size] = selected + [i]
            if target in possible:
                break
        if target not in possible:
            raise ValueError(f"Cannot assign {target} {split} items without splitting a scene.")
        chosen = set(possible[target])
        for i in chosen:
            for key in groups[i]:
                membership[key] = split
        groups = [g for i, g in enumerate(groups) if i not in chosen]
    for group in groups:
        for key in group:
            membership[key] = "train"
    pairs = []
    inventory = list(csv.DictReader((ROOT / "data/metadata/files.csv").open()))
    raw_inventory = {r["id"]: r for r in inventory if r["category"] == "raw"}
    for key, row in sorted(rows.items()):
        pairs.append({k: row[k] for k in ["id", "raw_path", "reference_path", "raw_sha256", "reference_sha256", "width", "height"]} |
                     {"scene_id": "scene_" + find(key), "split": membership[key],
                      "raw_rgb_sha256": raw_inventory[key]["rgb_sha256"]})
    challenging, challenging_exclusions = [], []
    quarantine = {r["id"]: r for r in policy.get("challenging_quarantine", [])}
    challenging_scene = {}
    for index, group in enumerate(policy.get("challenging_scene_groups", [])):
        for key in group["ids"]:
            challenging_scene[key] = "challenging_group_" + str(index + 1)
    for row in inventory:
        if row["category"] != "challenging":
            continue
        if row["format"] not in {"PNG", "JPEG"} or int(row["frames"]) != 1 or row["mode"] != "RGB":
            challenging_exclusions.append({"id": row["id"], "path": row["path"], "reason": "Unsupported multiframe MPO; preserved unchanged and quarantined"})
        elif row["id"] in quarantine:
            challenging_exclusions.append({"id": row["id"], "path": row["path"], **quarantine[row["id"]]})
        else:
            challenging.append({"id": row["id"], "raw_path": row["path"], "raw_sha256": row["sha256"],
                                "raw_rgb_sha256": row["rgb_sha256"],
                                "scene_id": challenging_scene.get(row["id"], "challenging_" + row["id"]), "split": "challenging"})
    document = {"version": 1, "status": "frozen" if args.freeze else "draft_pending_scene_review",
                "seed": 20261101, "split_unit": "reviewed scene cluster", "pairs": pairs, "challenging": challenging,
                "excluded_pairs": list(excluded.values()), "excluded_challenging": challenging_exclusions,
                "counts": dict(Counter(r["split"] for r in pairs)) | {"challenging": len(challenging)},
                "scene_clusters": len(clusters), "review_policy_sha256": file_hash(policy_path),
                "scene_review_sha256": file_hash(queue_path),
                "protected_development_ids": sorted(protected),
                "review_evidence_sha256": file_hash(review_path),
                "pending": ["Review all provisional singleton scenes and conservative cluster memberships", "Compare paired/challenging source scenes without using them for model tuning", "Confirm pair alignment beyond contact-sheet scale for suspicious cases", "Record reviewer acceptance before freezing"],
                "missing_annotation_ids": sorted(set(missing_annotations)),
                "holdout_policy": "Only diagnostic images may guide architecture/preprocessing debugging. Selection is checkpoint-only. Final tests and challenging images remain outside tuning. Draft assignments are not a research freeze."}
    if args.freeze:
        document["pending"] = []
        snapshot = ROOT / "data/metadata/reviews/v1"
        snapshot.mkdir(parents=True, exist_ok=False)
        for source, name in [(policy_path, "data_review.json"), (queue_path, "scene_review.csv"),
                             (review_path, "source_review.json"),
                             (ROOT / policy["completion_evidence_path"], "completion_review.json")]:
            shutil.copyfile(source, snapshot / name)
            (snapshot / name).chmod(0o444)
        document.update(review_policy_path="metadata/reviews/v1/data_review.json",
                        scene_review_path="metadata/reviews/v1/scene_review.csv",
                        review_evidence_path="metadata/reviews/v1/source_review.json",
                        completion_evidence_path="metadata/reviews/v1/completion_review.json",
                        completion_evidence_sha256=policy["completion_evidence_sha256"])
    target = ROOT / "data/metadata" / ("manifest.v1.json" if args.freeze else "manifest.draft.json")
    if target.exists() and json.loads(target.read_text())["status"] == "frozen":
        raise ValueError("Refusing to overwrite a frozen manifest; use a versioned protocol change.")
    target.write_text(json.dumps(document, indent=2) + "\n")
    if args.freeze:
        target.with_suffix(".json.sha256").write_text(file_hash(target) + "  " + target.name + "\n")
        target.chmod(0o444)
        target.with_suffix(".json.sha256").chmod(0o444)
    read_manifest(target, require_frozen=args.freeze)
    (ROOT / "artifacts/data_review/manifest_summary.json").write_text(json.dumps({k: v for k, v in document.items() if k not in {"pairs", "challenging"}}, indent=2) + "\n")
    print(json.dumps({"manifest": str(target.relative_to(ROOT)), "status": document["status"], "counts": document["counts"], "excluded_pairs": len(excluded), "excluded_challenging": len(challenging_exclusions), "scene_clusters": len(clusters), "missing_annotation_ids": document["missing_annotation_ids"]}, indent=2))


if __name__ == "__main__":
    main()
