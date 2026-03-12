#!/usr/bin/env python3
"""
Synchronize nnunet_id_map.json split labels to the active splits.json file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_JSON = DATA_DIR / "splits.json"
ID_MAP_JSON = DATA_DIR / "nnunet_id_map.json"
SUMMARY_TXT = DATA_DIR / "split_metadata_sync.txt"


def main() -> None:
    splits = json.loads(SPLITS_JSON.read_text())
    id_map = json.loads(ID_MAP_JSON.read_text())

    expected = {}
    for split in ("train", "val", "test"):
        for row in splits[split]:
            expected[row["filename"]] = split

    changed = 0
    missing = []
    for case_id, meta in id_map.items():
        filename = meta.get("filename")
        if filename not in expected:
            missing.append((case_id, filename))
            continue
        new_split = expected[filename]
        if meta.get("split") != new_split:
            meta["split"] = new_split
            changed += 1

    ID_MAP_JSON.write_text(json.dumps(id_map, indent=2))

    split_counts = Counter(meta["split"] for meta in id_map.values())
    lines = [
        "Split metadata sync complete",
        f"splits_json={SPLITS_JSON}",
        f"id_map_json={ID_MAP_JSON}",
        f"changed_entries={changed}",
        f"missing_entries={len(missing)}",
        f"final_split_counts={dict(split_counts)}",
    ]
    if missing:
        lines.append("sample_missing_entries:")
        lines.extend(f"  - {case_id}: {filename}" for case_id, filename in missing[:10])

    SUMMARY_TXT.write_text("\n".join(lines) + "\n")
    print(SUMMARY_TXT.read_text())


if __name__ == "__main__":
    main()
