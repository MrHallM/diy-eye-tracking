"""Download the Gaze-in-Wild eye videos with TEyeD's per-frame labels, as repackaged on Hugging Face
(`jpena-173/sm-eyes-v2`: 12 parquet shards, ~4.1 GB, every 3rd frame, 6 subjects, 640x480 IR).
Resumable: re-run to continue; finished shards are skipped by size.

    python training/prep/fetch_giw.py data/giw

Why: VRGaze has no lid labels (lids were left to the fine-tune). GIW+TEyeD has 34 eyelid landmarks, pupil and
iris ellipses, and per-frame blink/fixation/saccade labels. Licence: see DATASETS.md (private use only).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_vrgaze import fetch  # noqa: E402  (curl -C - with size check + retries)

import json  # noqa: E402
import urllib.request  # noqa: E402

REPO = "jpena-173/sm-eyes-v2"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"


def main():
    dst = Path(sys.argv[1])
    with urllib.request.urlopen(API, timeout=30) as r:
        files = [f for f in json.load(r) if f["type"] == "file"]
    for i, f in enumerate(sorted(files, key=lambda f: f["path"]), 1):
        fetch(f"{BASE}/{f['path']}", dst / f["path"], f["size"])
        print(f"[{i}/{len(files)}] {f['path']} ({f['size'] / 1e9:.2f} GB)", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
