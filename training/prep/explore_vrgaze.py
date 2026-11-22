"""Describe downloaded VRGaze subjects: folder layout, image sizes, label coverage, frame spacing per sequence.

    python training/prep/explore_vrgaze.py data/vrgaze
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1])
subjects = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
print("subjects on disk:", subjects)

rows = defaultdict(list)                    # (subject, session, sub) -> [(frame, yaw, pitch)]
with open(root / "metadata_train.csv", newline="") as f:
    for r in csv.DictReader(f):
        subj, sess, sub, name = r["image_l"].split("/")
        if subj in subjects:
            rows[(subj, sess, sub)].append((int(name.split("_")[0]), float(r["gaze_yaw_degrees"]),
                                            float(r["gaze_pitch_degrees"])))

for subj in subjects:
    seqs = {k: v for k, v in rows.items() if k[0] == subj}
    frames = sum(len(v) for v in seqs.values())
    zero = sum(1 for v in seqs.values() for _, y, p in v if y == 0 and p == 0)
    print(f"\n{subj}: {len(seqs)} sequences, {frames} labelled pairs, {zero} with gaze exactly (0,0)")
    for k, v in sorted(seqs.items(), key=lambda kv: (int(kv[0][1]), int(kv[0][2])))[:12]:
        v.sort()
        steps = [b[0] - a[0] for a, b in zip(v, v[1:])]
        yaws = [y for _, y, _ in v]
        pitches = [p for _, _, p in v]
        print(f"  seq {k[1]}/{k[2]}: {len(v):5d} pairs, frame# {v[0][0]}..{v[-1][0]}, step median "
              f"{statistics.median(steps) if steps else '-'}, yaw {min(yaws):+.1f}..{max(yaws):+.1f}, "
              f"pitch {min(pitches):+.1f}..{max(pitches):+.1f}")

sample = next((root / "train" / subjects[0]).rglob("*_L.jpg"))
im = Image.open(sample)
print(f"\nsample {sample.relative_to(root)}: {im.size[0]}x{im.size[1]} {im.mode}")
on_disk = {p.relative_to(root / 'train').as_posix() for p in (root / "train").rglob("*.jpg")}
listed = {f"{k[0]}/{k[1]}/{k[2]}/{fr}_L.jpg" for k, v in rows.items() for fr, _, _ in v}
print(f"left images on disk {sum(1 for p in on_disk if p.endswith('_L.jpg'))}, labelled {len(listed)}, "
      f"labelled+present {len(listed & on_disk)}")
