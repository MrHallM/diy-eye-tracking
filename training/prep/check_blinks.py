"""Do VRGaze's pupil/glint features mark blinks? Print them for evenly spaced frames of one prepped subject.

    python training/prep/check_blinks.py data/vrgaze data/prepped/vrgaze/id216.npz id216 [--n 24]
"""
import argparse
import csv

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("npz")
ap.add_argument("subject")
ap.add_argument("--n", type=int, default=24)
a = ap.parse_args()

d = np.load(a.npz)
idx = np.linspace(0, len(d["left"]) - 1, a.n).astype(int)
want = {f"{a.subject}/{d['seq'][i]}/{d['frame'][i]}_L.jpg": i for i in idx}
feats = {}
with open(f"{a.src}/metadata_train_features.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["image"] in want:
            feats[r["image"]] = r
for name, i in want.items():
    r = feats.get(name)
    vis = [float(v) for v in r["glints_visibility"].strip("()").split(",")] if r else []
    print(f"#{i:5d} y{d['yaw_deg'][i]:+5.1f} p{d['pitch_deg'][i]:+5.1f} mean {d['left'][i].mean():5.1f} "
          f"pupil {r['pupils'] if r else 'NO FEATURE ROW':>16}  glints visible {sum(v > 0.5 for v in vis)}/{len(vis)}")
