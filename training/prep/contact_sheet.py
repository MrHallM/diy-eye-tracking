"""Grid of prepped eye images with their gaze labels, to eyeball a prep run.

    python training/prep/contact_sheet.py data/prepped/vrgaze/id216.npz out.png [--n 24]
"""
import argparse

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--state", choices=["valid", "excluded"], help="only pairs used for training / left out")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

d = np.load(a.npz)
pool = np.arange(len(d["left"]))
if a.state:
    pool = pool[d["valid_left"] if a.state == "valid" else ~d["valid_left"]]
idx = np.sort(np.random.default_rng(a.seed).choice(pool, min(a.n, len(pool)), replace=False))
tiles = []
for i in idx:
    pair = np.hstack([d["left"][i], d["right"][i]])
    pair = cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)
    shut = "".join(s for s, k in (("L", "label_left"), ("R", "label_right")) if d[k][i, 2] == 1)
    cv2.putText(pair, f"y{d['yaw_deg'][i]:+.0f} p{d['pitch_deg'][i]:+.0f}" + (f" CLOSED {shut}" if shut else ""),
                (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if shut else (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(pair)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
cols = 4
rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)]
cv2.imwrite(a.out, np.vstack(rows))
print(f"wrote {a.out}: {len(tiles)} left|right pairs")
