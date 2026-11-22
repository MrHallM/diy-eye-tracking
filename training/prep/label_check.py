"""Are VRGaze's gaze labels the eye's real direction, or the target's? If they're real, the pupil position in the
image (from VRGaze's own features file) should track them closely. Compares the full label file with the
static-points file (fixations only).

    python training/prep/label_check.py data/vrgaze id216
"""
import ast
import csv
import sys
from pathlib import Path

import numpy as np

src, subject = Path(sys.argv[1]), sys.argv[2]
pupils = {}
with open(src / "metadata_train_features.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["image"].startswith(subject + "/") and r["image"].endswith("_L.jpg"):
            pupils[r["image"]] = ast.literal_eval(r["pupils"])


def check(name, path):
    xs, ys, yaw, pitch = [], [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            p = pupils.get(r["image_l"].replace("_fliped", "_flipped"))
            if r["image_l"].startswith(subject + "/") and p:
                xs.append(p[0]); ys.append(p[1])
                yaw.append(float(r["gaze_yaw_degrees"])); pitch.append(float(r["gaze_pitch_degrees"]))
    if len(xs) < 10:
        print(f"{name}: only {len(xs)} rows for {subject}")
        return
    xs, ys, yaw, pitch = map(np.asarray, (xs, ys, yaw, pitch))
    # best linear fit pupil (x, y) -> label, and how much of the label it explains
    A = np.c_[xs, ys, np.ones_like(xs)]
    for lab, nm in ((yaw, "yaw"), (pitch, "pitch")):
        coef, *_ = np.linalg.lstsq(A, lab, rcond=None)
        resid = lab - A @ coef
        r2 = 1 - resid.var() / lab.var()
        print(f"{name:14s} {nm:5s}: n={len(lab):5d}  pupil position explains R²={r2:.2f} of the label, "
              f"residual {np.abs(resid).mean():.1f}° (label spread {lab.std():.1f}°)")


check("all labels", src / "metadata_train.csv")
check("static points", src / "metadata_train_static_points.csv")
