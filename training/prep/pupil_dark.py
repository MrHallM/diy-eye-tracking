"""Blink / closed-eye detector for VRGaze: an open eye shows a near-black pupil, a closed lid doesn't.
Computes the fraction of very dark pixels in the RAW (not equalized) eye image.

    python training/prep/pupil_dark.py data/vrgaze id216 --sheet out.png     report + sheet of the darkest/lightest

Used by vrgaze.py to label closed frames (BabbleTrainer: lid 1 = closed, gaze 0.5) instead of trusting the gaze label.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

DARK = 35          # 8-bit level counted as "pupil black" in the raw IR image


def dark_fraction(gray):
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    return float((small < DARK).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("subject")
    ap.add_argument("--sheet")
    a = ap.parse_args()
    names = []
    with open(Path(a.src) / "metadata_train.csv", newline="") as f:
        names = [r["image_l"] for r in csv.DictReader(f) if r["image_l"].startswith(a.subject + "/")]
    fr = []
    for n in names:
        img = cv2.imread(str(Path(a.src) / "train" / n), cv2.IMREAD_GRAYSCALE)
        fr.append(dark_fraction(img))
    fr = np.asarray(fr)
    print(f"{a.subject}: {len(fr)} left images; dark fraction percentiles "
          + ", ".join(f"p{p}={np.percentile(fr, p) * 100:.2f}%" for p in (1, 2, 5, 10, 25, 50, 90)))
    for t in (0.002, 0.005, 0.01, 0.02):
        print(f"  below {t * 100:.1f}%: {(fr < t).sum()} frames ({(fr < t).mean() * 100:.1f}%)")
    if a.sheet:
        order = np.argsort(fr)
        pick = list(order[:16]) + list(order[len(order) // 2:len(order) // 2 + 8])
        tiles = []
        for i in pick:
            img = cv2.resize(cv2.imread(str(Path(a.src) / "train" / names[i]), cv2.IMREAD_GRAYSCALE), (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(img, f"{fr[i] * 100:.2f}%", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(img)
        rows = [np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(3)]
        cv2.imwrite(a.sheet, np.vstack(rows))
        print(f"sheet: rows 1-2 = 16 darkest-poor frames (lowest dark fraction), row 3 = median frames -> {a.sheet}")


if __name__ == "__main__":
    main()
