"""VRGaze -> Baballonia eye-model format, one .npz per subject.

    python training/prep/vrgaze.py data/vrgaze data/prepped/vrgaze [id216 id219 ...]   (default: every subject on disk)
    python training/prep/vrgaze.py data/vrgaze data/prepped/vrgaze_val --split val
    (fetch_vrgaze.py downloads + preps every subject in one go)

Per labelled pair (VRGaze gives ONE binocular gaze per left/right image pair):
  image  : 400x400 IR jpg -> grayscale -> 128x128 (INTER_AREA) -> cv2.equalizeHist, exactly what Baballonia does to
           an eye crop before stacking (ImageCollector.cs). VRGaze's right image is stored mirrored (_R_flipped); we
           keep it mirrored by default (both eyes then look the same way, as BabbleTrainer's per-side models expect
           nothing in particular) and record that in the npz, so it can be un-mirrored later.
  label  : BabbleTrainer's per-eye vector [pitch, yaw, lid, widen, squint, brow] (data.py ~L1017-1067):
           pitch/yaw = (degrees + 45) / 90 clamped to 0..1; lid 0.3 = normally open; widen/squint/brow 0.
           VRGaze labels blinks with a gaze too, so a frame with no visible pupil (pupil_blob() == 0) is labelled
           CLOSED instead: lid 1, gaze 0.5, as BabbleTrainer does for closed eyes.
  raw    : the original degrees, sequence and frame number, so sign conventions can be changed without re-prepping.

⚠️ Unverified until a Baballonia calibration is run: which sign is "up"/"right", and whether Baballonia's right
camera image matches VRGaze's mirrored right image. Both are cheap to flip later from the raw columns.

Output npz keys: left, right (uint8 [N,128,128]); label_left, label_right (float32 [N,6]); valid_left, valid_right
(bool [N]: False = eye state unsure, leave out of training); pupil_blob_left/right (int32 [N]); yaw_deg, pitch_deg
(float32 [N]); seq (str [N], "session/sub"); frame (int32 [N]); right_mirrored (bool).
"""
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SIZE = 128


OPEN_PX = 150      # pupil blob at least this big -> confidently open
# Between 0 and OPEN_PX it is ambiguous: lash clumps / lid shadows make 80-150 px blobs on closed eyes and small
# foreshortened pupils make them on open ones (checked visually on id216, 10-15), so those frames are excluded.


def pupil_blob(gray):
    """Area (px, at 400x400) of the largest near-black blob fully enclosed by the image: the pupil when the eye is
    open, none when the lid is closed. A 5x5 close first fills the LED glints inside the pupil (they can split it);
    a 9x9 open then removes thin lashes. Measured 10-15 on id216: open 373-810 px, closed 0, half-closed 494."""
    dark = (gray < 40).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    h, w = gray.shape
    enclosed = [s[4] for s in stats[1:] if s[4] > 60 and s[0] > 2 and s[1] > 2 and s[0] + s[2] < w - 2
                and s[1] + s[3] < h - 2]
    return max(enclosed, default=0)


def eye_state(blob):
    return "open" if blob >= OPEN_PX else "closed" if blob == 0 else "unsure"


def load_eye(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    blob = pupil_blob(img)
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(img), blob


def label(yaw_deg, pitch_deg, closed):
    if closed:                                   # BabbleTrainer: closed lid -> lid 1, gaze parked at 0.5
        return [0.5, 0.5, 1.0, 0.0, 0.0, 0.0]
    clamp = lambda v: min(max(v, 0.0), 1.0)
    return [clamp((pitch_deg + 45) / 90), clamp((yaw_deg + 45) / 90), 0.3, 0.0, 0.0, 0.0]


def read_pairs(src, split, wanted):
    pairs = defaultdict(list)
    with open(src / f"metadata_{split}.csv", newline="") as f:
        for r in csv.DictReader(f):
            subject = r["image_l"].split("/")[0]
            if subject in wanted:
                pairs[subject].append(r)
    return pairs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("subjects", nargs="*", help="default: every subject folder on disk for the split")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    a = ap.parse_args()
    prep_subjects(a.src, a.dst, a.split, set(a.subjects) or {p.name for p in (a.src / a.split).iterdir() if p.is_dir()})


def prep_subjects(src, dst, split, wanted, pairs=None):
    dst.mkdir(parents=True, exist_ok=True)
    pairs = pairs if pairs is not None else read_pairs(src, split, wanted)

    for subject in sorted(wanted):
        rows = pairs.get(subject, [])
        if not rows:
            print(f"{subject}: no labelled pairs found, skipped")
            continue
        left, right, lab_l, lab_r, yaw, pitch, seq, frame, missing = [], [], [], [], [], [], [], [], 0
        blob_l, blob_r = [], []

        def load_pair(r):
            try:
                return load_eye(src / split / r["image_l"]), load_eye(src / split / r["image_r"])
            except FileNotFoundError:
                return None

        # OpenCV releases the GIL while decoding, so threads give a near-linear speed-up (single-threaded prep
        # was ~5x slower than the download)
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as pool:
            loaded = list(pool.map(load_pair, rows, chunksize=64))
        for r, eyes in zip(rows, loaded):
            if eyes is None:
                missing += 1
                continue
            (l_img, l_blob), (r_img, r_blob) = eyes
            y, p = float(r["gaze_yaw_degrees"]), float(r["gaze_pitch_degrees"])
            left.append(l_img)
            right.append(r_img)
            lab_l.append(label(y, p, eye_state(l_blob) == "closed"))
            lab_r.append(label(y, p, eye_state(r_blob) == "closed"))
            blob_l.append(l_blob)
            blob_r.append(r_blob)
            yaw.append(y)
            pitch.append(p)
            parts = r["image_l"].split("/")
            seq.append(f"{parts[1]}/{parts[2]}")
            frame.append(int(parts[3].split("_")[0]))
        out = dst / f"{subject}.npz"
        blob_l, blob_r = np.asarray(blob_l, np.int32), np.asarray(blob_r, np.int32)
        # Per-eye closed detection is unreliable (10-15: a brighter image's pupil can sit above the fixed dark level;
        # a closed lid's shadow can pass as a pupil), but blinks are binocular. So gaze pre-training uses ONLY pairs
        # where BOTH eyes are confidently open; everything else is left out, and lids are learned in the per-user
        # fine-tune instead. valid_* is that pair rule, the same for both eyes.
        both_open = (blob_l >= OPEN_PX) & (blob_r >= OPEN_PX)
        valid_l = valid_r = both_open
        tmp = out.with_suffix(".tmp.npz")         # write then rename: a stopped run never leaves a broken .npz
        np.savez_compressed(tmp, left=np.stack(left), right=np.stack(right),
                            label_left=np.asarray(lab_l, np.float32), label_right=np.asarray(lab_r, np.float32),
                            valid_left=valid_l, valid_right=valid_r,
                            pupil_blob_left=blob_l, pupil_blob_right=blob_r,
                            yaw_deg=np.asarray(yaw, np.float32), pitch_deg=np.asarray(pitch, np.float32),
                            seq=np.asarray(seq), frame=np.asarray(frame, np.int32), right_mirrored=True)
        os.replace(tmp, out)
        pct = lambda m: f"{m.sum()} ({m.mean() * 100:.1f}%)"
        print(f"{subject}: {len(left)} pairs -> {out} ({out.stat().st_size / 1e6:.0f} MB), missing {missing} | "
              f"both eyes clearly open (used for gaze) {pct(both_open)} | left blob 0 {pct(blob_l == 0)}, "
              f"right blob 0 {pct(blob_r == 0)}")


if __name__ == "__main__":
    main()
