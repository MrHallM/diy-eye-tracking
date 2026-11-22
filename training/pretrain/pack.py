"""Pack prepped .npz subjects into memory-mapped arrays for training (the full set, ~1.8 M eye images at 128x128,
doesn't fit comfortably in RAM next to everything else).

    python training/pretrain/pack.py data/prepped/vrgaze data/packed/vrgaze_train                    pseudo labels
    python training/pretrain/pack.py data/prepped/vrgaze_val data/packed/vrgaze_val_static --labels static

--labels pseudo (default): the per-session pupil/glint mapping from prep/pseudolabel.py (needs <id>.pseudo.npz);
         static: only VRGaze's clean static points (for scoring); main: VRGaze's raw labels (mostly targets, avoid).
Keeps only pairs marked valid (both eyes clearly open) with a label, and stores each eye as its own sample:
  images.npy uint8 [N,128,128] · labels.npy float32 [N,2] = (pitch, yaw) in Baballonia's 0-1 · subject.npy int16 [N]
  eye.npy uint8 [N] (0 left, 1 right, right is VRGaze's mirrored image) · subjects.txt (index -> subject id)
"""
import argparse
from pathlib import Path

import numpy as np


def norm(deg):
    return np.clip((deg + 45) / 90, 0, 1).astype(np.float32)        # BabbleTrainer's 0-1 gaze label


def select(f, labels):
    """(row mask, [N,2] labels as (pitch, yaw) 0-1) for one prepped subject."""
    with np.load(f) as d:
        valid = d["valid_left"].copy()
        if labels == "main":
            return valid, d["label_left"][:, :2]
    pseudo = np.load(f.with_suffix(".pseudo.npz"))
    if labels == "pseudo":
        return valid & pseudo["pseudo_ok"], np.stack([norm(pseudo["pseudo_pitch"]), norm(pseudo["pseudo_yaw"])], 1)
    return valid & pseudo["is_static"], np.stack([norm(pseudo["static_pitch"]), norm(pseudo["static_yaw"])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--labels", choices=["pseudo", "static", "main"], default="pseudo")
    a = ap.parse_args()
    src, dst = a.src, a.dst
    files = sorted(f for f in src.glob("id*.npz") if not f.name.endswith((".tmp.npz", ".pseudo.npz")))
    if a.labels != "main":
        files = [f for f in files if f.with_suffix(".pseudo.npz").exists()]
    counts = [int(select(f, a.labels)[0].sum()) * 2 for f in files]
    n = sum(counts)
    dst.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(dst / "images.npy", mode="w+", dtype=np.uint8, shape=(n, 128, 128))
    labels = np.lib.format.open_memmap(dst / "labels.npy", mode="w+", dtype=np.float32, shape=(n, 2))
    subject = np.lib.format.open_memmap(dst / "subject.npy", mode="w+", dtype=np.int16, shape=(n,))
    eye = np.lib.format.open_memmap(dst / "eye.npy", mode="w+", dtype=np.uint8, shape=(n,))
    at = 0
    for si, f in enumerate(files):
        v, lab = select(f, a.labels)
        with np.load(f) as d:
            for e, img in enumerate(("left", "right")):
                k = int(v.sum())
                images[at:at + k] = d[img][v]
                eye_lab = lab[v].copy()
                if e == 1:
                    # VRGaze mirrors the right image to look like a left eye, which reverses yaw: verified 10-15,
                    # corr(pupil x, yaw) -0.54 left vs +0.45 mirrored right, pitch keeps its sign. So mirror yaw.
                    eye_lab[:, 1] = 1.0 - eye_lab[:, 1]
                labels[at:at + k] = eye_lab
                subject[at:at + k] = si
                eye[at:at + k] = e
                at += k
        print(f"{f.stem}: {counts[si]} eye images ({at}/{n})", flush=True)
    for a in (images, labels, subject, eye):
        a.flush()
    (dst / "subjects.txt").write_text("\n".join(f.stem for f in files) + "\n")
    print(f"packed {n} eye images from {len(files)} subjects -> {dst} ({n * 128 * 128 / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
