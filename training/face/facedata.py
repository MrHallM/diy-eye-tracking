"""Load qpteach recordings (camera frames + Quest Pro face weights) as training data for the Babble face model.

A SESSION is one folder written by `tools/qpteach.py record` (frames.mjpeg, frames.csv, qp.csv, prompts.csv).
One session = one way of wearing the headset. Hold out whole sessions to test placement.

The image the model sees must be the image Baballonia will give it, so the same steps run here, in the same order
(ImageTransformer.cs): crop the ROI -> grayscale -> x gamma -> (flip) -> resize to 224 -> /255.
ROI / flip / gamma default to what Baballonia is REALLY using: FaceCamera in LocalSettings.json.
"""
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eyerec      # noqa: E402
import facemap     # noqa: E402
import qplog       # noqa: E402

SIZE = 224
FRAME = 240        # frames are cached at camera size so crop + augmentation happen at training time


def baballonia_face_settings():
    """(roi (x, y, w, h) or None, hflip, vflip, gamma) from the settings file Baballonia really loads."""
    root = Path(os.environ.get("APPDATA", "")) / "ProjectBabble"
    files = sorted(root.glob("**/LocalSettings.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None, False, False, 1.0
    fc = json.loads(files[0].read_text(encoding="utf-8-sig")).get("FaceCamera", {})
    r = fc.get("Roi", {})
    roi = (int(r.get("X", 0)), int(r.get("Y", 0)), int(r.get("Width", 0)), int(r.get("Height", 0)))
    return roi, bool(fc.get("UseHorizontalFlip")), bool(fc.get("UseVerticalFlip")), float(fc.get("Gamma", 1.0))


def valid_roi(roi, w=FRAME, h=FRAME):
    """Baballonia's own rule: a bad ROI, or one as wide/tall as the frame, means the full frame."""
    if roi is None:
        return 0, 0, w, h
    x, y, rw, rh = roi
    if x < 0 or y < 0 or rw <= 0 or rh <= 0 or x + rw > w or y + rh > h or rw == w or rh == h:
        return 0, 0, w, h
    return x, y, rw, rh


def preprocess(img, roi=None, hflip=False, vflip=False, gamma=1.0):
    """uint8 gray camera frame -> float32 [224, 224] in 0..1, the way Baballonia does it."""
    x, y, w, h = valid_roi(roi, img.shape[1], img.shape[0])
    out = img[y:y + h, x:x + w]
    if gamma != 1.0:
        out = cv2.convertScaleAbs(out, alpha=gamma)
    out = cv2.resize(out, (SIZE, SIZE))
    if hflip:
        out = cv2.flip(out, 1)
    if vflip:
        out = cv2.flip(out, 0)
    return out.astype(np.float32) / 255.0


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


STOCK = Path(__file__).resolve().parents[2] / "builds" / "baballonia-uncapped" / "faceModel.onnx"


def _lag(t_cam, frames, t_qp, jaw_qp):
    """Camera-vs-Quest time offset, the way qpteach.py check does it: the STOCK model's JawOpen (it follows the jaw
    well, r ~0.75) cross-correlated with the Quest's JawDrop. (A plain image-brightness signal was tried first and
    was too weak: |r| 0.22.) Positive = camera frames arrive later."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(STOCK), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    jaw = np.array([sess.run(None, {inp: preprocess(f)[None, None]})[0][0][facemap.BABBLE.index("JawOpen")]
                    for f in frames])
    grid = np.arange(max(t_cam[0], t_qp[0]) + 1, min(t_cam[-1], t_qp[-1]) - 1, 0.01)
    q = np.interp(grid, t_qp, jaw_qp)
    lags = np.arange(-0.3, 0.301, 0.01)
    cs = [_corr(q, np.interp(grid, t_cam - lag, jaw)) for lag in lags]
    i = int(np.nanargmax(cs))
    return float(lags[i]), float(cs[i])


class Session:
    """frames: uint8 [N, 240, 240] (memory-mapped cache) · y: float32 [N, 45] Quest Pro targets (NaN = no teacher)
    · t: frame times · seg: index of the spoken prompt each frame belongs to (-1 = none)"""

    def __init__(self, folder, stride=2, lag=None):
        self.folder = Path(folder)
        self.name = self.folder.parent.parent.name + "/" + self.folder.name
        cache = self.folder / f"cache_{FRAME}_s{stride}.npy"
        tfile = self.folder / f"cache_{FRAME}_s{stride}_t.npy"
        if not cache.exists():
            ts, imgs = [], []
            for k, (row, jpg) in enumerate(eyerec.read_frames(self.folder)):
                if k % stride or row.get("ok_jpeg", "1") != "1":
                    continue
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                if img.shape != (FRAME, FRAME):
                    img = cv2.resize(img, (FRAME, FRAME))
                imgs.append(img)
                ts.append(float(row["t_epoch"]))
            tmp = cache.with_suffix(".tmp.npy")
            np.save(tmp, np.stack(imgs))
            os.replace(tmp, cache)
            np.save(tfile, np.array(ts))
        self.frames = np.load(cache, mmap_mode="r")
        t = np.load(tfile)

        if (self.folder / "qp.csv").exists():
            with open(self.folder / "qp.csv", newline="") as f:
                rows = [r for r in csv.DictReader(f) if r.get("face_flags", "1") not in ("0", "")]
            t_qp = np.array([float(r["t_epoch"]) for r in rows])
            qp = {k: np.array([float(r[k]) for r in rows]) for k in qplog.EXPR}

            lagfile = self.folder / f"cache_{FRAME}_s{stride}_lag.json"
            self.lag_r = float("nan")
            if lag is None and lagfile.exists():
                d = json.loads(lagfile.read_text())
                lag, self.lag_r = d["lag"], d["r"]
            if lag is None:
                lag, self.lag_r = _lag(t, self.frames, t_qp, qp["JawDrop"])
                lagfile.write_text(json.dumps({"lag": lag, "r": self.lag_r}))
                if self.lag_r < 0.4:
                    print(f"  ⚠ {self.name}: weak jaw match (r {self.lag_r:.2f}); the time offset is unreliable. "
                          f"Pass lag= by hand or check the jaw part of the routine was recorded.")
            self.lag = lag
            tq = t - lag                                 # the Quest time each frame belongs to
            Y = facemap.labels(qp)
            y = np.stack([np.interp(tq, t_qp, Y[:, i]) if facemap.MAPPED[i] else np.full(len(t), np.nan)
                          for i in range(Y.shape[1])], axis=1).astype(np.float32)
            # keep frames that have a Quest sample close by on both sides (no label made up across a gap)
            j = np.clip(np.searchsorted(t_qp, tq), 1, len(t_qp) - 1)
            ok = (t_qp[j] - t_qp[j - 1] < 0.06) & (tq > t_qp[0]) & (tq < t_qp[-1])
            self.keep = np.flatnonzero(ok)
        else:                                            # --no-qp recording: frames + prompts only (rest test)
            self.lag, self.lag_r = 0.0, float("nan")
            y = np.full((len(t), len(facemap.BABBLE)), np.nan, np.float32)
            self.keep = np.arange(len(t))
        self.labelled = (self.folder / "qp.csv").exists()
        self.t, self.y = t, y

        self.seg = np.full(len(t), -1)
        self.prompts = []
        pf = self.folder / "prompts.csv"
        if pf.exists():
            with open(pf, newline="") as f:
                for k, r in enumerate(csv.DictReader(f)):
                    t0, secs = float(r["t_epoch"]), float(r["seconds"])
                    self.prompts.append(r["prompt"])
                    self.seg[(t >= t0) & (t < t0 + secs)] = k

    def rest(self):
        """Frame indices inside the 'Relax ... keep still' prompts (for the rest-drift test)."""
        ks = [k for k, p in enumerate(self.prompts) if p.lower().startswith("relax")]
        return np.array([i for i in self.keep if self.seg[i] in ks], dtype=int)

    def split_in_time(self, val_frac=0.3, gap=0.5):
        """DRY-RUN ONLY: inside every prompt, the first part trains and the last part validates, with a gap between.
        Same wear, same lighting -> this shows the pipeline learns; it says nothing about placement."""
        tr, va = [], []
        for k in sorted(set(self.seg[self.keep])):
            idx = self.keep[self.seg[self.keep] == k]
            if k < 0 or len(idx) < 20:
                tr += list(idx)
                continue
            t0, t1 = self.t[idx[0]], self.t[idx[-1]]
            cut = t1 - (t1 - t0) * val_frac
            tr += [i for i in idx if self.t[i] < cut - gap]
            va += [i for i in idx if self.t[i] >= cut]
        return np.array(tr, dtype=int), np.array(va, dtype=int)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        s = Session(a)
        print(f"{s.name}: {len(s.frames)} cached frames, {len(s.keep)} usable, lag {s.lag * 1000:+.0f} ms "
              f"(|r| {abs(getattr(s, 'lag_r', float('nan'))):.2f}), {len(s.prompts)} prompts, rest frames {len(s.rest())}")
    print("Baballonia face settings (roi, hflip, vflip, gamma):", baballonia_face_settings())
