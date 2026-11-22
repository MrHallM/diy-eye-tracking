# Usage: python training/prep/giw_align_check.py data/giw/data   (result 2026-10-22 is in DATASETS.md)
"""Per stream: median px distance between a dark-blob pupil detector and the label, on 60 random fixation frames,
using (a) the stream's own labels, (b) the partner eye's labels at the same frame index. Also first vs last third."""
import io, sys, json, numpy as np, pyarrow.parquet as pq, cv2
from PIL import Image
src = sys.argv[1]
shards = [f"{src}/train-{i:05d}-of-00012.parquet" for i in range(1, 13)]
def det(b):
    g = cv2.GaussianBlur(np.array(Image.open(io.BytesIO(b)).convert("L")), (15, 15), 0)
    ys, xs = np.nonzero(g <= np.percentile(g, 1.5)); return xs.mean(), ys.mean()
lab = {}
for s in shards:
    t = pq.read_table(s, columns=["frame_idx", "pupil_cx", "pupil_cy", "pupil_w", "eye_movement"]).to_pydict()
    lab[s] = {f: (x, y, w, m) for f, x, y, w, m in zip(t["frame_idx"], t["pupil_cx"], t["pupil_cy"], t["pupil_w"], t["eye_movement"])}
out = {}
for k, s in enumerate(shards):
    partner = shards[k ^ 1]  # shards come in eye0/eye1 pairs
    fr = sorted(f for f, (x, y, w, m) in lab[s].items() if m == 3 and (w or 0) > 0)
    rng = np.random.default_rng(2); pick = set(rng.choice(fr, 60, replace=False).tolist())
    t = pq.read_table(s, columns=["frame_idx", "image"]); fi = t["frame_idx"].to_pylist()
    rows = t.take([i for i, f in enumerate(fi) if f in pick]).to_pylist()
    own, oth, pos = [], [], []
    for r in rows:
        cx, cy = det(r["image"]["bytes"]); f = r["frame_idx"]
        x, y, *_ = lab[s][f]; own.append(np.hypot(x - cx, y - cy)); pos.append(f / max(fi))
        if f in lab[partner] and (lab[partner][f][2] or 0) > 0:
            px, py, *_ = lab[partner][f]; oth.append(np.hypot(px - cx, py - cy))
    own, pos = np.array(own), np.array(pos)
    out[s[-28:]] = dict(own=round(float(np.median(own)), 1), partner=round(float(np.median(oth)), 1) if oth else None,
                        first_third=round(float(np.median(own[pos < 1/3])), 1), last_third=round(float(np.median(own[pos > 2/3])), 1),
                        frac_within_15px=round(float(np.mean(own < 15)), 2))
    print(s[-28:], out[s[-28:]], flush=True)
