"""First look at the GIW + TEyeD parquet shards: label coverage, blink counts, and a contact sheet with the eyelid
and pupil labels drawn on, to judge how close these images are to our cameras before writing a real prep step.

    python training/prep/explore_giw.py data/giw/data data/giw_explore

Writes stats.json + sheet_<shard>.jpg into the output folder.
"""
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

MOVES = {1: "error", 2: "blink", 3: "fixation", 4: "saccade", 5: "pursuit"}
LABEL_COLS = ["subject", "session", "eye", "frame_idx", "eye_movement", "validity_pupil", "validity_lid",
              "validity_iris", "lid_lm_2d", "pupil_cx", "pupil_cy", "pupil_w", "pupil_h", "pupil_angle"]


def openness(lid):
    """Lid gap at the widest point / corner-to-corner width. Upper = first 17 points, lower = last 17,
    both assumed to run corner to corner in the same direction (checked visually on the sheet)."""
    if lid is None:
        return None
    p = np.asarray(lid, np.float32).reshape(34, 2)
    up, lo = p[:17], p[17:]
    width = np.linalg.norm(up[-1] - up[0])
    return float(np.max(np.abs(lo[:, 1] - up[:, 1])) / width) if width > 0 else None


def draw(row):
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")), cv2.COLOR_GRAY2BGR)
    if row["lid_lm_2d"] is not None:
        p = np.asarray(row["lid_lm_2d"], np.float32).reshape(34, 2)
        cv2.polylines(img, [p[:17].astype(np.int32)], False, (0, 255, 0), 2)
        cv2.polylines(img, [p[17:].astype(np.int32)], False, (0, 160, 255), 2)
        cv2.circle(img, tuple(p[0].astype(int)), 6, (255, 0, 255), -1)  # first point: shows the direction
    if row["pupil_cx"] is not None and (row["pupil_w"] or 0) > 0 and (row["pupil_h"] or 0) > 0:  # -1 = no pupil
        cv2.ellipse(img, ((row["pupil_cx"], row["pupil_cy"]), (row["pupil_w"], row["pupil_h"]), row["pupil_angle"]),
                    (0, 0, 255), 2)
    o = openness(row["lid_lm_2d"])
    cv2.putText(img, f"{MOVES.get(row['eye_movement'], '?')} open={o:.2f}" if o is not None else
                f"{MOVES.get(row['eye_movement'], '?')} no lid", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    return cv2.resize(img, (320, 240))


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    rng = np.random.default_rng(0)
    for shard in sorted(src.glob("*.parquet")):
        t = pq.read_table(shard, columns=[c for c in LABEL_COLS])
        d = t.to_pydict()
        n = t.num_rows
        moves = np.array([m if m is not None else 0 for m in d["eye_movement"]])
        opens = np.array([o for o in map(openness, d["lid_lm_2d"]) if o is not None])
        stats[shard.name] = {
            "rows": n, "subject": sorted(set(d["subject"])), "eyes": sorted(set(d["eye"])),
            "valid_pupil": round(float(np.mean([bool(v) for v in d["validity_pupil"]])), 3),
            "valid_lid": round(float(np.mean([bool(v) for v in d["validity_lid"]])), 3),
            "moves": {MOVES.get(k, str(k)): int((moves == k).sum()) for k in np.unique(moves)},
            "openness_p5_p50_p95": [round(float(x), 3) for x in np.percentile(opens, [5, 50, 95])] if len(opens) else None,
        }
        print(shard.name, json.dumps(stats[shard.name]), flush=True)

        # contact sheet: 8 random frames + 4 blink frames, read with images only for the picked rows
        blink = np.flatnonzero(moves == 2)
        pick = list(rng.choice(n, 8, replace=False)) + (list(rng.choice(blink, min(4, len(blink)), replace=False)) if len(blink) else [])
        full = pq.read_table(shard, columns=LABEL_COLS + ["image"]).take(sorted(int(i) for i in pick)).to_pylist()
        tiles = [draw(r) for r in full]
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(out / f"sheet_{shard.stem}.jpg"), sheet)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
