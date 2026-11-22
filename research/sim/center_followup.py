"""
center_followup.py - two follow-ups to center_sim.py (same data, same noise model):
  A. Per-axis floor + HORIZONTAL-only step detection (the 2-D test mixed in the big vertical wander).
  B. Does head orientation explain the vertical centre? (VRChat gives us head pitch for free.) HeadOrnt rows are
     unlabelled in the data; row 0 has a median near -9 deg (pitch-like), so all three rows are tested.
"""
import math
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from b24_load import DATA, load  # noqa: E402
from center_sim import FS, noise, units, wmedian, window_est  # noqa: E402

G = "fix + still + >=0.5 s"


def main():
    rng = np.random.default_rng(2)
    recs = [r for r in load() if r["TimeGaze"].size > FS * 120]
    with h5py.File(DATA, "r") as f:
        t = f["TIMESERIES"]
        allrecs = load()                                  # same order as the file
        orn = [np.array(f[t["HeadOrnt"][i, 0]]) for i in range(t["HeadOrnt"].shape[0])]
    keep = [i for i, r in enumerate(allrecs) if r["TimeGaze"].size > FS * 120]
    orn = [orn[i] for i in keep]

    # ---------- A ----------
    fl = {"x": {}, "y": {}}
    for W in (120, 300, 600):
        ex, ey = [], []
        for r in recs:
            gx, gy = r["GazeInHeadX"], r["GazeInHeadY"]
            U = units(r, gx + noise(gx.size, rng, 3, 4), gy + noise(gy.size, rng, 3, 4))[G]
            Ut = units(r, gx, gy)[G]
            if len(Ut[0]) < 8:
                continue
            cx, cy = wmedian(Ut[1], Ut[3]), wmedian(Ut[2], Ut[3])
            for te in np.arange(U[0][0] + W, U[0][-1] + 1e-9, 10.0):
                e = window_est(U, te, W)
                if e:
                    ex.append(abs(e[0] - cx)); ey.append(abs(e[1] - cy))
        fl["x"][W], fl["y"][W] = np.array(ex), np.array(ey)
    print("A. per-axis floor, 5 deg noise, gate", repr(G), "(deg; median / 95th)")
    for W in (120, 300, 600):
        print(f"   {W:>3}s  horizontal {np.median(fl['x'][W]):4.2f} / {np.percentile(fl['x'][W], 95):4.2f}   "
              f"vertical {np.median(fl['y'][W]):4.2f} / {np.percentile(fl['y'][W], 95):4.2f}")
    longrecs = [r for r in recs if r["TimeGaze"][-1] - r["TimeGaze"][0] >= 600]
    print(f"   horizontal-only step detection ({len(longrecs)} recordings >= 10 min; median/90th s, detected %)")
    for W in (120, 300):
        thr = np.percentile(fl["x"][W], 95)
        line = []
        for step in (2.0, 3.0, 5.0):
            lat = []
            for r in longrecs:
                gx, gy, tt = r["GazeInHeadX"], r["GazeInHeadY"], r["TimeGaze"]
                ts = tt[0] + (tt[-1] - tt[0]) / 2
                U = units(r, gx + noise(gx.size, rng, 3, 4) + np.where(tt >= ts, step, 0.0),
                          gy + noise(gy.size, rng, 3, 4))[G]
                base = window_est(U, ts, W)
                if not base:
                    continue
                for te in np.arange(ts + 5, ts + 2 * W + 1e-9, 5.0):
                    e = window_est(U, te, W)
                    if e and abs(e[0] - base[0]) > thr:
                        lat.append(te - ts)
                        break
            line.append(f"{step:.0f} deg: {np.median(lat) if lat else float('nan'):4.0f}/"
                        f"{np.percentile(lat, 90) if lat else float('nan'):4.0f}s {100 * len(lat) / len(longrecs):3.0f}%")
        print(f"   window {W}s (threshold {thr:4.2f})   " + "   ".join(line))

    # ---------- B ----------
    print("\nB. vertical eye centre vs head orientation")
    sess = []
    within = {0: [], 1: [], 2: []}
    for r, o in zip(recs, orn):
        ti = r["TimeIMU"]
        k = min(ti.size, o.shape[1])
        tg = r["TimeGaze"]
        rows = [np.interp(tg, ti[:k], o[j, :k]) for j in range(3)]
        m = (r["Class"] == 1) & np.isfinite(r["GazeInHeadY"])
        if m.sum() < 500:
            continue
        ey = r["GazeInHeadY"][m]
        sess.append([np.median(ey)] + [np.median(rows[j][m]) for j in range(3)])
        for j in range(3):                                   # within-session: demeaned, subsampled
            hj = rows[j][m][::25]
            within[j].append(np.c_[hj - hj.mean(), ey[::25] - ey[::25].mean()])
    S = np.array(sess)
    for j in range(3):
        c = np.corrcoef(S[:, 0], S[:, j + 1])[0, 1]
        W_ = np.vstack(within[j])
        cw = np.corrcoef(W_[:, 0], W_[:, 1])[0, 1]
        slope = np.polyfit(W_[:, 0], W_[:, 1], 1)[0]
        # how much of the across-session vertical spread is left after removing the head-row fit
        p = np.polyfit(S[:, j + 1], S[:, 0], 1)
        resid = S[:, 0] - np.polyval(p, S[:, j + 1])
        print(f"   HeadOrnt row {j}: across sessions r={c:+.2f} (vertical centre SD {S[:, 0].std():.2f} -> "
              f"{resid.std():.2f} deg after fit)   within sessions r={cw:+.2f}, slope {slope:+.2f} deg eye per deg head")


if __name__ == "__main__":
    main()
