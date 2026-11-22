"""
center_sim.py - would a target-free "centre prior" catch drift on a ~5 deg tracker? Simulated on REAL head-free gaze
(Burlingham24: 169 recordings, ~30 people, 9 everyday tasks, Tobii glasses 50 Hz + head IMU; see b24_load.py).

Idea (10-11): no targets needed. Gaze-in-head sits near the person's own centre most of the time, so a slow,
robust estimate of that centre, taken at good moments (steady fixations, head still), shows when the tracker's
"forward" has moved (slip / drift) and by how much.

Treat Tobii gaze as the truth, make a worse tracker from it (measured = truth + offset + noise), and ask:
  1. FLOOR   - with NO drift, how far does the windowed centre estimate wander from the session's centre?
               A real drift smaller than this can't be told apart from normal viewing. Per window length + gating.
  2. TASKS   - the same person, different task: how far does the true centre move? That is "false drift" the
               estimator will report when you switch activity (VRChat: mirror, menus, dancing...).
  3. STEP    - inject a sudden offset (a slip) halfway through: how long until the estimate sees it?

Noise model (per axis): white sigma_w + AR(1) wander sigma_a (tau 1 s) -> RMS sqrt(w^2+a^2). Default 3 + 4 = 5 deg RMS.
This is a GUESS at Baballonia's error shape (never measured, see reference_baballonia_internals); --noise changes it.

    python -E -P analysis/sim/center_sim.py                 (all three tests, default noise)
    python -E -P analysis/sim/center_sim.py --noise 0 0     (no tracker noise: viewing behaviour alone)
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).parent))
from b24_load import load  # noqa: E402

FS = 50.0
GATES = {                       # name: (fixations only, head speed max deg/s, min fixation s, duration-weighted)
    "all samples":            (False, None, 0.0, False),
    "fixations":              (True, None, 0.0, False),
    "fix + head still":       (True, 20.0, 0.0, False),
    "fix + still + >=0.5 s":  (True, 20.0, 0.5, False),
    "fix + still, dur-weight": (True, 20.0, 0.0, True),
}
WINDOWS = [30, 60, 120, 300, 600]


def wmedian(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return v[o][np.searchsorted(c, c[-1] / 2)]


def noise(n, rng, sw, sa, tau=1.0):
    if sw == 0 and sa == 0:
        return np.zeros(n)
    phi = math.exp(-1 / (FS * tau))
    e = rng.normal(0, sa * math.sqrt(1 - phi * phi), n)
    ar, _ = lfilter([1.0], [1.0, -phi], e, zi=[phi * rng.normal(0, sa)])   # AR(1), stationary start
    return rng.normal(0, sw, n) + ar


def units(rec, gx, gy):
    """Return the samples/fixations the estimator may use, as arrays t, x, y, weight, per gate."""
    t = rec["TimeGaze"]
    ti, hv = rec["TimeIMU"], rec["HeadVelAngMag"]
    k = min(ti.size, hv.size)                         # the two IMU arrays can differ by one sample
    hs = np.interp(t, ti[:k], hv[:k])
    cls = rec["Class"]
    ok = np.isfinite(gx) & np.isfinite(gy) & (cls != 999)
    fix = (cls == 1) & ok
    # fixation runs
    d = np.diff(np.r_[0, fix.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    F = []
    for a, b in zip(starts, ends):
        F.append(((t[a] + t[b - 1]) / 2, gx[a:b].mean(), gy[a:b].mean(), (b - a) / FS, np.median(hs[a:b])))
    F = np.array(F) if F else np.zeros((0, 5))
    out = {}
    for name, (fonly, hmax, dmin, dw) in GATES.items():
        if not fonly:
            step = 5                                    # 10 Hz is plenty for a median
            out[name] = (t[ok][::step], gx[ok][::step], gy[ok][::step], np.ones(ok.sum())[::step])
            continue
        m = np.ones(len(F), bool)
        if hmax is not None:
            m &= F[:, 4] < hmax
        m &= F[:, 3] >= dmin
        out[name] = (F[m, 0], F[m, 1], F[m, 2], F[m, 3] if dw else np.ones(m.sum()))
    return out


def window_est(u, t_end, W, min_n=8):
    t, x, y, w = u
    m = (t > t_end - W) & (t <= t_end)
    if m.sum() < min_n:
        return None
    return wmedian(x[m], w[m]), wmedian(y[m], w[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", nargs=2, type=float, default=[3.0, 4.0], metavar=("WHITE", "AR"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    sw, sa = a.noise
    print(f"loading... (noise white {sw} + AR {sa} -> {math.hypot(sw, sa):.1f} deg RMS per axis)")
    recs = [r for r in load() if r["TimeGaze"].size > FS * 120]
    print(f"{len(recs)} recordings >= 2 min")

    # ---------- 1. FLOOR ----------
    floor = defaultdict(list)                         # (gate, W) -> per-estimate error (deg, 2-D)
    true_centre = {}
    for k, r in enumerate(recs):
        gx, gy = r["GazeInHeadX"], r["GazeInHeadY"]
        nx, ny = noise(gx.size, rng, sw, sa), noise(gx.size, rng, sw, sa)
        U_true = units(r, gx, gy)
        U_meas = units(r, gx + nx, gy + ny)
        for g in GATES:
            t, x, y, w = U_true[g]
            if len(t) < 8:
                continue
            cx, cy = wmedian(x, w), wmedian(y, w)       # session centre from TRUE gaze, same gate
            true_centre[(k, g)] = (cx, cy)
            tm = U_meas[g][0]
            for W in WINDOWS:
                for te in np.arange(tm[0] + W, tm[-1] + 1e-9, 10.0):
                    e = window_est(U_meas[g], te, W)
                    if e:
                        floor[(g, W)].append(math.hypot(e[0] - cx, e[1] - cy))
    print("\n1. FLOOR: windowed centre estimate vs the session's own centre, NO drift (deg; median / 95th pct)")
    print("   gate                       " + "".join(f"{W:>13}s" for W in WINDOWS))
    for g in GATES:
        cells = []
        for W in WINDOWS:
            v = np.array(floor[(g, W)])
            cells.append(f"{np.median(v):5.2f} / {np.percentile(v, 95):5.2f}" if v.size else "      -      ")
        print(f"   {g:<26} " + " ".join(f"{c:>13}" for c in cells))

    # ---------- 2. TASKS ----------
    g = "fix + head still"
    by_sub = defaultdict(list)
    for k, r in enumerate(recs):
        if (k, g) in true_centre:
            by_sub[r["sub"]].append((r["task"], *true_centre[(k, g)]))
    dx, dy, dd = [], [], []
    for s, L in by_sub.items():
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                dx.append(abs(L[i][1] - L[j][1])); dy.append(abs(L[i][2] - L[j][2]))
                dd.append(math.hypot(L[i][1] - L[j][1], L[i][2] - L[j][2]))
    multi = sum(1 for L in by_sub.values() if len(L) > 1)
    print(f"\n2. TASKS: same person, two different tasks, true centre ({g}); {multi} people with 2+ tasks, "
          f"{len(dd)} pairs")
    for name, v in (("horizontal", dx), ("vertical", dy), ("2-D", dd)):
        v = np.array(v)
        print(f"   {name:<10} median {np.median(v):5.2f}   75th {np.percentile(v, 75):5.2f}   "
              f"95th {np.percentile(v, 95):5.2f}   max {v.max():5.2f} deg")
    cxs = np.array([c[0] for (k, gg), c in true_centre.items() if gg == g])
    cys = np.array([c[1] for (k, gg), c in true_centre.items() if gg == g])
    print(f"   session centres across everyone: x mean {cxs.mean():+.2f} (SD {cxs.std():.2f}), "
          f"y mean {cys.mean():+.2f} (SD {cys.std():.2f}) deg")
    per_task = defaultdict(list)
    for k, r in enumerate(recs):
        if (k, g) in true_centre:
            per_task[r["task"]].append(true_centre[(k, g)])
    print("   per task (mean x, y):  " + "  ".join(
        f"{tk} {np.mean([c[0] for c in v]):+.1f},{np.mean([c[1] for c in v]):+.1f}" for tk, v in sorted(per_task.items())))

    # ---------- 3. STEP ----------
    print("\n3. STEP: sudden horizontal offset halfway through (recordings >= 10 min); detect when the estimate moves")
    print("   past that window's 95th-pct floor. Time from the step to detection (s; median / 90th), detected %")
    g = "fix + still + >=0.5 s"
    longrecs = [r for r in recs if r["TimeGaze"][-1] - r["TimeGaze"][0] >= 600]
    print(f"   gate '{g}', {len(longrecs)} recordings")
    for W in (60, 120, 300):
        thr = np.percentile(floor[(g, W)], 95)
        line = []
        for step in (2.0, 3.0, 5.0):
            lat, hit = [], 0
            for r in longrecs:
                gx, gy = r["GazeInHeadX"].copy(), r["GazeInHeadY"]
                t = r["TimeGaze"]
                ts = t[0] + (t[-1] - t[0]) / 2
                gx_m = gx + noise(gx.size, rng, sw, sa) + np.where(t >= ts, step, 0.0)
                gy_m = gy + noise(gy.size, rng, sw, sa)
                U = units(r, gx_m, gy_m)[g]
                base = window_est(U, ts, W)
                if not base:
                    continue
                found = None
                for te in np.arange(ts + 5, ts + W * 2 + 1e-9, 5.0):
                    e = window_est(U, te, W)
                    if e and math.hypot(e[0] - base[0], e[1] - base[1]) > thr:
                        found = te - ts
                        break
                if found is not None:
                    hit += 1
                    lat.append(found)
            n = len(longrecs)
            line.append(f"{step:.0f} deg: {np.median(lat) if lat else float('nan'):4.0f}/"
                        f"{np.percentile(lat, 90) if lat else float('nan'):4.0f}s {100 * hit / n:3.0f}%")
        print(f"   window {W:>3}s (floor 95th {thr:4.2f} deg)   " + "   ".join(line))


if __name__ == "__main__":
    main()
