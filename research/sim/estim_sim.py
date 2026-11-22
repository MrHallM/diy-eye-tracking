"""
estim_sim.py - how much can the ESTIMATION LAYER (project_eye_tracking_design, item 1) actually fix?

Real head-free gaze (Burlingham24, Tobii 50 Hz, see b24_load.py) is the truth. A worse tracker is made from it:
    measured = truth + white noise + slow wandering noise (AR(1), tau 1 s)
               + GLITCH frames (single-frame jumps of 10-40 deg, 0.5 % of frames)
               + BLINK artefact (synthetic blink times, ~15/min: gaze dragged down ~20 deg and noisy while the lid moves)
Then four outputs are compared against the truth on people the tuning never saw:
    raw            the tracker as it is
    1euro-babble   Baballonia's own smoothing defaults (min cutoff 0.5, beta 3 on a -1..1 signal; here /45 deg)
    1euro-tuned    one-euro with min cutoff / beta tuned on the other half of the people
    layer          blink HOLD + physically-impossible-jump gate (one-frame confirm) + the tuned one-euro

The tracker's noise SHAPE has never been measured, so three shapes with the same 5 deg total are run:
    white-heavy (5 + 0), mixed (3 + 4, the guess used in center_sim.py), slow-heavy (1 + 4.9).

Also tested: design item 2's idea that "raw minus estimated" drifting one way reveals SLIP. A 5 deg step is added to
the raw stream halfway; does the divergence stay different afterwards?

    python -E -P analysis/sim/estim_sim.py            (about 3-4 minutes)
"""
import math
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
from b24_load import load  # noqa: E402

FS = 50.0
DT = 1 / FS
MAX_REC, MAX_S = 70, 240          # recordings used, seconds of each
VMAX = 800.0                      # deg/s: above the fastest human saccades (main sequence tops out ~500-700)
P_GLITCH, P_MISS = 0.005, 0.10    # glitch frames; share of blinks the lid signal fails to flag


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def segment(rec, rng):
    """First MAX_S seconds of a recording. The public file has no gaps and no blink marks (blinks are already
    filled in), so blink TIMES are synthetic: about 15 a minute, each 120-280 ms. The eye position under the lid is
    whatever the file has."""
    gx, gy = rec["GazeInHeadX"], rec["GazeInHeadY"]
    n = min(len(gx), len(gy), len(rec["Class"]), int(MAX_S * FS))
    if n < 60 * FS or not (np.isfinite(gx[:n]).all() and np.isfinite(gy[:n]).all()):
        return None
    blink = np.zeros(n, bool)
    t = rng.exponential(4.0)
    while t * FS < n - 30:
        a = int(t * FS)
        blink[a:a + int(rng.integers(6, 15))] = True
        t += 0.5 + rng.exponential(4.0)
    return gx[:n].copy(), gy[:n].copy(), blink, (rec["Class"][:n] == 1) & ~blink


def ar_white(n, rng, sw, sa, tau=1.0):
    out = rng.normal(0, sw, n) if sw else np.zeros(n)
    if sa:
        phi = math.exp(-1 / (FS * tau))
        e = rng.normal(0, sa * math.sqrt(1 - phi * phi), n)
        ar, _ = lfilter([1.0], [1.0, -phi], e, zi=[phi * rng.normal(0, sa)])
        out = out + ar
    return out


def make_tracker(x, y, blink, rng, sw, sa):
    n = len(x)
    mx, my = x + ar_white(n, rng, sw, sa), y + ar_white(n, rng, sw, sa)
    g = rng.random(n) < P_GLITCH
    amp, ang = rng.uniform(10, 40, n), rng.uniform(0, 2 * math.pi, n)
    mx[g] += (amp * np.cos(ang))[g]
    my[g] += (amp * np.sin(ang))[g]
    # blink artefact: 2 frames of lid travel either side of the closed period
    s, e = runs(blink)
    art = np.zeros(n)
    flag = np.zeros(n, bool)
    for a, b in zip(s, e):
        a2, b2 = max(a - 2, 0), min(b + 2, n)
        art[a2:b2] = 1.0
        art[a2:a] = np.linspace(0.3, 0.8, a - a2) if a > a2 else art[a2:a]
        art[b:b2] = np.linspace(0.8, 0.3, b2 - b) if b2 > b else art[b:b2]
        if rng.random() > P_MISS:
            flag[min(a2 + 1, n - 1):min(b2 + 1, n)] = True          # lid signal: one frame late
    my = my - 20 * art + rng.normal(0, 6, n) * art
    mx = mx + rng.normal(0, 6, n) * art
    return mx, my, flag, g, art > 0


def one_euro(mx, my, mincut, beta, scale=1.0):
    """Standard 1-euro, both axes. scale: the signal is divided by it before the speed term (Baballonia filters a
    -1..1 value, so its beta acts on a signal ~45x smaller than degrees)."""
    n = len(mx)
    ox, oy = np.empty(n), np.empty(n)
    px, py, dx, dy = mx[0], my[0], 0.0, 0.0
    ad = 1 / (1 + 1 / (2 * math.pi * 1.0 * DT))                      # derivative cutoff 1 Hz
    ox[0], oy[0] = px, py
    for i in range(1, n):
        dx += ad * ((mx[i] - px) / DT / scale - dx)
        dy += ad * ((my[i] - py) / DT / scale - dy)
        ax = 1 / (1 + 1 / (2 * math.pi * (mincut + beta * abs(dx)) * DT))
        ay = 1 / (1 + 1 / (2 * math.pi * (mincut + beta * abs(dy)) * DT))
        px += ax * (mx[i] - px)
        py += ay * (my[i] - py)
        ox[i], oy[i] = px, py
    return ox, oy


def layer(mx, my, flag, mincut, beta, use_blink=True, use_gate=True):
    """Blink hold + impossible-jump gate + 1-euro. Causal: frame i uses only frames <= i."""
    n = len(mx)
    ox, oy = np.empty(n), np.empty(n)
    px, py, dx, dy = mx[0], my[0], 0.0, 0.0
    ad = 1 / (1 + 1 / (2 * math.pi * 1.0 * DT))
    jump = VMAX * DT                                                 # 16 deg in one 20 ms frame
    pend = None
    reopen = 0
    ox[0], oy[0] = px, py
    for i in range(1, n):
        x, y = mx[i], my[i]
        if use_blink and flag[i]:
            reopen = 2                                               # keep holding 2 frames after the lid reopens
            pend = None
            ox[i], oy[i] = px, py
            continue
        if reopen:
            reopen -= 1
            ox[i], oy[i] = px, py
            if reopen == 0:
                dx = dy = 0.0
            continue
        if use_gate:
            far = math.hypot(x - px, y - py) > jump
            if pend is not None:
                if math.hypot(x - pend[0], y - pend[1]) <= jump:     # it stayed there: a real, big saccade
                    px, py = (x + pend[0]) / 2, (y + pend[1]) / 2
                    dx = dy = 0.0
                    pend = None
                    ox[i], oy[i] = px, py
                    continue
                pend = None                                          # it came back: the last frame was a glitch
                far = math.hypot(x - px, y - py) > jump
            if far:
                pend = (x, y)                                        # wait one frame before believing it
                ox[i], oy[i] = px, py
                continue
        dx += ad * ((x - px) / DT - dx)
        dy += ad * ((y - py) / DT - dy)
        ax = 1 / (1 + 1 / (2 * math.pi * (mincut + beta * abs(dx)) * DT))
        ay = 1 / (1 + 1 / (2 * math.pi * (mincut + beta * abs(dy)) * DT))
        px += ax * (x - px)
        py += ay * (y - py)
        ox[i], oy[i] = px, py
    return ox, oy


def lag_ms(tx, ty, ox, oy):
    """Delay of the output behind the truth: the shift that best lines up their velocities."""
    tv = np.concatenate([np.diff(tx), np.diff(ty)])
    best, bl = -2, 0
    for k in range(0, 11):
        ov = np.concatenate([np.diff(ox)[k:], np.diff(oy)[k:]])
        t2 = np.concatenate([np.diff(tx)[:len(tx) - 1 - k], np.diff(ty)[:len(ty) - 1 - k]])
        c = np.corrcoef(ov, t2)[0, 1]
        if c > best:
            best, bl = c, k
    return bl * 1000 / FS


def metrics(segs, outs):
    """segs: list of (x, y, blink, fix, near_blink); outs: list of (ox, oy). Pooled."""
    e_all, e_fix, e_blk, darts, mins, lags = [], [], [], 0, 0.0, []
    for (x, y, blink, fix, nb), (ox, oy) in zip(segs, outs):
        err = np.hypot(ox - x, oy - y)
        e_all.append(err)
        e_fix.append(err[fix & ~nb])
        e_blk.append(err[nb])
        tsp = np.hypot(np.diff(x), np.diff(y)) * FS
        osp = np.hypot(np.diff(ox), np.diff(oy)) * FS
        tquiet = np.convolve(tsp, np.ones(5), "same") / 5 < 30        # the eye really is still around here
        d = (osp > 150) & tquiet
        darts += int(((d[1:]) & (~d[:-1])).sum())
        mins += len(x) / FS / 60
        lags.append(lag_ms(x, y, ox, oy))
    rms = lambda v: float(np.sqrt(np.mean(np.concatenate(v) ** 2)))   # noqa: E731
    return rms(e_all), rms(e_fix), rms(e_blk), darts / mins, float(np.median(lags))


def main():
    recs = load()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(recs))
    data = []
    for i in order:
        s = segment(recs[i], rng)
        if s is not None:
            data.append((recs[i]["sub"], s))
        if len(data) >= MAX_REC:
            break
    tune = [d for sub, d in data if zlib.crc32(sub.encode()) % 2 == 0]
    test = [d for sub, d in data if zlib.crc32(sub.encode()) % 2 == 1]
    hrs = lambda L: sum(len(d[0]) for d in L) / FS / 3600               # noqa: E731
    nb = sum(len(runs(d[2])[0]) for d in test)
    print(f"{len(data)} recordings: tune {len(tune)} ({hrs(tune):.2f} h), test {len(test)} ({hrs(test):.2f} h, "
          f"{nb} synthetic blinks, {nb / (hrs(test) * 60):.1f}/min)")

    for label, sw, sa in [("WHITE-HEAVY 5 + 0", 5.0, 0.0), ("MIXED 3 + 4", 3.0, 4.0), ("SLOW-HEAVY 1 + 4.9", 1.0, 4.9)]:
        def sim(L, seed):
            r = np.random.default_rng(seed)
            segs, meas = [], []
            for x, y, blink, fix in L:
                mx, my, flag, g, near = make_tracker(x, y, blink, r, sw, sa)
                segs.append((x, y, blink, fix, near))
                meas.append((mx, my, flag))
            return segs, meas
        tsegs, tmeas = sim(tune, 1)
        # tune one-euro on the tuning people: lowest error in fixations with lag <= 40 ms
        best = None
        for mc in (0.2, 0.5, 1.0, 2.0, 4.0):
            for beta in (0.0, 0.002, 0.007, 0.02, 0.06):
                o = [layer(mx, my, fl, mc, beta) for mx, my, fl in tmeas]
                m = metrics(tsegs, o)
                if m[4] <= 40 and (best is None or m[1] < best[0]):
                    best = (m[1], mc, beta)
        _, mc, beta = best
        segs, meas = sim(test, 2)
        outs = {
            "raw": [(mx, my) for mx, my, _ in meas],
            "1euro-babble": [one_euro(mx, my, 0.5, 3.0, scale=45.0) for mx, my, _ in meas],
            "1euro-tuned": [one_euro(mx, my, mc, beta) for mx, my, _ in meas],
            "layer: blink hold only": [layer(mx, my, fl, mc, beta, True, False) for mx, my, fl in meas],
            "layer: jump gate only": [layer(mx, my, fl, mc, beta, False, True) for mx, my, fl in meas],
            "layer (all)": [layer(mx, my, fl, mc, beta) for mx, my, fl in meas],
        }
        print(f"\n=== tracker noise {label} deg (per axis) · tuned one-euro: min cutoff {mc}, beta {beta}")
        print(f"{'output':24}{'error all':>10}{'in fixations':>13}{'around blinks':>14}{'false darts/min':>16}{'delay ms':>9}")
        for k, o in outs.items():
            m = metrics(segs, o)
            print(f"{k:24}{m[0]:10.2f}{m[1]:13.2f}{m[2]:14.2f}{m[3]:16.1f}{m[4]:9.0f}")
        # what no filter can remove: the slow part of the noise, on its own
        if sa:
            print(f"{'(slow noise alone)':24}{sa * math.sqrt(2):10.2f}   <- 2-D size of the wandering part; "
                  f"only calibration can remove it")

        if label.startswith("MIXED"):
            # design item 2: does raw-minus-estimated show a slip?
            pre, post, peak = [], [], []
            for (x, y, blink, fix, near), (mx, my, fl) in zip(segs, meas):
                h = len(mx) // 2
                mx2 = mx.copy()
                mx2[h:] += 5.0
                ox, oy = layer(mx2, my, fl, mc, beta)
                d = mx2 - ox                                        # signed divergence, horizontal
                w = int(5 * FS)
                pre.append(d[h - 6 * w:h].mean())
                post.append(d[h + w:h + 7 * w].mean())
                peak.append(d[h:h + w].mean())
            print(f"\nSLIP TEST (5 deg step added to raw, mixed noise): mean of raw-minus-estimated, horizontal")
            print(f"   30 s before the slip {np.mean(pre):+.2f} deg · first 5 s after {np.mean(peak):+.2f} · "
                  f"5-35 s after {np.mean(post):+.2f}   (spread between recordings ±{np.std(post):.2f})")


if __name__ == "__main__":
    main()
