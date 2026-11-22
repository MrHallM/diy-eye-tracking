"""Stats for one Quest Pro play session logged by tools/qplog.py (--all) + tools/poselog.py.
usage: python analysis/play/session_stats.py tests/QP_<date>"""
import sys, csv, numpy as np
from pathlib import Path
d = Path(sys.argv[1])
rows = list(csv.reader(open(d / 'qp.csv')))
H = rows[0]; A = np.array([[float(x) for x in r] for r in rows[1:] if len(r) == len(H)])
c = {n: i for i, n in enumerate(H)}; col = lambda n: A[:, c[n]]
t = col('t_epoch'); dur = t[-1] - t[0]; dt = np.diff(t)
print(f'== EYE LOG: {len(t)} samples over {dur/60:.1f} min, median rate {1/np.median(dt):.1f} Hz, gaps >100 ms: {(dt>0.1).sum()} (longest {dt.max()*1000:.0f} ms)')
valid = (col('lvalid') == 1) & (col('rvalid') == 1)
print(f'   both eyes valid {valid.mean()*100:.1f}%  | mean conf L/R {col("lconf")[valid].mean():.3f}/{col("rconf")[valid].mean():.3f}')
yaw = (col('lyaw') + col('ryaw')) / 2; pit = (col('lpitch') + col('rpitch')) / 2
verg = col('ryaw') - col('lyaw')
# blinks from the face weights
ec = np.maximum(col('EyesClosedL'), col('EyesClosedR'))
closed = ec > 0.5; edges = np.diff(closed.astype(int)); starts = np.where(edges == 1)[0] + 1; ends = np.where(edges == -1)[0] + 1
if len(ends) and len(starts) and ends[0] < starts[0]: ends = ends[1:]
n = min(len(starts), len(ends)); bd = (t[ends[:n]] - t[starts[:n]]) * 1000
print(f'== BLINKS (EyesClosed > 0.5): {n} in {dur/60:.1f} min = {n/(dur/60):.1f}/min | duration median {np.median(bd):.0f} ms, p10 {np.percentile(bd,10):.0f}, p90 {np.percentile(bd,90):.0f}, long (>500 ms) {(bd>500).sum()}')
if n:
    inval = np.mean([~valid[s:e].all() for s, e in zip(starts[:n], ends[:n])]) * 100
    moved = []
    for s, e in zip(starts[:n], ends[:n]):
        seg = yaw[s:e]; moved.append(np.ptp(seg) if len(seg) > 1 else 0)
    print(f'   gaze marked INVALID during {inval:.0f}% of blinks | gaze change inside a blink: median {np.median(moved):.2f}°, p90 {np.percentile(moved,90):.2f}° (≈0 = QP HOLDS gaze through blinks)')
op = ~closed & valid
print(f'== GAZE RANGE (eyes open+valid, deg; +yaw = left, +pitch = up)')
for nm, v in [('yaw', yaw[op]), ('pitch', pit[op])]:
    p = np.percentile(v, [1, 5, 25, 50, 75, 95, 99]); print(f'   {nm:5s} p1 {p[0]:+6.1f} p5 {p[1]:+6.1f} p25 {p[2]:+6.1f} median {p[3]:+6.1f} p75 {p[4]:+6.1f} p95 {p[5]:+6.1f} p99 {p[6]:+6.1f}')
inside = (np.abs(yaw[op]) <= 25) & (np.abs(pit[op]) <= 20)
print(f'   time inside headcal\'s ±25 x ±20 box: {inside.mean()*100:.1f}%  | vergence (R-L yaw) median {np.median(verg[op]):+.2f}°, spread p5-p95 {np.percentile(verg[op],5):+.1f}..{np.percentile(verg[op],95):+.1f}')
# fixations by velocity threshold
vel = np.hypot(np.diff(yaw), np.diff(pit)) / np.maximum(dt, 1e-3)
fix = (vel < 30) & op[1:] & op[:-1]
e2 = np.diff(np.r_[0, fix.astype(int), 0]); fs = np.where(e2 == 1)[0]; fe = np.where(e2 == -1)[0]
fd = (t[np.minimum(fe, len(t)-1)] - t[fs]) * 1000; keep = fd >= 100
print(f'== FIXATIONS (<30°/s, ≥100 ms): {keep.sum()} | duration median {np.median(fd[keep]):.0f} ms, p90 {np.percentile(fd[keep],90):.0f} ms | saccade-ish samples {(vel>100).mean()*100:.1f}%')
noise = [np.std(yaw[s:e]) for s, e, k in zip(fs, fe, keep) if k and e - s >= 10]
print(f'   precision inside fixations (yaw SD): median {np.median(noise):.2f}°  | frozen samples (gaze identical to previous while open): {(np.diff(yaw)[op[1:]]==0).mean()*100:.1f}%')
# face: what range did the expressions reach
face = [h for h in H[19:]]
# judge "does it fire" by the MAX and the time spent above 0.5 - a high percentile hides faces held < ~2 s
rate = 1 / np.median(dt)
mx = {h: (col(h).max(), (col(h) > 0.5).sum() / rate) for h in face}
top = sorted(mx.items(), key=lambda kv: -kv[1][0])
print('== FACE (Quest Pro: max, seconds above 0.5) top 25:'); print('   ' + ', '.join(f'{k} {m:.2f}/{s:.1f}s' for k, (m, s) in top[:25]))
print('   max never above 0.3: ' + ', '.join(k for k, (m, s) in top if m < 0.3))
# head pose + eye-head coordination
P = d / 'pose.csv'
if P.exists():
    pr = list(csv.reader(open(P))); ph = pr[0]; PA = np.array([[float(x) for x in r] for r in pr[1:] if len(r) == len(ph) and r[1] == '1'])
    pc = {n: i for i, n in enumerate(ph)}; pt = PA[:, pc['t_epoch']]; hy = np.unwrap(np.radians(PA[:, pc['yaw']])); hy = np.degrees(hy); hp = PA[:, pc['pitch']]
    print(f'== HEAD: {len(pt)} samples, {1/np.median(np.diff(pt)):.0f} Hz | yaw range p5..p95 {np.percentile(hy,5):+.0f}..{np.percentile(hy,95):+.0f}°, pitch p5..p95 {np.percentile(hp,5):+.0f}..{np.percentile(hp,95):+.0f}°')
    hyi = np.interp(t, pt, hy); hv = np.gradient(hyi, t); ev = np.gradient(yaw, t)
    turn = (np.abs(hv) > 30) & op & (np.abs(ev) < 300)
    if turn.sum() > 20:
        g = np.polyfit(hv[turn], ev[turn], 1)[0]
        print(f'   during head turns >30°/s ({turn.sum()} samples): eye-vs-head yaw velocity slope {g:+.2f} (≈ -1 = eyes counter-rotate = holding on the world, VOR; ≈0 = eyes ride with the head), corr {np.corrcoef(hv[turn], ev[turn])[0,1]:+.2f}')
    # eye position right after a big head turn: does gaze recentre?
    print(f'   eye yaw while head is still (<10°/s): median |yaw| {np.median(np.abs(yaw[op & (np.abs(hv)<10)])):.1f}°  vs during turns {np.median(np.abs(yaw[turn])):.1f}°')
