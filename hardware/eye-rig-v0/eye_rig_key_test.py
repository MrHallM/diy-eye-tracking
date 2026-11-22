"""Eye rig v0 - KEYED lens-boss coupon (24 Oct 2026). Run with FreeCADCmd; read the output
for "ALL GATES PASSED" (FreeCADCmd exits 0 even when an assert fails).

Why: on lens_boss_coupon the ring could still rotate under force. The v13 bore is egg-shaped
(R ~20 on most sides, bulging to ~25 between ~20 and ~110 deg), but that coupon capped the
boss at R 21 in the bulge, so what was left was nearly round. Here the boss follows the whole
outline (v13_bore_profile_v2.csv, innermost ring material in the boss's 0.3-2.5 band),
shrunk with a true 2D offset so the steep sides of the bulge get the same clearance as the
rest. An egg in an egg can't turn.

Two bosses, both keyed at 0.20 (easy), each with 6 CRUSH RIBS that reach past the ring wall by
0.05 (1 dot) / 0.15 (2 dots): the ribs squash on assembly, so the ring is held from every side
with no play (10-24: at least 4 contact points).
"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Part
from FreeCAD import Vector as V
import gates

OUT = os.path.join(HERE, "stl")
os.makedirs(OUT, exist_ok=True)

CLEARANCES = (0.20, 0.20)        # keyed body: an easy fit (2 dots on the first coupon)
RIB_SQUASH = (0.05, 0.15)        # crush ribs reach this far PAST the ring's wall (10-24:
                                 # "at least 4 contact points so all sides are secure")
RIB_ANGLES = (0, 60, 130, 190, 280, 325)   # 6 ribs, kept >= 25 deg off the bore's tabs (~159, ~244)
RIB_W = 1.0                      # tangential width (>= 0.8 print minimum)
RIB_LEAD = 0.5                   # 45 deg lead-in at the rib top so the ring starts on easily
BOSS_ID = 34.0
BOSS_H = 2.4
PLATE_T = 3.0
FLANGE_R = 26.0          # wider than before: the bulge reaches R 25.1
PITCH = 56.0             # ring is 52.8 across
DOT_D, DOT_DEPTH = 1.6, 0.6


def bore_points():
    """the ring's bore outline as (x, y) about the bore centre, 720 points."""
    import numpy as np
    rows = [line.split(",") for line in open(os.path.join(HERE, "v13_bore_profile_v2.csv")) if not line.startswith("#")]
    a = np.radians([float(r[0]) for r in rows]); rad = np.array([float(r[1]) for r in rows])
    assert len(rad) == 720, "profile should have 720 half-degree rows"
    return np.c_[rad * np.cos(a), rad * np.sin(a)]


def offset_inward(p, clearance):
    """Move every outline point inward along its normal, then re-sample by angle taking the
    innermost hit per half degree (the outline is star-shaped about the centre). makeOffset2D
    on the 720-gon gave a cracked mesh, so this stays a plain polygon. Checked below by brute
    force: every boss point must be >= clearance from every bore point."""
    import numpy as np
    t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    n = np.c_[t[:, 1], -t[:, 0]]; n /= np.linalg.norm(n, axis=1)[:, None]
    n *= np.sign((n * -p).sum(1))[:, None]                  # point the normal at the centre
    q = p + n * clearance
    ang = (np.degrees(np.arctan2(q[:, 1], q[:, 0])) % 360 * 2).astype(int) % 720
    rad = np.hypot(q[:, 0], q[:, 1])
    out = np.full(720, np.inf)
    np.minimum.at(out, ang, rad)
    for k in np.where(~np.isfinite(out))[0]:                # empty bin: take the neighbour
        out[k] = min(out[(k - 1) % 720], out[(k + 1) % 720])
    # the per-bin minimum is saw-toothed near the tabs (~155 deg); its outward teeth printed
    # as 0.1 mm2 slivers in the thin gate. A +-1 deg minimum filter only ever SHRINKS the
    # boss, so it removes the teeth without eating into the clearance.
    out = np.min([np.roll(out, s) for s in range(-2, 3)], axis=0)
    a = np.radians(np.arange(720) / 2)
    b = np.c_[out * np.cos(a), out * np.sin(a)]
    # a boss point may never come closer than the clearance to the bore outline
    d = np.sqrt(((b[:, None, :] - p[None, :, :]) ** 2).sum(-1)).min(1)
    while d.min() < clearance - 0.005:                      # shrink only the offending points
        bad = d < clearance - 0.005
        out[bad] -= 0.01
        b = np.c_[out * np.cos(a), out * np.sin(a)]
        d = np.sqrt(((b[:, None, :] - p[None, :, :]) ** 2).sum(-1)).min(1)
    return b, d


def boss(cx, cy, clearance, z0, z1):
    b, d = offset_inward(bore_points(), clearance)
    print("BOSS     clearance %.2f: nearest approach to the bore %.3f, mean %.3f" % (clearance, d.min(), d.mean()))
    pts = [V(cx + x, cy + y, z0) for x, y in b]
    return Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(V(0, 0, z1 - z0))


def bore_r(deg):
    p = bore_points()
    import numpy as np
    return float(np.hypot(*p[int(round(deg * 2)) % 720]))


def rib(cx, cy, deg, squash, z0, z1):
    """a radial rib from inside the boss out to (bore radius + squash), with a 45 deg
    chamfer on its outer top edge (an upward-facing slope, so no overhang)."""
    r1 = bore_r(deg) + squash
    r0 = r1 - 1.5                                   # buried 1.3+ mm in the boss body
    prof = [V(r0, -RIB_W / 2, z0), V(r1, -RIB_W / 2, z0), V(r1, -RIB_W / 2, z1 - RIB_LEAD),
            V(r1 - RIB_LEAD, -RIB_W / 2, z1), V(r0, -RIB_W / 2, z1)]
    s = Part.Face(Part.makePolygon(prof + [prof[0]])).extrude(V(0, RIB_W, 0))
    s.rotate(V(0, 0, 0), V(0, 0, 1), deg)
    s.translate(V(cx, cy, 0))
    return s


def cyl(r, x, y, z0, z1):
    return Part.makeCylinder(r, z1 - z0, V(x, y, z0))


centres = [FLANGE_R + i * PITCH for i in range(len(CLEARANCES))]
coupon = Part.makeBox(PITCH * (len(CLEARANCES) - 1), 6.0, PLATE_T, V(centres[0], -3.0, 0))
for cx in centres:
    coupon = coupon.fuse(cyl(FLANGE_R, cx, 0, 0, PLATE_T))
for cx, clr, sq in zip(centres, CLEARANCES, RIB_SQUASH):
    coupon = coupon.fuse(boss(cx, 0, clr, PLATE_T - 0.5, PLATE_T + BOSS_H))   # 0.5 overlap, never tangent
    for a in RIB_ANGLES:
        coupon = coupon.fuse(rib(cx, 0, a, sq, PLATE_T - 0.5, PLATE_T + BOSS_H))
for i, cx in enumerate(centres):
    coupon = coupon.cut(cyl(BOSS_ID / 2, cx, 0, -1, PLATE_T + BOSS_H + 1))
    for dx in ((0.0,), (-2.5, 2.5))[i]:          # dots at 270 deg, where the bore is R ~20
        coupon = coupon.cut(cyl(DOT_D / 2, cx + dx, -23.0, PLATE_T - DOT_DEPTH, PLATE_T + 1))
coupon = coupon.removeSplitter()

gates.check_and_save(coupon, os.path.join(OUT, "lens_key_coupon.stl"))
print("ALL GATES PASSED")
