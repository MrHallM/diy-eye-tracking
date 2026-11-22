"""Keyed lens boss for the v13 ring: the ring's own (egg-shaped) bore outline, shrunk by a
checked inward offset, plus crush ribs. Shared by eye_rig_key_test.py (coupon) and
rig_plate.py (the real spectacle plate). Functions copied from eye_rig_key_test.py."""
import os, math
import Part
from FreeCAD import Vector as V

HERE = os.path.dirname(os.path.abspath(__file__))
# v2 (24 Oct, after key coupon 1 wouldn't go on): no rib inside the bulge (20-110 deg, where
# the ring's supports leave scars); the largest gap between ribs stays under 180 deg, so they
# still surround the ring
RIB_ANGLES = (0, 120, 190, 240, 290, 330)
BULGE_R = 21.0          # bore radius above this = the bulge
BULGE_MARGIN = 1.0      # the boss stops this far short of the bulge walls (support scars)
TAB_ANGLES = (140.0, 178.0)   # the inward tab beside the camera (measured R 18.7-19.4 at 153-163 deg, z 2.6-3.8)
TAB_RELIEF_TOP = 1.4    # boss height under the tab: 1.2 below its underside instead of 0.2
RIB_W = 1.0
RIB_LEAD = 0.5

def bore_points():
    """the ring's bore outline as (x, y) about the bore centre, 720 points."""
    import numpy as np
    rows = [line.split(",") for line in open(os.path.join(HERE, "v13_bore_profile_v2.csv")) if not line.startswith("#")]
    a = np.radians([float(r[0]) for r in rows]); rad = np.array([float(r[1]) for r in rows])
    assert len(rad) == 720, "profile should have 720 half-degree rows"
    return np.c_[rad * np.cos(a), rad * np.sin(a)]


def offset_inward(p, clearance, bulge_margin=None):
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
    if bulge_margin is not None:                            # keep clear of the bulge walls
        rb = np.hypot(p[:, 0], p[:, 1])                     # same half-degree bins as out
        inb = rb > BULGE_R
        out[inb] = np.minimum(out[inb], rb[inb] - bulge_margin)
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


def tab_relief(cx, cy, z_cut, z_top):
    """a sector prism to CUT from the boss under the camera-side tab (TAB_ANGLES)."""
    pts = [V(cx, cy, z_cut)]
    for k in range(int(TAB_ANGLES[0] * 2), int(TAB_ANGLES[1] * 2) + 1):
        a = math.radians(k / 2)
        pts.append(V(cx + 30 * math.cos(a), cy + 30 * math.sin(a), z_cut))
    return Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(V(0, 0, z_top - z_cut + 1))


def boss(cx, cy, clearance, z0, z1, bulge_margin=None):
    b, d = offset_inward(bore_points(), clearance, bulge_margin)
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


FLANK_BUMP_R = 0.6       # Ø1.2 vertical pins (>= 0.8 print minimum)
FLANK_LEAD = 0.5         # 45 deg cone on top so the ring starts on


def flank_points():
    """Where the bulge's two side walls are steepest (the walls that stop rotation), as
    (point on the bore wall, outward unit normal into the ring) about the bore centre."""
    import numpy as np
    p = bore_points()
    r = np.hypot(p[:, 0], p[:, 1])
    dr = np.roll(r, -1) - np.roll(r, 1)                     # radius change per bin
    lo = int(np.argmax(np.where(np.arange(720) < 240, dr, -1e9)))   # rising flank (bulge starts, ~20-30 deg)
    hi = int(np.argmin(np.where(np.arange(720) < 360, dr, 1e9)))    # falling flank (bulge ends, ~100-110 deg)
    out = []
    for k in (lo, hi):
        t = p[(k + 1) % 720] - p[(k - 1) % 720]
        n = np.array([t[1], -t[0]]); n /= np.linalg.norm(n)
        if np.dot(n, p[k]) < 0:                             # point OUT of the hole, into the ring
            n = -n
        out.append((k / 2.0, p[k], n))
    return out


def flank_bumps(cx, cy, squash, z0, z1):
    """Two vertical pins on the boss's bulge side walls, each pressing `squash` into the ring's
    flank wall, so turning either way squashes one (10-24: it rotated until it hit the wall)."""
    s = None
    for deg, pt, n in flank_points():
        c = pt - n * (FLANK_BUMP_R - squash)                # pin edge reaches squash past the wall
        pin = Part.makeCylinder(FLANK_BUMP_R, z1 - z0 - FLANK_LEAD, V(cx + c[0], cy + c[1], z0))
        pin = pin.fuse(Part.makeCone(FLANK_BUMP_R, FLANK_BUMP_R - FLANK_LEAD * 0.999, FLANK_LEAD,
                                     V(cx + c[0], cy + c[1], z1 - FLANK_LEAD)))
        print("FLANK    bump at %.1f deg, bore R %.2f, squash %.2f" % (deg, (pt[0] ** 2 + pt[1] ** 2) ** 0.5, squash))
        s = pin if s is None else s.fuse(pin)
    return s


def cyl(r, x, y, z0, z1):
    return Part.makeCylinder(r, z1 - z0, V(x, y, z0))


