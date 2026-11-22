"""Eye rig v0 - KEYED lens-boss coupon v3 (24 Oct 2026). Run with FreeCADCmd; read the output for
"ALL GATES PASSED".

Key coupon v1 post 2 (0.15 rib squash, full bulge, no tab relief) gave zero wiggle with a ring
printed on Arachne, but it still turned "ever so slightly, until it hits the wall": the bulge's
side walls have the 0.20 body clearance and the ribs only push outward. v3 = that post plus two
pins on the bulge side walls, pressing sideways into the ring's flank walls (lensboss.flank_bumps):
0.10 (1 dot) / 0.15 (2 dots).
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Part
from FreeCAD import Vector as V
import gates, lensboss as lb

OUT = os.path.join(HERE, "stl")
os.makedirs(OUT, exist_ok=True)

CLEARANCE = 0.20
RIB_SQUASH = 0.15
RIB_ANGLES_V1 = (0, 60, 130, 190, 280, 325)
FLANK_SQUASH = (0.10, 0.15)
BOSS_H = 2.4
PLATE_T = 3.0
FLANGE_R = 26.0
PITCH = 56.0
BOSS_ID = 34.0
DOT_D, DOT_DEPTH = 1.6, 0.6

centres = [FLANGE_R + i * PITCH for i in range(len(FLANK_SQUASH))]
coupon = Part.makeBox(PITCH * (len(FLANK_SQUASH) - 1), 6.0, PLATE_T, V(centres[0], -3.0, 0))
for cx in centres:
    coupon = coupon.fuse(lb.cyl(FLANGE_R, cx, 0, 0, PLATE_T))
for cx, fs in zip(centres, FLANK_SQUASH):
    b = lb.boss(cx, 0, CLEARANCE, PLATE_T - 0.5, PLATE_T + BOSS_H)
    for a in RIB_ANGLES_V1:
        b = b.fuse(lb.rib(cx, 0, a, RIB_SQUASH, PLATE_T - 0.5, PLATE_T + BOSS_H))
    b = b.fuse(lb.flank_bumps(cx, 0, fs, PLATE_T - 0.5, PLATE_T + BOSS_H))
    coupon = coupon.fuse(b)
for i, cx in enumerate(centres):
    coupon = coupon.cut(lb.cyl(BOSS_ID / 2, cx, 0, -1, PLATE_T + BOSS_H + 1))
    for dx in ((0.0,), (-2.5, 2.5))[i]:
        coupon = coupon.cut(lb.cyl(DOT_D / 2, cx + dx, -23.0, PLATE_T - DOT_DEPTH, PLATE_T + 1))
coupon = coupon.removeSplitter()

gates.check_and_save(coupon, os.path.join(OUT, "lens_key_coupon_v3.stl"))
print("ALL GATES PASSED")
