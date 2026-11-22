"""Eye rig v0 - KEYED lens-boss coupon v2 (24 Oct 2026). Run with FreeCADCmd; read the
output for "ALL GATES PASSED".

Key coupon 1 wouldn't go on: the camera side (beside the ring's inward tab at ~153-163 deg)
stayed up while the rest seated. Two causes, both in the ring's printed overhangs:
- the tab's underside sat only 0.2 above the boss top (and was a supported overhang), and
- the boss filled the bulge (20-110 deg), which is a roofed pocket the slicer fills with tree
  supports, so scars there block it.
v2 (lensboss.py): the boss drops to 1.4 tall under the tab, stops 1.0 short of the bulge
walls, and no rib sits in the bulge. Squash 0.05 (1 dot) / 0.10 (2 dots).
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
RIB_SQUASH = (0.05, 0.10)
BOSS_H = 2.4
PLATE_T = 3.0
FLANGE_R = 26.0
PITCH = 56.0
BOSS_ID = 34.0
DOT_D, DOT_DEPTH = 1.6, 0.6

centres = [FLANGE_R + i * PITCH for i in range(len(RIB_SQUASH))]
coupon = Part.makeBox(PITCH * (len(RIB_SQUASH) - 1), 6.0, PLATE_T, V(centres[0], -3.0, 0))
for cx in centres:
    coupon = coupon.fuse(lb.cyl(FLANGE_R, cx, 0, 0, PLATE_T))
for cx, sq in zip(centres, RIB_SQUASH):
    b = lb.boss(cx, 0, CLEARANCE, PLATE_T - 0.5, PLATE_T + BOSS_H, bulge_margin=lb.BULGE_MARGIN)
    for a in lb.RIB_ANGLES:
        b = b.fuse(lb.rib(cx, 0, a, sq, PLATE_T - 0.5, PLATE_T + BOSS_H))
    b = b.cut(lb.tab_relief(cx, 0, PLATE_T + lb.TAB_RELIEF_TOP, PLATE_T + BOSS_H))
    coupon = coupon.fuse(b)
for i, cx in enumerate(centres):
    coupon = coupon.cut(lb.cyl(BOSS_ID / 2, cx, 0, -1, PLATE_T + BOSS_H + 1))
    for dx in ((0.0,), (-2.5, 2.5))[i]:
        coupon = coupon.cut(lb.cyl(DOT_D / 2, cx + dx, -23.0, PLATE_T - DOT_DEPTH, PLATE_T + 1))
coupon = coupon.removeSplitter()

gates.check_and_save(coupon, os.path.join(OUT, "lens_key_coupon_v2.stl"))
print("ALL GATES PASSED")
