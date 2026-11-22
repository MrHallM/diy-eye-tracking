"""Eye rig v0 - SPECTACLE PLATE (24 Oct 2026, draft). Run with FreeCADCmd; read the output for
"ALL GATES PASSED" (FreeCADCmd exits 0 even when an assert fails).

Two keyed lens bosses with crush ribs (lensboss.py), IPD 64 apart, for the v13 Left / Right
rings, on one flat plate with Ø34 view holes. Prints flat, bosses up (the eye side).
Geometry: the plate is the "lens plane". The left ring's file is already in this frame
(lens-side face z = 0, eye toward +z, the left eye at -x seen from the eye side), so the
left boss uses the measured profile as is and the right boss is its mirror (the v13 Right
STL is the exact mirror of the Left).

A tab at the top centre takes one M4 bolt to the eye-relief arm (next part, not drawn yet).
RIB_SQUASH is a placeholder until the lens_key_coupon result: 0.05 or 0.15.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Part
from FreeCAD import Vector as V
import gates, lensboss as lb

OUT = os.path.join(HERE, "stl")
os.makedirs(OUT, exist_ok=True)

IPD = 64.0
CLEARANCE = 0.20                 # keyed body (easy fit, coupon 1)
# 24 Oct, TESTED: key coupon v1, post 2 = zero wiggle, zero rotation, with a ring printed on
# Arachne (full collar). So the plate uses v1 exactly: 0.15 squash, v1 rib angles, the boss
# filling the whole bulge, no tab relief. (v2's relief + bulge margin were fixes for a ring
# whose collar had been dropped by the classic wall generator.)
RIB_SQUASH = 0.15
RIB_ANGLES_V1 = (0, 60, 130, 190, 280, 325)
BOSS_H = 2.4
PLATE_T = 4.0
FLANGE_R = 26.0
VIEW_D = 34.0
TAB_W, TAB_Y0, TAB_Y1 = 24.0, 20.0, 46.0
M4_CLEAR = 4.4
TAB_HOLE_Y = 38.0


def keyed_boss_at_origin():
    s = lb.boss(0, 0, CLEARANCE, PLATE_T - 0.5, PLATE_T + BOSS_H)      # 0.5 overlap, never tangent
    for a in RIB_ANGLES_V1:
        s = s.fuse(lb.rib(0, 0, a, RIB_SQUASH, PLATE_T - 0.5, PLATE_T + BOSS_H))
    return s


left = keyed_boss_at_origin()
right = left.mirror(V(0, 0, 0), V(1, 0, 0))          # mirror across the YZ plane
left.translate(V(-IPD / 2, 0, 0))
right.translate(V(IPD / 2, 0, 0))

plate = lb.cyl(FLANGE_R, -IPD / 2, 0, 0, PLATE_T).fuse(lb.cyl(FLANGE_R, IPD / 2, 0, 0, PLATE_T))
plate = plate.fuse(Part.makeBox(TAB_W, TAB_Y1 - TAB_Y0 + 6, PLATE_T, V(-TAB_W / 2, TAB_Y0 - 6, 0)))
plate = plate.fuse(left).fuse(right)
for x in (-IPD / 2, IPD / 2):
    plate = plate.cut(lb.cyl(VIEW_D / 2, x, 0, -1, PLATE_T + BOSS_H + 1))
plate = plate.cut(lb.cyl(M4_CLEAR / 2, 0, TAB_HOLE_Y, -1, PLATE_T + 1))
plate = plate.removeSplitter()

gates.check_and_save(plate, os.path.join(OUT, "rig_plate_v0.stl"))
print("IPD %.1f, clearance %.2f, rib squash %.2f (placeholder)" % (IPD, CLEARANCE, RIB_SQUASH))
print("ALL GATES PASSED")
