"""Eye-tracking desk rig v0 - TEST COUPONS (24 Oct 2026). Run with FreeCADCmd:

    "<FreeCAD folder>\\bin\\freecadcmd.exe" eye_rig_test.py

FreeCADCmd exits 0 even when an assert fails, so read the output for "ALL GATES PASSED".

Two small prints that check the two mechanisms the full rig depends on, before any big part:

1. lens_boss_coupon.stl - three stand-in "Quest 3 lenses" for the v13 ring to press
   onto. The ring's bore is NOT round (v13_bore_profile.csv, read from the STL: R
   19.65-20.6 over most of the lens band, plus a 29-99 deg cut-out, and inward tabs
   from z 2.6 up), so each boss is the bore's own outline shrunk by a clearance of
   0.10 / 0.20 / 0.30 mm (1 / 2 / 3 engraved dots), only 2.4 tall so it stays under
   the tabs. The ring only goes on one way round, as it would on the lens. Printed
   holes come out a little small, so one of these should be snug. That clearance
   goes into the rig. (A first version used round tubes OD 39.6-40.0: the ring's
   tightest points are R 19.65, so all three would have jammed.)
2. clamp_coupon.stl - a slotted bar and a block with an M4 clearance hole and a
   nut trap, i.e. one rig joint. Checks the M4 clearance, that the nut drops into
   its trap, and that one tightened bolt stops the block sliding.
"""
import Mesh, Part, MeshPart, os, math
from FreeCAD import Vector as V

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stl")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------ parameters (mm)
CLEARANCES = (0.10, 0.20, 0.30)   # boss outline = ring bore outline minus this, all round. Centred on 0.20 from test prints: rigid part in a printed hole is snug at 0.10-0.14/side (Slimterface magnets, dongle board), both parts printed here so the errors add; 0.45 flopped (Babble hinge)
BOSS_CAP_R = 21.0       # in the bore's 29-99 deg cut-out (R ~24) the boss stops at this radius
BOSS_ID = 34.0          # view hole - the rig's real plate keeps this so you can see through
BOSS_H = 2.4            # the bore's inward tabs start at z 2.6 above the ring's lens-side face
PLATE_T = 3.0
FLANGE_R = 25.0         # the ring's base seats on this
PITCH = 52.0            # boss spacing: the v13 ring is 52.8 across, so at 46 a ring on one boss hit the next tube (seen in Blender)
DOT_D, DOT_DEPTH = 1.6, 0.6

M4_CLEAR = 4.4
NUT_AF, NUT_T = 7.3, 3.4    # M4 nut is 7.0 across flats, 3.2 thick
BAR_L, BAR_W, BAR_T = 60.0, 14.0, 5.0
SLOT_L = 30.0
BLOCK_L, BLOCK_W, BLOCK_T = 20.0, 14.0, 8.0

MIN_FEATURE = 0.80
THIN_PX = 0.05
THIN_DISC = 0.75


def box(x0, y0, z0, x1, y1, z1):
    return Part.makeBox(x1 - x0, y1 - y0, z1 - z0, V(x0, y0, z0))


def cyl(r, x, y, z0, z1):
    return Part.makeCylinder(r, z1 - z0, V(x, y, z0))


def hexagon(af, x, y, z0, z1):
    r = af / math.sqrt(3.0)                     # circumradius from across-flats
    pts = [V(x + r * math.cos(math.radians(30 + 60 * i)), y + r * math.sin(math.radians(30 + 60 * i)), z0)
           for i in range(6)]
    face = Part.Face(Part.makePolygon(pts + [pts[0]]))
    return face.extrude(V(0, 0, z1 - z0))


# ------------------------------------------------ coupon 1: lens bosses
def bore_profile():
    """Tightest bore radius per degree (0-359) from v13_bore_profile.csv; NaN = the cut-out."""
    rad = []
    with open(os.path.join(HERE, "v13_bore_profile.csv")) as f:
        for line in f:
            if line.startswith("#"):
                continue
            rad.append(float(line.split(",")[1]))
    assert len(rad) == 360, "bore profile should have 360 rows"
    return rad


def boss(cx, cy, clearance, z0, z1):
    pts = []
    for deg, r in enumerate(bore_profile()):
        r = BOSS_CAP_R if (r != r or r > BOSS_CAP_R) else r      # r != r is NaN
        r -= clearance
        pts.append(V(cx + r * math.cos(math.radians(deg)), cy + r * math.sin(math.radians(deg)), z0))
    face = Part.Face(Part.makePolygon(pts + [pts[0]]))
    return face.extrude(V(0, 0, z1 - z0))


n = len(CLEARANCES)
half = FLANGE_R
plate_w = PITCH * (n - 1) + 2 * half
# flange rings (the ring's base seats on these) joined by short bridges, not a solid
# plate: same test, about half the filament
coupon = box(half, -3.0, 0, half + PITCH * (n - 1), 3.0, PLATE_T)
for i in range(n):
    coupon = coupon.fuse(cyl(FLANGE_R, half + i * PITCH, 0, 0, PLATE_T))
for i, clr in enumerate(CLEARANCES):
    coupon = coupon.fuse(boss(half + i * PITCH, 0, clr, PLATE_T - 0.5, PLATE_T + BOSS_H))   # 0.5 overlap, never tangent
for i in range(n):
    cx = half + i * PITCH
    coupon = coupon.cut(cyl(BOSS_ID / 2, cx, 0, -1, PLATE_T + BOSS_H + 1))
    for dx in ((0.0,), (-2.5, 2.5), (-4.0, 0.0, 4.0))[i]:   # 1/2/3 engraved dots on the flange
        coupon = coupon.cut(cyl(DOT_D / 2, cx + dx, -22.3, PLATE_T - DOT_DEPTH, PLATE_T + 1))
coupon = coupon.removeSplitter()

# ------------------------------------------------ coupon 2: slot + nut-trap block
bar = box(0, 0, 0, BAR_L, BAR_W, BAR_T)
s0 = (BAR_L - SLOT_L) / 2
slot = box(s0 + M4_CLEAR / 2, BAR_W / 2 - M4_CLEAR / 2, -1, s0 + SLOT_L - M4_CLEAR / 2, BAR_W / 2 + M4_CLEAR / 2, BAR_T + 1)
slot = slot.fuse(cyl(M4_CLEAR / 2, s0 + M4_CLEAR / 2, BAR_W / 2, -1, BAR_T + 1))
slot = slot.fuse(cyl(M4_CLEAR / 2, s0 + SLOT_L - M4_CLEAR / 2, BAR_W / 2, -1, BAR_T + 1))
bar = bar.cut(slot).removeSplitter()
by0 = BAR_W + 6.0                                          # block sits beside the bar on the same plate
block = box(0, by0, 0, BLOCK_L, by0 + BLOCK_W, BLOCK_T)
bc = (BLOCK_L / 2, by0 + BLOCK_W / 2)
block = block.cut(cyl(M4_CLEAR / 2, bc[0], bc[1], -1, BLOCK_T + 1))
block = block.cut(hexagon(NUT_AF, bc[0], bc[1], BLOCK_T - NUT_T, BLOCK_T + 1))   # trap opens UPWARD: no ceiling over air
block = block.removeSplitter()
clamp = bar.fuse(block) if bar.common(block).Volume > 0 else Part.makeCompound([bar, block])


# ------------------------------------------------ gates (from the dongle case / dock)
def overhang(shape, z_bed):
    m = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.02, AngularDeflection=0.1)
    bad, worst = 0.0, 0.0
    for f in m.Facets:
        nz = f.Normal.z
        if nz >= -1e-6 or max(p[2] for p in f.Points) < z_bed + 0.01:
            continue
        a = math.degrees(math.asin(min(1.0, -nz)))
        worst = max(worst, a)
        if a > 45.0 + 1e-3:
            bad += f.Area
    return bad, worst


def thin_features(shape, z_bed, step=0.2, min_area=0.1):
    """Copied from an earlier print-case project: slice every layer,
    raster it, open with a THIN_DISC disc; whatever vanishes is too narrow to print."""
    import numpy as np
    from scipy import ndimage
    from matplotlib.path import Path
    bb = shape.BoundBox
    x0, y0 = bb.XMin - 1.0, bb.YMin - 1.0
    nx = int((bb.XLength + 2.0) / THIN_PX) + 1
    ny = int((bb.YLength + 2.0) / THIN_PX) + 1
    xs = x0 + (np.arange(nx) + 0.5) * THIN_PX
    ys = y0 + (np.arange(ny) + 0.5) * THIN_PX
    r = THIN_DISC / 2.0 / THIN_PX
    k = int(math.ceil(r))
    jj, ii = np.mgrid[-k:k + 1, -k:k + 1]
    disc = (ii * ii + jj * jj) <= r * r
    out = []
    z = z_bed + step / 2.0 + 0.013
    while z < bb.ZMax:
        mask = np.zeros((ny, nx), dtype=bool)
        for w in shape.slice(V(0, 0, 1), z):
            pts = np.array([(p.x, p.y) for p in w.discretize(Deflection=0.005)])
            ix0 = max(int((pts[:, 0].min() - x0) / THIN_PX) - 1, 0)
            ix1 = min(int((pts[:, 0].max() - x0) / THIN_PX) + 2, nx)
            iy0 = max(int((pts[:, 1].min() - y0) / THIN_PX) - 1, 0)
            iy1 = min(int((pts[:, 1].max() - y0) / THIN_PX) + 2, ny)
            gx, gy = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1])
            inside = Path(pts).contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
            mask[iy0:iy1, ix0:ix1] ^= inside.reshape(gx.shape)
        if mask.any():
            lost = mask & ~ndimage.binary_opening(mask, structure=disc)
            lab, nlab = ndimage.label(lost, structure=np.ones((3, 3)))
            for i in range(1, nlab + 1):
                sel = lab == i
                a = sel.sum() * THIN_PX * THIN_PX
                if a >= min_area:
                    cy, cx = ndimage.center_of_mass(sel)
                    out.append((z - z_bed, a, x0 + (cx + 0.5) * THIN_PX, y0 + (cy + 0.5) * THIN_PX))
        z += step
    return out


def save(shape, name):
    m = Mesh.Mesh(shape.tessellate(0.02))
    assert m.isSolid() and not m.hasNonManifolds() and not m.hasSelfIntersections(), \
        "%s: mesh is not watertight - the slicer will drop walls" % name
    assert abs(m.Volume - shape.Volume) <= 0.01 * shape.Volume, \
        "%s: mesh volume %.0f != solid %.0f - cracked tessellation" % (name, m.Volume, shape.Volume)
    m.write(os.path.join(OUT, name))


assert min(r for r in bore_profile() if r == r) - max(CLEARANCES) - BOSS_ID / 2 >= MIN_FEATURE, "boss wall under 0.8"
assert BOSS_H + PLATE_T <= 10, "boss coupon taller than planned"
for nm, part in (("lens_boss_coupon.stl", coupon), ("clamp_coupon.stl", clamp)):
    assert part.isValid(), "%s is not a valid solid" % nm
    bad, worst = overhang(part, 0.0)
    print("OVERHANG %-22s >45 deg: %.2f mm2 (flattest downward face %.1f deg)" % (nm, bad, worst))
    assert bad < 0.5, "%s needs support" % nm
    thin = thin_features(part, 0.0)
    print("THIN     %-22s features a %.2f disc can't fit: %d" % (nm, THIN_DISC, len(thin)))
    for zt, a, x, y in thin[:6]:
        print("      z %.2f, %.2f mm2 at (%.1f, %.1f)" % (zt, a, x, y))
    assert not thin, "%s has walls under %.1f mm" % (nm, MIN_FEATURE)
    save(part, nm)
    bb = part.BoundBox
    print("SAVED    %-22s %.1f x %.1f x %.1f mm, %.2f cm3" % (nm, bb.XLength, bb.YLength, bb.ZLength, part.Volume / 1000))
print("ALL GATES PASSED")
