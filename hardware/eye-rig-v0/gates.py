"""Print gates shared by the eye-rig scripts (copied from an earlier print-case project
where they were tested on known shapes).
Import from a FreeCADCmd script after putting this folder on sys.path."""
import Mesh, MeshPart, os, math
from FreeCAD import Vector as V

MIN_FEATURE = 0.80
THIN_PX = 0.05
THIN_DISC = 0.75


def overhang(shape, z_bed):
    """area (mm2) of downward faces flatter than 45 deg from vertical and not on the bed,
    plus the flattest downward face found (deg from vertical)."""
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
    """slice every layer, raster it, open it with a THIN_DISC disc; whatever vanishes is
    too narrow to print. Returns [(z above the bed, mm2, x, y)]."""
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


def save(shape, path):
    """watertight gate, then write the STL."""
    m = Mesh.Mesh(shape.tessellate(0.02))
    name = os.path.basename(path)
    assert m.isSolid() and not m.hasNonManifolds() and not m.hasSelfIntersections(), \
        "%s: mesh is not watertight - the slicer will drop walls" % name
    assert abs(m.Volume - shape.Volume) <= 0.01 * shape.Volume, \
        "%s: mesh volume %.0f != solid %.0f - cracked tessellation" % (name, m.Volume, shape.Volume)
    m.write(path)


def check_and_save(shape, path, z_bed=0.0):
    """every gate is a hard stop; prints one line per gate."""
    name = os.path.basename(path)
    assert shape.isValid(), "%s is not a valid solid" % name
    bad, worst = overhang(shape, z_bed)
    print("OVERHANG %-26s >45 deg: %.2f mm2 (flattest downward face %.1f deg)" % (name, bad, worst))
    assert bad < 0.5, "%s needs support" % name
    thin = thin_features(shape, z_bed)
    print("THIN     %-26s features a %.2f disc can't fit: %d" % (name, THIN_DISC, len(thin)))
    for zt, a, x, y in thin[:6]:
        print("      z %.2f, %.2f mm2 at (%.1f, %.1f)" % (zt, a, x, y))
    assert not thin, "%s has walls under %.1f mm" % (name, MIN_FEATURE)
    save(shape, path)
    bb = shape.BoundBox
    print("SAVED    %-26s %.1f x %.1f x %.1f mm, %.2f cm3" % (name, bb.XLength, bb.YLength, bb.ZLength, shape.Volume / 1000))
