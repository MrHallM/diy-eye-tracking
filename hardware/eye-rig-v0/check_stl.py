"""Run the print gates on someone else's STL (as it sits on the bed). FreeCADCmd only reads
script args via an env var, so:  set CHECK_STL=<path>  then  freecadcmd.exe check_stl.py
Prints the overhang + thin-wall results and where the thin spots are (z above the bed, x, y)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Mesh, Part
import gates

path = os.environ["CHECK_STL"]
m = Mesh.Mesh(path)
print("MESH     %s: %d facets, solid=%s, non-manifold=%s, self-intersect=%s"
      % (os.path.basename(path), m.CountFacets, m.isSolid(), m.hasNonManifolds(), m.hasSelfIntersections()))
sh = Part.Shape()
sh.makeShapeFromMesh(m.Topology, 0.05)
solid = Part.makeSolid(sh)
z_bed = solid.BoundBox.ZMin
thin = gates.thin_features(solid, z_bed)
print("THIN     features a %.2f mm disc can't fit: %d" % (gates.THIN_DISC, len(thin)))
by_z = {}
for zt, a, x, y in thin:
    by_z.setdefault(round(zt, 1), []).append((a, x, y))
for zt in sorted(by_z):
    pts = by_z[zt]
    print("  z %.1f: %d spots, %.1f mm2 total, e.g. (%.1f, %.1f)" % (zt, len(pts), sum(p[0] for p in pts), pts[0][1], pts[0][2]))
