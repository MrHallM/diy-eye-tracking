"""Search print orientations of the v13 ring for the least support. Run with the training venv:
    training\\.venv\\Scripts\\python.exe -E -P ring_orient_search.py <ring.stl>
For each candidate "up" direction the mesh is rotated, dropped onto the bed, and scored by the
area of downward faces steeper than 45 deg from vertical that are NOT on the bed (= what the
slicer would support), plus the area resting on the bed (adhesion). Also reports the overhang
that falls inside the parts that matter: the lens-side bore band (bore radius < 26, ring z < 3)."""
import sys
import numpy as np
import trimesh

m = trimesh.load(sys.argv[1])
BORE_C = np.array([-37.90, -40.20])        # left ring; z = 0 lens-side face
tri = m.triangles                          # (n, 3, 3)
area = m.area_faces
cen = m.triangles_center
d = cen[:, :2] - BORE_C
in_band = (np.hypot(*d.T) < 26) & (cen[:, 2] < 3.0)


def score(up):
    up = up / np.linalg.norm(up)
    n = m.face_normals @ up                 # component along the new "up"
    h = (cen - cen.mean(0)) @ up
    zmin = (m.vertices - cen.mean(0)) @ up
    zmin = zmin.min()
    hmin = (tri - cen.mean(0)) @ up         # per-vertex heights
    on_bed = (hmin.max(1) - zmin) < 0.05
    over = (n < -0.7072) & ~on_bed
    return area[over].sum(), area[on_bed & (n < -0.99)].sum(), area[over & in_band].sum()


# Fibonacci sphere of candidate up-directions, plus the two obvious ones
k = 4000
i = np.arange(k) + 0.5
phi = np.arccos(1 - 2 * i / k); th = np.pi * (1 + 5 ** 0.5) * i
dirs = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
dirs = np.vstack([[0, 0, 1], [0, 0, -1], dirs])
res = np.array([score(u) for u in dirs])
print("as designed (lens side down): support %.0f mm2, bed %.0f mm2, in lens band %.0f mm2" % tuple(res[0]))
print("flipped (eye side down):      support %.0f mm2, bed %.0f mm2, in lens band %.0f mm2" % tuple(res[1]))
order = np.argsort(res[:, 0])
print("best 8 orientations by support area (up = direction in the ring's file coords):")
for j in order[:8]:
    u = dirs[j]
    tilt = np.degrees(np.arccos(np.clip(u[2], -1, 1)))
    print("  up=(%+.2f %+.2f %+.2f) tilt %5.1f deg from as-designed: support %6.0f  bed %5.0f  in lens band %5.0f"
          % (u[0], u[1], u[2], tilt, *res[j]))
