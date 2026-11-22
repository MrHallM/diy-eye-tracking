"""Quick look at STLs without Blender: matplotlib, flat-shaded. Run with FreeCADCmd:
    freecadcmd.exe render_stl.py   (renders every stl/*.stl to render/<name>.png)"""
import os, glob
import Mesh
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "render"), exist_ok=True)
light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
for path in sorted(glob.glob(os.path.join(HERE, "stl", "*.stl"))):
    m = Mesh.Mesh(path)
    tris = np.array([[list(p) for p in f.Points] for f in m.Facets])
    nrm = np.array([list(f.Normal) for f in m.Facets])
    shade = 0.35 + 0.65 * np.clip(nrm @ light, 0, 1)
    fig = plt.figure(figsize=(8, 5), dpi=110)
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(tris, facecolors=np.c_[shade * 0.85, shade * 0.9, shade, np.ones(len(shade))], linewidths=0))
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=35, azim=-60); ax.set_axis_off()
    ax.set_title(os.path.basename(path))
    out = os.path.join(HERE, "render", os.path.basename(path)[:-4] + ".png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("RENDERED", out)
