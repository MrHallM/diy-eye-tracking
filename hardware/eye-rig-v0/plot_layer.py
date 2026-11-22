"""Draw one layer's extrusion moves from Bambu 3MF G-code, several files side by side.
FreeCADCmd (for matplotlib): set PLOT_Z=1.6 and PLOT_FILES=a.3mf;b.3mf;...  PLOT_OUT=out.png"""
import os, re, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z_want = float(os.environ.get("PLOT_Z", "1.6"))
files = os.environ["PLOT_FILES"].split(";")
fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 4.4), dpi=110, squeeze=False)
for ax, f in zip(axes[0], files):
    g = zipfile.ZipFile(f).read("Metadata/plate_1.gcode").decode("utf-8", "ignore").splitlines()
    z, x, y, segs = 0.0, 0.0, 0.0, []
    for l in g:
        m = re.match(r"; Z_HEIGHT: ([\d.]+)", l)
        if m:
            z = float(m.group(1)); continue
        if l.startswith("G1") or l.startswith("G0"):
            nx = re.search(r"X([-\d.]+)", l); ny = re.search(r"Y([-\d.]+)", l); ne = re.search(r"E([\d.]+)", l)
            px, py = x, y
            if nx: x = float(nx.group(1))
            if ny: y = float(ny.group(1))
            if ne and abs(z - z_want) < 0.01 and (nx or ny):
                segs.append(((px, py), (x, y)))
    for (a, b) in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], lw=0.8, color="k")
    ax.set_aspect("equal"); ax.set_title("%s\nz %.1f: %d moves" % (os.path.basename(f), z_want, len(segs)), fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(os.environ["PLOT_OUT"], bbox_inches="tight")
print("PLOTTED", os.environ["PLOT_OUT"])
