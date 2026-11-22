"""
kptclick.py - click mouth keypoints on Babble mouth-camera frames, and compare two people's (or Claude's) clicks.

    python tools/kptclick.py label tests/KPT_2026-10-22               click the points, saves human_points.json
    python tools/kptclick.py compare tests/KPT_2026-10-22             human vs Claude: pixel distance per point

Why (10-22): the Babble Quest-Pro teacher could also learn mouth keypoints (an idea from Shinra Meisin's mouth
model). Before anyone labels hundreds of frames, test whether Claude can place the points well enough to pre-fill
them: Claude placed its points first (claude_points.json, from gridded enlargements), a human clicks the same frames
here without seeing them, then `compare`.

label: the frame is shown 3x. Click the point named at the top. Keys: Z = undo, S = skip (can't see it),
Enter/Space = next frame once all points are done, Q = save and quit. Saved after every frame (resumable: frames
already done are skipped).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

POINTS = [
    "mouth corner, image LEFT side",
    "mouth corner, image RIGHT side",
    "upper lip (nose side): outer edge, middle",
    "lower lip (chin side): outer edge, middle",
    "chin tip: bottom edge of the chin",
]
COLORS = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255)]
Z = 3


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def label(folder):
    out = folder / "human_points.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    frames = [f for f in sorted((folder / "frames").glob("f*.png")) if f.name not in data]
    win = "keypoints - Z undo, S skip, Enter next, Q quit"
    cv2.namedWindow(win)
    pts = []

    def click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(POINTS):
            pts.append([round(x / Z, 1), round(y / Z, 1)])
    cv2.setMouseCallback(win, click)
    for n, f in enumerate(frames, 1):
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        base = cv2.cvtColor(cv2.resize(g, None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        pts.clear()
        while True:
            im = base.copy()
            for i, p in enumerate(pts):
                if p is not None:
                    cv2.circle(im, (int(p[0] * Z), int(p[1] * Z)), 5, COLORS[i], -1)
            msg = (f"{n}/{len(frames)}  click: {POINTS[len(pts)]}" if len(pts) < len(POINTS)
                   else f"{n}/{len(frames)}  done - Enter for next frame (Z undo)")
            cv2.rectangle(im, (0, 0), (im.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(im, msg, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow(win, im)
            k = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1 or k == ord("q"):
                print(f"saved {len(data)} frames -> {out}")
                return
            if k == ord("z") and pts:
                pts.pop()
            elif k == ord("s") and len(pts) < len(POINTS):
                pts.append(None)
            elif k in (13, 32) and len(pts) == len(POINTS):
                data[f.name] = list(pts)
                save(out, data)
                break
    cv2.destroyAllWindows()
    print(f"all {len(data)} frames done -> {out}. Next: python tools/kptclick.py compare {folder}")


def compare(folder, a="human_points.json", b="claude_points.json"):
    A, B = json.loads((folder / a).read_text()), json.loads((folder / b).read_text())
    frames = sorted(k for k in A if k in B)
    d = np.full((len(frames), len(POINTS)), np.nan)
    for i, f in enumerate(frames):
        for j in range(len(POINTS)):
            if A[f][j] is not None and B[f][j] is not None:
                d[i, j] = np.hypot(A[f][j][0] - B[f][j][0], A[f][j][1] - B[f][j][1])
    print(f"{a} vs {b}, {len(frames)} frames, distance in px of the 240x240 image:")
    for j, name in enumerate(POINTS):
        col = d[:, j][~np.isnan(d[:, j])]
        print(f"  {name:45s} mean {col.mean():5.1f}  median {np.median(col):5.1f}  max {col.max():5.1f}")
    print(f"  {'ALL':45s} mean {np.nanmean(d):5.1f}  median {np.nanmedian(d):5.1f}")
    tiles = []
    for f in frames:            # sheet: filled = A, ring = B, same colour per point
        im = cv2.cvtColor(cv2.resize(cv2.imread(str(folder / "frames" / f), 0), None, fx=2, fy=2), cv2.COLOR_GRAY2BGR)
        for j in range(len(POINTS)):
            if A[f][j] is not None:
                cv2.circle(im, (int(A[f][j][0] * 2), int(A[f][j][1] * 2)), 4, COLORS[j], -1)
            if B[f][j] is not None:
                cv2.circle(im, (int(B[f][j][0] * 2), int(B[f][j][1] * 2)), 7, COLORS[j], 2)
        tiles.append(im)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(folder / "compare.png"), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))
    save(folder / "compare.json", {"frames": frames, "points": POINTS, "dist_px": d.tolist()})
    print(f"sheet -> {folder / 'compare.png'} (dot = {a}, ring = {b})")


if __name__ == "__main__":
    {"label": label, "compare": compare}[sys.argv[1]](Path(sys.argv[2]))
