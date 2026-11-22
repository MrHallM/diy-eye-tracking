"""
b24_load.py - read Burlingham24's public data (github.com/facebookresearch/Reality-Labs-Egocentric-ET, CC-BY 4.0,
data/data.mat, sha256 d0145383...baafa) into numpy. Download it into EyeTrackingLab/data/burlingham24/ (git-ignored).

What it holds (getData.m in that repo): 169 recordings, Tobii Glasses at 50 Hz, gaze-in-head X/Y in degrees, a
fixation class per sample (1 = fixation), head angular speed from the IMU at 100 Hz, task name and subject id.
"""
from pathlib import Path

import h5py
import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "burlingham24" / "data.mat"


def _str(f, ref):
    a = np.array(f[ref]).ravel()
    if a.dtype == object:                          # a cell inside a cell: follow the reference
        return _str(f, a[0])
    return "".join(chr(int(c)) for c in a)


def load(path=DATA, fields=("GazeInHeadX", "GazeInHeadY", "Class", "HeadVelAngMag", "TimeGaze", "TimeIMU")):
    recs = []
    with h5py.File(path, "r") as f:
        t = f["TIMESERIES"]
        n = t["Class"].shape[0]
        for i in range(n):
            r = {"task": _str(f, t["Task"][i, 0]), "sub": _sub(f, t["SubID"][i, 0])}
            for k in fields:
                r[k] = np.array(f[t[k][i, 0]], dtype=float).ravel()
            recs.append(r)
    return recs


def _sub(f, ref):
    a = np.array(f[ref])
    if a.dtype == object:
        return _sub(f, a.ravel()[0])
    if a.dtype.kind in "ui" and a.size > 1:
        return "".join(chr(int(c)) for c in a.ravel())
    return str(a.ravel()[0])


if __name__ == "__main__":
    recs = load()
    print(len(recs), "recordings")
    tasks = {}
    for r in recs:
        tasks.setdefault(r["task"], set()).add(r["sub"])
    for k, v in sorted(tasks.items()):
        print(f"  {k}: {sum(1 for r in recs if r['task'] == k)} recordings, {len(v)} subjects")
    r = recs[0]
    for k in ("GazeInHeadX", "GazeInHeadY", "Class", "HeadVelAngMag", "TimeGaze", "TimeIMU"):
        v = r[k]
        print(f"  {k}: n={v.size} nan={np.isnan(v).mean():.2f} min={np.nanmin(v):.3f} med={np.nanmedian(v):.3f} "
              f"max={np.nanmax(v):.3f}")
    print("  Class values:", np.unique(r["Class"][~np.isnan(r["Class"])])[:10])
    print("  dt gaze:", np.nanmedian(np.diff(r["TimeGaze"])), " dt imu:", np.nanmedian(np.diff(r["TimeIMU"])))
    hours = sum(np.isfinite(r["GazeInHeadX"]).sum() for r in recs) / 50 / 3600
    print(f"  total valid gaze: {hours:.1f} h")
