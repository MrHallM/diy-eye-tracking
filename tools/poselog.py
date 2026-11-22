"""
poselog.py - record the headset's pose from SteamVR with a timestamp, so eye data can be compared against head motion
(VOR scale check, head-speed gating of the centre prior, head-moving calibration; see chapters/drift-calibration.md
"Centre prior from head pose and gaze statistics").

    python tools/poselog.py --out tests/<run>/pose.csv                    (250 Hz until Ctrl+C)
    python tools/poselog.py --rate 500 --seconds 60 --out ...
    python tools/poselog.py --selftest                                    (checks the angle maths, no SteamVR)

Columns: t_epoch, valid, x/y/z (m, standing universe), qw/qx/qy/qz, yaw/pitch/roll (deg, yaw left +, pitch up +,
roll right-ear-down +), wx/wy/wz (deg/s, HEAD frame: wx = pitch rate, wy = yaw rate, wz = roll rate), speed (deg/s).

Timestamps are time.time(), the SAME clock as osclog.py / eyerec.py / a1grid.py, so the CSVs line up. 🔑 The camera
is ~42 ms behind (F2, tools/latency.py), so to compare eye vs head motion, shift the HEAD data later by the measured
lag (or the eye data earlier) during analysis. Nothing in the live tracking is slowed down.

Poses are read with prediction 0 s = SteamVR's estimate of "now". ⚠️ openvr.init LAUNCHES SteamVR when it isn't
running, so this refuses unless vrserver.exe is already up (same guard as a1grid.py).
"""
import argparse
import csv
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def angles(m):
    """3x4 or 3x3 rotation (OpenVR: x right, y up, -z forward) -> yaw, pitch, roll in degrees."""
    r = np.asarray(m, dtype=float)[:3, :3]
    fwd = -r[:, 2]                                   # where the face points
    yaw = math.degrees(math.atan2(-fwd[0], -fwd[2]))  # 0 = facing -z, + = turned left
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[1]))))
    # roll: head's up vector vs the up vector it would have with zero roll
    up = r[:, 1]
    flat_right = np.cross(fwd, [0.0, 1.0, 0.0])
    n = np.linalg.norm(flat_right)
    if n < 1e-9:
        return yaw, pitch, 0.0
    flat_right /= n
    zero_up = np.cross(flat_right, fwd)
    roll = math.degrees(math.atan2(float(np.dot(up, flat_right)), float(np.dot(up, zero_up))))
    return yaw, pitch, roll


def quat(m):
    r = np.asarray(m, dtype=float)[:3, :3]
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return w, x, y, z


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def selftest():
    ok = True
    # OpenVR yaw about +y (left turn = +), pitch about +x (look up = +), roll about -z (forward axis) right-ear-down = +
    for want in [(0, 0, 0), (30, 0, 0), (-50, 0, 0), (0, 20, 0), (0, -35, 0), (0, 0, 15), (25, 10, -20), (170, -5, 5)]:
        y, p, r = want
        m = rot("y", y) @ rot("x", p) @ rot("z", -r)
        got = angles(m)
        err = max(abs((g - w + 180) % 360 - 180) for g, w in zip(got, want))
        ok &= err < 1e-6
        print(f"  want {want} got ({got[0]:.3f}, {got[1]:.3f}, {got[2]:.3f}) {'ok' if err < 1e-6 else 'FAIL'}")
    q = quat(rot("y", 90))
    ok &= abs(q[0] - math.sqrt(0.5)) < 1e-9 and abs(q[2] - math.sqrt(0.5)) < 1e-9
    print("selftest", "passed" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", help="csv path (default pose_<time>.csv here)")
    ap.add_argument("--rate", type=float, default=250.0, help="samples per second (default 250)")
    ap.add_argument("--seconds", type=float, help="stop after this long")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    if b"vrserver.exe" not in subprocess.run(["tasklist", "/FI", "IMAGENAME eq vrserver.exe"],
                                             capture_output=True).stdout:
        sys.exit("SteamVR isn't running (openvr.init would launch it): start SteamVR first")
    import openvr
    try:
        openvr.init(openvr.VRApplication_Background)
    except openvr.OpenVRError as e:
        sys.exit(f"openvr.init failed: {e}")
    sysvr = openvr.VRSystem()
    hmd = openvr.k_unTrackedDeviceIndex_Hmd

    out = Path(a.out or f"pose_{datetime.now():%Y%m%d_%H%M%S}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = bad = 0
    gaps = []
    period = 1.0 / a.rate
    t0 = time.time()
    t_next = time.perf_counter()
    last = None
    print(f"logging HMD pose at {a.rate:g} Hz -> {out} (Ctrl+C stops)")
    try:
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_epoch", "valid", "x", "y", "z", "qw", "qx", "qy", "qz", "yaw", "pitch", "roll",
                        "wx", "wy", "wz", "speed"])
            while a.seconds is None or time.time() - t0 < a.seconds:
                poses = sysvr.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0.0,
                                                              openvr.k_unMaxTrackedDeviceCount)
                t = time.time()
                p = poses[hmd]
                valid = int(bool(p.bPoseIsValid))
                m = np.array([[p.mDeviceToAbsoluteTracking[i][j] for j in range(4)] for i in range(3)])
                yaw, pitch, roll = angles(m)
                qw, qx, qy, qz = quat(m)
                wv = np.degrees(np.array([p.vAngularVelocity[i] for i in range(3)]))
                wh = m[:3, :3].T @ wv                     # world -> head frame
                w.writerow([f"{t:.6f}", valid, f"{m[0, 3]:.5f}", f"{m[1, 3]:.5f}", f"{m[2, 3]:.5f}",
                            f"{qw:.6f}", f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}",
                            f"{yaw:.3f}", f"{pitch:.3f}", f"{roll:.3f}",
                            f"{wh[0]:.2f}", f"{wh[1]:.2f}", f"{-wh[2]:.2f}", f"{np.linalg.norm(wh):.2f}"])
                n += 1
                bad += not valid
                if last is not None:
                    gaps.append(t - last)
                last = t
                t_next += period
                sleep = t_next - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    t_next = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        openvr.shutdown()
    dur = time.time() - t0
    g = np.array(gaps) * 1000 if gaps else np.array([0.0])
    summ = (f"poselog {datetime.fromtimestamp(t0):%Y-%m-%d %H:%M:%S} · {dur:.0f} s · {n} samples "
            f"({n / max(dur, 1e-9):.1f} Hz) · invalid {bad}\n"
            f"sample interval ms: median {np.median(g):.2f} · p99 {np.percentile(g, 99):.2f} · max {g.max():.1f}\n")
    out.with_suffix(".summary.txt").write_text(summ, encoding="utf-8")
    print(summ, end="")


if __name__ == "__main__":
    sys.exit(main())
