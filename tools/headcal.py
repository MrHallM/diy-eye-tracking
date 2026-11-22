"""
headcal.py - one-dot head-movement calibration (CalibMe17-style, see chapters/drift-calibration.md): a dot stays FIXED
in the room, you keep looking at it and slowly move your head (left-right, up-down, circles). Every head pose then
says exactly where the dot is relative to your head = where your eyes point, so ~20 s gives thousands of
(eye image, true eye-in-head direction) pairs over the whole range. That is what learns SCALE and the vertical
offset, which the target-free centre prior can't (analysis/sim/README.md).

    python tools/headcal.py --out tests/HC_<date>/run1 --record http://<eye-board>/ --serve 8090 --osc 8888
    python tools/headcal.py --seconds 30 --dist 2.0 --bg dark ...
    python tools/headcal.py --selftest                       (geometry check, no SteamVR)

Tell the wearer BEFORE starting: "keep your eyes on the dot; move your head slowly, left-right, up-down, then circles;
not fast" (fast head motion = VOR + camera lag; slow keeps the ~42 ms camera lag small: at 20 deg/s it is ~0.8 deg).

Output: headcal.csv at the pose rate: t_epoch, head yaw/pitch/roll, head speed (deg/s), and the dot in HEAD space:
tyaw/tpitch (deg, Fick: yaw first; + = left/up from the head's forward) and tdist (m). Use these as the eye-in-head
truth (both eyes see the dot from slightly different places: at 2 m the left/right eye difference is ~0.9 deg, see
the vergence note in run.json). Frames/OSC land in the same folder on the same clock (eyerec.py / osclog.py).
At the end it prints coverage: the share of 5x5 deg cells in +-25 x +-20 deg that got samples (CalibMe17: coverage,
not sample count, drives accuracy; aim for > 85%).

⚠️ openvr.init LAUNCHES SteamVR when it isn't running, so this refuses unless vrserver.exe is up.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))
from poselog import angles  # noqa: E402

IPD_M = 0.064                                         # your IPD (64 mm is about average), for the vergence note only


def target_in_head(m, p_dot):
    """HMD pose (3x4, OpenVR: x right, y up, -z forward) + dot position (world) -> yaw, pitch (deg), distance (m).
    Fick angles in head space: yaw + = left, pitch + = up, like poselog's head angles."""
    r = np.asarray(m, float)[:3, :3]
    v = r.T @ (np.asarray(p_dot, float) - np.asarray(m, float)[:3, 3])
    d = float(np.linalg.norm(v))
    x, y, z = v / d                                   # head frame, forward = -z
    yaw = math.degrees(math.atan2(-x, -z))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, y))))
    return yaw, pitch, d


def coverage(yaws, pitches, h=25, v=20, cell=5):
    nx, ny = 2 * h // cell, 2 * v // cell
    hit = np.zeros((ny, nx), bool)
    for a, b in zip(yaws, pitches):
        i, j = int((b + v) // cell), int((a + h) // cell)
        if 0 <= i < ny and 0 <= j < nx:
            hit[i, j] = True
    return hit.mean(), hit


def selftest():
    from poselog import rot
    ok = True
    p_dot = np.array([0.0, 1.6, -2.0])                     # 2 m ahead at eye height
    for yaw, pitch in [(0, 0), (20, 0), (-30, 0), (0, 15), (0, -20), (25, 10)]:
        r = rot("y", yaw) @ rot("x", pitch)               # head turned LEFT by yaw, UP by pitch
        m = np.c_[r, [0.0, 1.6, 0.0]]
        ty, tp, d = target_in_head(m, p_dot)
        # head turned left by yaw -> the dot is to the head's RIGHT (negative); up -> dot is below
        # rebuild the ray from (tyaw, tpitch) in head space, rotate it into the world: it must hit the dot
        y_, p_ = math.radians(ty), math.radians(tp)
        v_head = np.array([-math.cos(p_) * math.sin(y_), math.sin(p_), -math.cos(p_) * math.cos(y_)])
        hit = m[:3, 3] + r @ v_head * d
        good = np.allclose(hit, p_dot, atol=1e-9) and abs(d - 2.0) < 1e-9
        if pitch == 0:
            good &= abs(ty + yaw) < 1e-6                  # pure turns: the dot moves the opposite way
        if yaw == 0:
            good &= abs(tp + pitch) < 1e-6
        ok &= good
        print(f"  head yaw {yaw:+d} pitch {pitch:+d} -> dot at yaw {ty:+.3f} pitch {tp:+.3f} dist {d:.3f} "
              f"{'ok' if good else 'FAIL'}")
    cov, _ = coverage([-22, 0, 22], [-17, 0, 17])
    ok &= abs(cov - 3 / 80) < 1e-9
    print("selftest", "passed" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--dist", type=float, default=2.0, help="dot distance in metres (far = less parallax)")
    ap.add_argument("--dot-deg", type=float, default=0.8, help="dot size in degrees")
    ap.add_argument("--bg", choices=["bright", "dark"], default="dark")
    ap.add_argument("--rate", type=float, default=250.0, help="pose samples per second")
    ap.add_argument("--record", help="also record raw frames from this board URL (eyerec.py)")
    ap.add_argument("--serve", type=int, help="with --record: re-serve the stream on 127.0.0.1:<port> for Baballonia")
    ap.add_argument("--osc", type=int, help="also log Baballonia's OSC output on this port (osclog.py)")
    ap.add_argument("--osc-forward", help="with --osc: re-send packets to host:port (VRCFT)")
    ap.add_argument("--out", help="run folder (default: headcal_<time> here)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    if b"vrserver.exe" not in subprocess.run(["tasklist", "/FI", "IMAGENAME eq vrserver.exe"],
                                             capture_output=True).stdout:
        sys.exit("SteamVR isn't running (openvr.init would launch it): start SteamVR with the headset on first")
    import cv2
    import openvr
    try:
        openvr.init(openvr.VRApplication_Overlay)
    except openvr.OpenVRError as e:
        sys.exit(f"SteamVR overlay failed ({type(e).__name__} {e})")
    sysvr, ov = openvr.VRSystem(), openvr.IVROverlay()
    hmd = openvr.k_unTrackedDeviceIndex_Hmd

    def pose():
        p = sysvr.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0.0,
                                                  openvr.k_unMaxTrackedDeviceCount)[hmd]
        return bool(p.bPoseIsValid), np.array([[p.mDeviceToAbsoluteTracking[i][j] for j in range(4)]
                                                for i in range(3)])

    valid, m0 = pose()
    if not valid:
        openvr.shutdown(); sys.exit("no valid headset pose: is the headset tracked?")
    # dot: a.dist ahead of the starting head position, LEVEL (horizontal forward only), facing the wearer
    fwd = -m0[:3, 2].copy(); fwd[1] = 0; fwd /= np.linalg.norm(fwd)
    p_dot = m0[:3, 3] + a.dist * fwd
    face = -fwd                                           # quad normal (+z of the overlay) points back at the wearer
    right = np.cross([0.0, 1.0, 0.0], face); right /= np.linalg.norm(right)
    up = np.cross(face, right)
    md = openvr.HmdMatrix34_t()
    for i in range(3):
        md.m[i][0], md.m[i][1], md.m[i][2], md.m[i][3] = right[i], up[i], face[i], p_dot[i]

    tmp = Path(tempfile.mkdtemp())
    bg, fg = (235, 25) if a.bg == "bright" else (10, 200)
    img = np.zeros((256, 256, 4), np.uint8)
    cv2.circle(img, (128, 128), 126, (fg, fg, fg, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (128, 128), 30, (bg, bg, bg, 255), -1, cv2.LINE_AA)   # a centre hole: a crisp point to hold
    cv2.imwrite(str(tmp / "dot.png"), img)
    cv2.imwrite(str(tmp / "bg.png"), np.full((16, 16, 4), (bg, bg, bg, 255), np.uint8))
    back = ov.createOverlay("etl.hc.back", "headcal backdrop")
    ov.setOverlayFromFile(back, str(tmp / "bg.png"))
    ov.setOverlayWidthInMeters(back, 4.0)
    mb = openvr.HmdMatrix34_t()
    for i in range(3):
        mb.m[i][i] = 1.0
    mb.m[2][3] = -1.15                                    # head-locked, 1.15 m ahead: steady screen brightness
    ov.setOverlayTransformTrackedDeviceRelative(back, hmd, mb)
    dot = ov.createOverlay("etl.hc.dot", "headcal dot")
    ov.setOverlayFromFile(dot, str(tmp / "dot.png"))
    ov.setOverlayWidthInMeters(dot, 2 * a.dist * math.tan(math.radians(a.dot_deg / 2)))
    ov.setOverlaySortOrder(dot, 1)
    ov.setOverlayTransformAbsolute(dot, openvr.TrackingUniverseStanding, md)

    out = Path(a.out or f"headcal_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    state = {"target": "", "phase": "intro"}
    rec = osc = None
    if a.record:
        from eyerec import EyeRecorder
        rec = EyeRecorder(a.record, out, a.serve, tag=lambda: (state["target"], state["phase"])).start()
    if a.osc:
        from osclog import OscLogger
        osc = OscLogger(a.osc, out / "osc.csv", a.osc_forward).start()
    half_verg = math.degrees(math.atan(IPD_M / 2 / a.dist))
    json.dump({"started": datetime.now().isoformat(timespec="seconds"), "dot_world": p_dot.tolist(),
               "dist_m": a.dist, "dot_deg": a.dot_deg, "bg": a.bg, "seconds": a.seconds, "rate": a.rate,
               "record": a.record, "serve": a.serve, "osc": a.osc,
               "vergence_note": f"each eye is ~{half_verg:.2f} deg inward of the head-centre direction at "
                                f"{a.dist} m (IPD {IPD_M * 1000:.0f} mm); tdist shrinks as the head moves"},
              open(out / "run.json", "w"), indent=1)

    ov.showOverlay(back); ov.showOverlay(dot)
    print(f"dot fixed {a.dist} m ahead. 3 s to settle, then {a.seconds:.0f} s: keep your eyes ON the dot and move "
          f"your head slowly (left-right, up-down, circles). Ctrl+C stops.")
    ys, ps, n = [], [], 0
    try:
        time.sleep(3.0)
        state.update(target="dot", phase="headcal")
        with open(out / "headcal.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_epoch", "valid", "yaw", "pitch", "roll", "speed", "tyaw", "tpitch", "tdist"])
            t0, period, t_next = time.time(), 1.0 / a.rate, time.perf_counter()
            last = None
            while time.time() - t0 < a.seconds:
                ok, m = pose()
                t = time.time()
                hy, hp, hr = angles(m)
                ty, tp, td = target_in_head(m, p_dot)
                if last is not None and t > last[0]:
                    r_rel = last[1][:3, :3].T @ m[:3, :3]     # rotation since the last sample
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r_rel) - 1) / 2))))
                    speed = ang / (t - last[0])
                else:
                    speed = 0.0
                last = (t, m)
                w.writerow([f"{t:.6f}", int(ok), f"{hy:.3f}", f"{hp:.3f}", f"{hr:.3f}", f"{speed:.1f}",
                            f"{ty:.3f}", f"{tp:.3f}", f"{td:.4f}"])
                if ok:
                    ys.append(ty); ps.append(tp)
                n += 1
                t_next += period
                s = t_next - time.perf_counter()
                if s > 0:
                    time.sleep(s)
                else:
                    t_next = time.perf_counter()
    except KeyboardInterrupt:
        print("stopped early")
    finally:
        state.update(target="", phase="end")
        for h in (dot, back):
            ov.destroyOverlay(h)
        openvr.shutdown()
    cov, hit = coverage(ys, ps)
    rows = "\n".join("   " + "".join("#" if c else "." for c in row) for row in hit[::-1])
    print(f"{n} pose samples · dot-in-head yaw {min(ys, default=0):+.0f}..{max(ys, default=0):+.0f}, "
          f"pitch {min(ps, default=0):+.0f}..{max(ps, default=0):+.0f} deg · coverage {100 * cov:.0f}% of "
          f"+-25 x +-20 deg in 5 deg cells{' (aim > 85%: do another run and reach the empty corners)' if cov < 0.85 else ''}")
    print(rows)
    for name, tool in (("recorder", rec), ("osc", osc)):
        if tool:
            print(f"\n[{name}]\n" + tool.stop())
    print(f"\n-> {out}")


if __name__ == "__main__":
    sys.exit(main())
