"""
a1grid.py - target grid for the A1 accuracy test (protocol P1 in analysis/library/chapters/evaluation.md, after QP24):
7x7 = 49 targets over +-25 deg horizontal / +-20 deg vertical, random order. Each dot jumps in at 2.3 deg, shrinks to
0.57 deg over 1 s, then holds for 1 s. Score the window 1.0-1.5 s after the jump (the "score" phase in the CSV).
Run it once on a bright and once on a dark background.

    python tools/a1grid.py --display vr --bg dark --out tests/A1_<date>/dark1 \
        --record http://<eye-board>/ --serve 8090 --osc 8888            (the real test, in the headset)
    python tools/a1grid.py --display screen --distance-cm 60 --out ...   (on the monitor: for dry runs only)
    python tools/a1grid.py --grid 3x3 ...                                (9-point validation for S2 / A2)
    add --wait-viewer 3 to start the grid only once Baballonia has connected to the --serve relay

--display vr: head-locked SteamVR overlay at 1 m, so the angles are exact. Needs SteamVR running.
--display screen: full-screen window on the primary monitor. The angles are only right with your eyes level with the
  screen centre at --distance-cm, and a monitor rarely covers +-25 x +-20 deg, so the grid is shrunk to fit (the real
  angles go in the CSV). Physical size is read from Windows (EDID); override with --screen-cm.
--record / --serve / --osc start eyerec.py / osclog.py in this process, so every frame and OSC message lands in the
  same folder on the same clock, and frames.csv gets the target number + phase of each frame.

Output: targets.csv, one row per target: idx, target (grid id), yaw_deg, pitch_deg, the unit direction tx ty tz (x
right, y up, z forward; error = arccos(g . t)), t_on, t_score0, t_score1, t_off (time.time(), shared with the other
tools), and x_px/y_px in screen mode. Plus run.json with the settings. Esc stops early; the finished rows stay.
"""
import argparse
import csv
import json
import math
import random
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent))
START_DEG, END_DEG, SHRINK_S, HOLD_S = 2.3, 0.57, 1.0, 1.0
SCORE0, SCORE1 = 1.0, 1.5                         # seconds after the jump
INTRO_S = 3.0


def direction(yaw, pitch):
    """Fick angles (yaw first, then pitch) -> unit vector, x right, y up, z forward."""
    y, p = math.radians(yaw), math.radians(pitch)
    return math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)


def dot_deg(t):
    return START_DEG + (END_DEG - START_DEG) * min(1.0, t / SHRINK_S)


def phase(t):
    return "shrink" if t < SCORE0 else "score" if t < SCORE1 else "hold"


class Screen:
    """Full-screen OpenCV window; a target at (yaw, pitch) is where that ray from the eye hits the screen plane."""

    def __init__(self, distance_cm, screen_cm, bright):
        import ctypes
        import cv2
        self.cv2 = cv2
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # real pixels, not 125%-scaled ones
        except OSError:
            pass
        u32, gdi = ctypes.windll.user32, ctypes.windll.gdi32
        self.w, self.h = u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)
        dc = u32.GetDC(0)
        mm_w = gdi.GetDeviceCaps(dc, 4)                      # HORZSIZE, from the monitor's EDID
        u32.ReleaseDC(0, dc)
        self.px_cm = self.w / (screen_cm or mm_w / 10)
        self.d = distance_cm * self.px_cm                    # viewing distance in pixels
        self.bg, self.fg = (235, 25) if bright else (10, 200)
        self.win = "A1 grid (Esc stops)"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.frame = np.full((self.h, self.w), self.bg, np.uint8)
        self.info = f"screen {self.w}x{self.h}px, {self.w / self.px_cm:.1f} cm wide, eye at {distance_cm} cm"

    def fit(self, hfov, vfov):
        # A ray's height on the screen grows by 1/cos(yaw), so the CORNER targets decide the vertical limit
        # (10-11 dry run: clamping yaw and pitch separately put the corner dots at y = -18 px).
        r = 1.5 * self.d * math.tan(math.radians(START_DEG / 2))
        h = min(hfov, math.degrees(math.atan((self.w / 2 - r) / self.d)))
        v = min(vfov, math.degrees(math.atan((self.h / 2 - r) / self.d * math.cos(math.radians(h)))))
        return h, v

    def pos(self, yaw, pitch):
        y, p = math.radians(yaw), math.radians(pitch)
        return self.w / 2 + self.d * math.tan(y), self.h / 2 - self.d * math.tan(p) / math.cos(y)

    def show(self, yaw, pitch, deg):
        x, y = self.pos(yaw, pitch)
        r = self.d * math.tan(math.radians(deg / 2)) / math.cos(math.radians(math.hypot(yaw, pitch)))
        self.frame[:] = self.bg
        s = 16                                               # 4-bit sub-pixel drawing so shrinking looks smooth
        self.cv2.circle(self.frame, (int(x * s), int(y * s)), max(1, int(r * s)), self.fg, -1, self.cv2.LINE_AA, 4)
        self.cv2.imshow(self.win, self.frame)
        return self.cv2.waitKey(1) == 27

    def close(self):
        self.cv2.destroyAllWindows()


class VR:
    """Head-locked SteamVR overlay: a dot texture 1 m away along the target ray, facing the eye, over a backdrop."""

    def __init__(self, bright, dist=1.0):
        import cv2
        import openvr
        self.vr, self.dist = openvr, dist
        import subprocess
        # openvr.init LAUNCHES SteamVR when it isn't running (10-11 test left vrserver + vrmonitor behind): refuse.
        if b"vrserver.exe" not in subprocess.run(["tasklist", "/FI", "IMAGENAME eq vrserver.exe"],
                                                 capture_output=True).stdout:
            sys.exit("SteamVR isn't running: start it with the headset on, then run this again")
        try:
            openvr.init(openvr.VRApplication_Overlay)
        except openvr.OpenVRError as e:
            sys.exit(f"SteamVR overlay failed ({type(e).__name__} {e}): start SteamVR with the headset on first")
        self.ov = openvr.IVROverlay()
        tmp = Path(tempfile.mkdtemp())
        bg, fg = (235, 25) if bright else (10, 200)
        img = np.zeros((256, 256, 4), np.uint8)
        cv2.circle(img, (128, 128), 126, (fg, fg, fg, 255), -1, cv2.LINE_AA)
        cv2.imwrite(str(tmp / "dot.png"), img)
        cv2.imwrite(str(tmp / "bg.png"), np.full((16, 16, 4), (bg, bg, bg, 255), np.uint8))
        self.back = self.ov.createOverlay("etl.a1.back", "A1 backdrop")
        self.ov.setOverlayFromFile(self.back, str(tmp / "bg.png"))
        self.ov.setOverlayWidthInMeters(self.back, 4.0)      # +-60 deg at 1.15 m: fills the view
        self.ov.setOverlayTransformTrackedDeviceRelative(self.back, openvr.k_unTrackedDeviceIndex_Hmd,
                                                         self._mat(0, 0, dist + 0.15))
        self.dot = self.ov.createOverlay("etl.a1.dot", "A1 target")
        self.ov.setOverlayFromFile(self.dot, str(tmp / "dot.png"))
        self.ov.setOverlaySortOrder(self.dot, 1)
        self.ov.showOverlay(self.back)
        self.ov.showOverlay(self.dot)
        self.info = f"SteamVR head-locked overlay at {dist} m"

    def _mat(self, yaw, pitch, d):
        # HMD space: x right, y up, -z forward. R = Ry(-yaw) Rx(pitch); the quad (which faces +z) sits at R(0,0,-d).
        cy, sy = math.cos(math.radians(-yaw)), math.sin(math.radians(-yaw))
        cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        r = [[cy, sy * sp, sy * cp], [0, cp, -sp], [-sy, cy * sp, cy * cp]]
        m = self.vr.HmdMatrix34_t()
        for i in range(3):
            for j in range(3):
                m.m[i][j] = r[i][j]
            m.m[i][3] = -d * r[i][2]
        return m

    def fit(self, hfov, vfov):
        return hfov, vfov

    def show(self, yaw, pitch, deg):
        self.ov.setOverlayWidthInMeters(self.dot, 2 * self.dist * math.tan(math.radians(deg / 2)))
        self.ov.setOverlayTransformTrackedDeviceRelative(self.dot, self.vr.k_unTrackedDeviceIndex_Hmd,
                                                         self._mat(yaw, pitch, self.dist))
        time.sleep(0.008)                                    # ~120 updates/s is plenty for a shrinking dot
        return False                                         # no keyboard in the headset; Ctrl+C in the console

    def close(self):
        for h in (self.dot, self.back):
            self.ov.destroyOverlay(h)
        self.vr.shutdown()


def grid(cols, rows, hfov, vfov):
    xs = [0.0] if cols == 1 else [-hfov + 2 * hfov * i / (cols - 1) for i in range(cols)]
    ys = [0.0] if rows == 1 else [vfov - 2 * vfov * j / (rows - 1) for j in range(rows)]
    return [(f"r{j}c{i}", x, y) for j, y in enumerate(ys) for i, x in enumerate(xs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--display", choices=["vr", "screen"], default="vr")
    ap.add_argument("--bg", choices=["bright", "dark"], default="bright")
    ap.add_argument("--grid", default="7x7", help="COLSxROWS (7x7 = A1, 3x3 = 9-point validation)")
    ap.add_argument("--hfov", type=float, default=25.0, help="outermost column, +- deg")
    ap.add_argument("--vfov", type=float, default=20.0, help="outermost row, +- deg")
    ap.add_argument("--seed", type=int, help="target order seed (default: random, saved in run.json)")
    ap.add_argument("--distance-cm", type=float, default=60.0, help="screen mode: eye to screen")
    ap.add_argument("--screen-cm", type=float, help="screen mode: visible width (default: from Windows)")
    ap.add_argument("--record", help="also record raw frames from this board URL (eyerec.py)")
    ap.add_argument("--serve", type=int, help="with --record: re-serve the stream on 127.0.0.1:<port> for Baballonia")
    ap.add_argument("--osc", type=int, help="also log Baballonia's OSC output on this port (osclog.py)")
    ap.add_argument("--osc-forward", help="with --osc: re-send packets to host:port (VRCFT)")
    ap.add_argument("--wait-viewer", type=float, default=0, metavar="S",
                    help="with --serve: start the grid only after Baballonia connects to the relay, then S s more")
    ap.add_argument("--out", help="run folder (default: a1_<time> here)")
    a = ap.parse_args()

    out = Path(a.out or f"a1_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    cols, rows = (int(v) for v in a.grid.lower().split("x"))
    state = {"target": "", "phase": "intro"}                 # read by the recorder thread for each frame
    rec = osc = None
    if a.record:
        from eyerec import EyeRecorder
        rec = EyeRecorder(a.record, out, a.serve, tag=lambda: (state["target"], state["phase"])).start()
    if a.osc:
        from osclog import OscLogger
        osc = OscLogger(a.osc, out / "osc.csv", a.osc_forward).start()
    if a.wait_viewer and rec and a.serve:
        # Baballonia never reconnects a dropped stream, so the relay must be up BEFORE it connects (10-11).
        print(f"recording + relay up: set Baballonia's camera to http://127.0.0.1:{a.serve}/ · the grid starts once "
              f"it connects (Ctrl+C gives up)")
        try:
            while rec.relay.viewers == 0:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n" + rec.stop()); return
        print(f"viewer connected, frames so far {rec.n}; grid in {a.wait_viewer:.0f} s")
        time.sleep(a.wait_viewer)
    disp = VR(a.bg == "bright") if a.display == "vr" else Screen(a.distance_cm, a.screen_cm, a.bg == "bright")
    hfov, vfov = disp.fit(a.hfov, a.vfov)
    if (hfov, vfov) != (a.hfov, a.vfov):
        print(f"⚠️ {disp.info}: only +-{hfov:.1f} x +-{vfov:.1f} deg fits, grid shrunk to that (dry-run numbers only)")
    seed = a.seed if a.seed is not None else random.randrange(10 ** 6)
    targets = grid(cols, rows, hfov, vfov)
    random.Random(seed).shuffle(targets)

    json.dump({"started": datetime.now().isoformat(timespec="seconds"), "display": disp.info, "bg": a.bg,
               "grid": a.grid, "hfov": hfov, "vfov": vfov, "seed": seed, "dot_deg": [START_DEG, END_DEG],
               "shrink_s": SHRINK_S, "hold_s": HOLD_S, "score_window_s": [SCORE0, SCORE1],
               "record": a.record, "serve": a.serve, "osc": a.osc}, open(out / "run.json", "w"), indent=1)

    print(f"{disp.info} · {len(targets)} targets · seed {seed} · {INTRO_S:.0f} s intro at the centre, then go")
    aborted = False
    with open(out / "targets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "target", "yaw_deg", "pitch_deg", "tx", "ty", "tz", "t_on", "t_score0", "t_score1", "t_off",
                    "x_px", "y_px"])
        try:
            t_end = time.time() + INTRO_S
            while time.time() < t_end and not aborted:
                aborted = disp.show(0, 0, START_DEG)
            for k, (tid, yaw, pitch) in enumerate(targets):
                if aborted:
                    break
                state.update(target=k, phase="shrink")
                aborted = disp.show(yaw, pitch, START_DEG)   # the first frame of this target is now on screen
                t_on = time.time()
                while not aborted and (t := time.time() - t_on) < SHRINK_S + HOLD_S:
                    state["phase"] = phase(t)
                    aborted = disp.show(yaw, pitch, dot_deg(t))
                x, y = disp.pos(yaw, pitch) if hasattr(disp, "pos") else ("", "")
                w.writerow([k, tid, f"{yaw:.3f}", f"{pitch:.3f}"] + [f"{v:.6f}" for v in direction(yaw, pitch)]
                           + [f"{t_on:.6f}", f"{t_on + SCORE0:.6f}", f"{t_on + SCORE1:.6f}", f"{time.time():.6f}"]
                           + ([f"{x:.1f}", f"{y:.1f}"] if x != "" else ["", ""]))
                f.flush()
        except KeyboardInterrupt:
            aborted = True
    state.update(target="", phase="end")
    disp.close()
    print("stopped early, finished targets are saved" if aborted else "grid done")
    for name, tool in (("recorder", rec), ("osc", osc)):
        if tool:
            print(f"\n[{name}]\n" + tool.stop())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
