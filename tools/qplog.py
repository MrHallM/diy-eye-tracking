"""
qplog.py - log the Quest Pro's eye (and face) tracking on the PC, straight from Virtual Desktop, while you play.

Uses: (1) REAL VRChat gaze statistics from the tester's own eyes, to re-run the centre-prior simulation on VRChat behaviour
instead of everyday tasks (analysis/sim/README.md); (2) a same-protocol Quest Pro baseline with a1grid/headcal.

How: the Virtual Desktop Streamer (v1.30+) publishes the headset's face/eye state in shared memory
"VirtualDesktop.BodyState" and signals "VirtualDesktop.BodyStateEvent" on every update. VRCFaceTracking's VD module
reads the SAME memory (github.com/guygodin/VirtualDesktop.VRCFaceTracking, MIT: FaceState.cs, TrackingModule.cs).
Reading is passive: VRCFT, VRChat and OSC are untouched, and nothing has to change in VRChat.

    python tools/qplog.py --out tests/QP_<date>/vrchat1.csv          (until Ctrl+C; add --seconds N)
    python tools/qplog.py --all ...                                  (also all 70 face expressions)
    python tools/qplog.py --selftest                                 (fake shared memory, no headset)
Run tools/poselog.py at the same time for head pose (same time.time() clock).

FaceState layout (C# Sequential, default packing -> 360 bytes): byte FaceFlags, bool IsEyeFollowingBlendshapesValid,
[2 pad], float ExpressionWeights[70], float ExpressionConfidences[2], bool LeftEyeIsValid, bool RightEyeIsValid,
[2 pad], Pose Left {quat x,y,z,w; pos x,y,z}, Pose Right, float LeftEyeConfidence, float RightEyeConfidence.

Gaze angles: the eye pose's forward (-z, OpenXR) as yaw (+ = left) / pitch (+ = up), the same signs as poselog and
headcal. 🔴 Frame + signs NOT yet checked live: on the first run look far left, then up, and check yaw > 0, pitch > 0.
"""
import argparse
import csv
import ctypes
import math
import mmap
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
MAP_NAME, EVENT_NAME = "VirtualDesktop.BodyState", "VirtualDesktop.BodyStateEvent"
FMT = struct.Struct("<BB2x70f2f??2x7f7f2f")
assert FMT.size == 360
EXPR = ("BrowLowererL BrowLowererR CheekPuffL CheekPuffR CheekRaiserL CheekRaiserR CheekSuckL CheekSuckR ChinRaiserB "
        "ChinRaiserT DimplerL DimplerR EyesClosedL EyesClosedR EyesLookDownL EyesLookDownR EyesLookLeftL "
        "EyesLookLeftR EyesLookRightL EyesLookRightR EyesLookUpL EyesLookUpR InnerBrowRaiserL InnerBrowRaiserR "
        "JawDrop JawSidewaysLeft JawSidewaysRight JawThrust LidTightenerL LidTightenerR LipCornerDepressorL "
        "LipCornerDepressorR LipCornerPullerL LipCornerPullerR LipFunnelerLb LipFunnelerLt LipFunnelerRb "
        "LipFunnelerRt LipPressorL LipPressorR LipPuckerL LipPuckerR LipStretcherL LipStretcherR LipSuckLb "
        "LipSuckLt LipSuckRb LipSuckRt LipTightenerL LipTightenerR LipsToward LowerLipDepressorL "
        "LowerLipDepressorR MouthLeft MouthRight NoseWrinklerL NoseWrinklerR OuterBrowRaiserL OuterBrowRaiserR "
        "UpperLidRaiserL UpperLidRaiserR UpperLipRaiserL UpperLipRaiserR TongueTipInterdental TongueTipAlveolar "
        "TongueFrontDorsalPalate TongueMidDorsalPalate TongueBackDorsalVelar TongueOut TongueRetreat").split()
assert len(EXPR) == 70
EYE_EXPR = ["EyesClosedL", "EyesClosedR", "LidTightenerL", "LidTightenerR", "UpperLidRaiserL", "UpperLidRaiserR",
            "InnerBrowRaiserL", "InnerBrowRaiserR", "OuterBrowRaiserL", "OuterBrowRaiserR"]


def gaze(qx, qy, qz, qw):
    """Eye orientation quaternion -> yaw (+ left), pitch (+ up) of its forward axis (-z), in degrees."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    # rotate (0, 0, -1): third column of the rotation matrix, negated
    fx = -(2 * (x * z + w * y))
    fy = -(2 * (y * z - w * x))
    fz = -(1 - 2 * (x * x + y * y))
    return math.degrees(math.atan2(-fx, -fz)), math.degrees(math.asin(max(-1.0, min(1.0, fy))))


def parse(buf):
    v = FMT.unpack_from(buf)
    flags, eyeexp_valid = v[0], v[1]
    expr, conf = v[2:72], v[72:74]
    lvalid, rvalid = v[74], v[75]
    lq, rq = v[76:80], v[83:87]
    lconf, rconf = v[90], v[91]
    return {"flags": flags, "eyeexp_valid": eyeexp_valid, "expr": expr, "face_conf": conf,
            "lvalid": lvalid, "rvalid": rvalid, "lq": lq, "rq": rq, "lconf": lconf, "rconf": rconf}


class Source:
    def __init__(self, name=MAP_NAME, event=EVENT_NAME):
        k32 = ctypes.windll.kernel32
        k32.OpenFileMappingW.restype = ctypes.c_void_p
        h = k32.OpenFileMappingW(0x0004, False, name)        # FILE_MAP_READ; fails if VD isn't publishing
        if not h:
            raise RuntimeError(f"shared memory '{name}' not found: start the Virtual Desktop Streamer and connect "
                               f"the Quest Pro (eye tracking on in VD's settings)")
        k32.CloseHandle(ctypes.c_void_p(h))
        self.mm = mmap.mmap(-1, FMT.size, tagname=name, access=mmap.ACCESS_READ)
        k32.OpenEventW.restype = ctypes.c_void_p
        self.ev = k32.OpenEventW(0x00100000, False, event)     # SYNCHRONIZE
        self.k32 = k32

    def wait(self, ms=50):
        """Block until VD signals an update (or ms pass). Returns False on timeout / no event."""
        if not self.ev:
            time.sleep(0.005)
            return True
        return self.k32.WaitForSingleObject(ctypes.c_void_p(self.ev), ms) == 0

    def read(self):
        return parse(self.mm[:FMT.size])


def log(src, out, seconds, all_expr):
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["t_epoch", "lvalid", "rvalid", "lconf", "rconf", "lyaw", "lpitch", "ryaw", "rpitch",
            "lqx", "lqy", "lqz", "lqw", "rqx", "rqy", "rqz", "rqw", "eyeexp_valid", "face_flags"]
    names = EXPR if all_expr else EYE_EXPR
    idx = [EXPR.index(n) for n in names]
    n = timeouts = same = 0
    last = None
    t0 = time.time()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + names)
        try:
            while seconds is None or time.time() - t0 < seconds:
                if not src.wait():
                    timeouts += 1
                    continue
                s = src.read()
                t = time.time()
                # with --all the face is the point: a mouth change with still eyes must not count as a duplicate
                key = (s["lq"], s["rq"], s["expr"] if all_expr else (s["expr"][12], s["expr"][13]))
                if key == last:                               # event without new data (or polling): skip duplicates
                    same += 1
                    continue
                last = key
                ly, lp = gaze(*s["lq"])
                ry, rp = gaze(*s["rq"])
                w.writerow([f"{t:.6f}", int(s["lvalid"]), int(s["rvalid"]), f"{s['lconf']:.3f}", f"{s['rconf']:.3f}",
                            f"{ly:.3f}", f"{lp:.3f}", f"{ry:.3f}", f"{rp:.3f}"]
                           + [f"{q:.6f}" for q in s["lq"] + s["rq"]]
                           + [int(s["eyeexp_valid"]), s["flags"]] + [f"{s['expr'][i]:.4f}" for i in idx])
                n += 1
                if n % 500 == 0:
                    f.flush()
                    print(f"\r{n} samples, {n / (t - t0):.1f} Hz, gaze L {ly:+5.1f},{lp:+5.1f}  R {ry:+5.1f},{rp:+5.1f}",
                          end="")
        except KeyboardInterrupt:
            pass
    dur = time.time() - t0
    summ = (f"qplog {datetime.fromtimestamp(t0):%Y-%m-%d %H:%M:%S} · {dur:.0f} s · {n} samples "
            f"({n / max(dur, 1e-9):.1f} Hz) · event timeouts {timeouts} · duplicate reads skipped {same}\n")
    out.with_suffix(".summary.txt").write_text(summ, encoding="utf-8")
    print("\n" + summ, end="")


def selftest():
    ok = True
    # gaze(): identity = straight ahead; a +20 deg turn about +y (left) and +10 about +x (up)
    def qaxis(ax, deg):
        h = math.radians(deg) / 2
        return {"x": (math.sin(h), 0, 0, math.cos(h)), "y": (0, math.sin(h), 0, math.cos(h))}[ax]
    for q, want in [((0, 0, 0, 1), (0, 0)), (qaxis("y", 20), (20, 0)), (qaxis("x", 10), (0, 10)),
                    (qaxis("y", -35), (-35, 0)), (qaxis("x", -25), (0, -25))]:
        g = gaze(*q)
        good = abs(g[0] - want[0]) < 1e-6 and abs(g[1] - want[1]) < 1e-6
        ok &= good
        print(f"  q {tuple(round(c, 3) for c in q)} -> yaw {g[0]:+.2f} pitch {g[1]:+.2f} {'ok' if good else 'FAIL'}")
    # end-to-end: publish a fake FaceState under a private name and read it back through Source
    name = f"etl.qplog.selftest.{int(time.time())}"
    expr = [0.0] * 70
    expr[12], expr[13] = 0.25, 0.75
    lq, rq = qaxis("y", 15), qaxis("x", -5)
    blob = FMT.pack(1, True, *expr, 0.9, 0.8, True, False, *lq, 0, 0, 0, *rq, 0, 0, 0, 0.95, 0.5)
    writer = mmap.mmap(-1, FMT.size, tagname=name)
    writer[:] = blob
    src = Source(name, event="etl.none")
    s = src.read()
    good = (s["lvalid"] and not s["rvalid"] and abs(s["expr"][13] - 0.75) < 1e-6 and abs(s["lconf"] - 0.95) < 1e-6
            and abs(gaze(*s["lq"])[0] - 15) < 1e-4 and abs(gaze(*s["rq"])[1] + 5) < 1e-4)
    ok &= good
    print(f"  shared-memory round trip {'ok' if good else 'FAIL'}")
    out = Path(__file__).resolve().parents[1] / "data" / "qplog_selftest.csv"
    log(src, out, 0.3, False)
    rows = list(csv.reader(open(out)))
    ok &= len(rows) == 2 and rows[1][5].startswith("15.0")          # header + 1 row (duplicates skipped)
    out.unlink(); out.with_suffix(".summary.txt").unlink()
    writer.close()
    print("selftest", "passed" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="csv path (default qp_<time>.csv here)")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--all", action="store_true", help="log all 70 face expressions, not just the eye/brow ones")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        src = Source()
    except RuntimeError as e:
        sys.exit(str(e))
    out = Path(a.out or f"qp_{datetime.now():%Y%m%d_%H%M%S}.csv")
    print(f"logging Quest Pro eye data from Virtual Desktop -> {out} (Ctrl+C stops)")
    log(src, out, a.seconds, a.all)


if __name__ == "__main__":
    sys.exit(main())
