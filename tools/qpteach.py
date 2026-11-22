"""
qpteach.py - Quest Pro as TEACHER for the Babble mouth camera. Records the mouth camera and the Quest Pro's 70 face
weights (Virtual Desktop shared memory) at the same time, so Babble's face model can be fine-tuned on the user's own face.
Plan + facts: memory note project_babble_qp_teacher. Step 1 = this: does the pairing work at all?

    python tools/qpteach.py record --out tests/QPT_<date>/raw/p0       speaks a ~90 s face routine while recording
    python tools/qpteach.py check tests/QPT_<date>/raw/p0              time offset + stock model vs Quest Pro

Needs: Virtual Desktop streaming, "Forward tracking data to PC" on, Quest Pro Natural Facial Expressions ON, and a VR
app or SteamVR running (VD only publishes face data then). The patched mouth-camera firmware serves several clients,
so Baballonia can keep running. Reuses eyerec.EyeRecorder (camera) and qplog.Source/log (face) on the same
time.time() clock, so the two join on t_epoch.

check: runs Baballonia's stock faceModel.onnx on every frame with Baballonia's own preprocessing (gray, full-frame ROI,
resize 224, /255), finds the camera-vs-Quest time offset from the jaw (cross-correlation), then reports how well each
stock Babble output follows its Quest Pro counterpart. Low correlation = room for the teacher to help. Left/right is
checked both ways in case the camera image is mirrored relative to Meta's L/R.
"""
import argparse
import csv
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import eyerec   # noqa: E402
import qplog    # noqa: E402

CAM = "http://192.168.8.192/"
FACE_MODEL = Path("builds/baballonia-uncapped/faceModel.onnx")

# (seconds, spoken prompt). Slow, repeated movements give the cross-correlation something to lock on to.
ROUTINE = [
    (8, "Relax your face and keep still."),
    (10, "Open your jaw wide and close it, slowly, four times."),
    (6, "Smile big, then relax. Three times."),
    (6, "Smile with only the left side of your mouth."),
    (6, "Now only the right side."),
    (5, "Pucker your lips, like a kiss."),
    (5, "Make an oo shape."),
    (6, "Move your jaw left, then right. Repeat."),
    (5, "Push your jaw forward, then back."),
    (5, "Puff out your cheeks."),
    (5, "Stick your tongue out, then in."),
    (6, "Move your whole mouth to the left, then to the right."),
    (15, "Now talk normally. Count from one to twenty out loud."),
    (5, "Relax again and keep still."),
]

# Babble output -> Quest Pro expressions averaged to compare against it (Meta FB face tracking v2 names)
PAIRS = {
    "JawOpen": ["JawDrop"],
    "JawForward": ["JawThrust"],
    "JawLeft": ["JawSidewaysLeft"], "JawRight": ["JawSidewaysRight"],
    "MouthSmileLeft": ["LipCornerPullerL"], "MouthSmileRight": ["LipCornerPullerR"],
    "MouthFrownLeft": ["LipCornerDepressorL"], "MouthFrownRight": ["LipCornerDepressorR"],
    "MouthStretchLeft": ["LipStretcherL"], "MouthStretchRight": ["LipStretcherR"],
    "MouthPucker": ["LipPuckerL", "LipPuckerR"],
    "MouthFunnel": ["LipFunnelerLb", "LipFunnelerLt", "LipFunnelerRb", "LipFunnelerRt"],
    "MouthLeft": ["MouthLeft"], "MouthRight": ["MouthRight"],
    "CheekPuffLeft": ["CheekPuffL"], "CheekPuffRight": ["CheekPuffR"],
    "MouthClose": ["LipsToward"],
    "TongueOut": ["TongueOut"],
}
SWAP = {"Left": "Right", "Right": "Left"}


class Speaker:
    """One long-lived PowerShell SAPI process, so each prompt speaks without a start-up delay."""

    def __init__(self):
        self.p = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command",
             "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
             "while (($l = [Console]::In.ReadLine()) -ne $null) { $s.SpeakAsync($l) | Out-Null }"],
            stdin=subprocess.PIPE, text=True)

    def say(self, text):
        print(f"  >> {text}")
        self.p.stdin.write(text.replace("\n", " ") + "\n")
        self.p.stdin.flush()

    def close(self):
        self.p.stdin.close()
        self.p.wait(10)


def record(out, cam, loops=1, no_qp=False):
    """loops: say the routine this many times (one folder = ONE way of wearing the headset; re-wear = new folder).
    no_qp: camera + prompts only, for the placement baseline with Quest Pro face tracking OFF (no labels needed)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    total = sum(d for d, _ in ROUTINE) * loops + 2
    face = None
    if not no_qp:
        src = qplog.Source()                              # fails clearly if VD isn't publishing
        s = src.read()
        if not any(s["expr"]):
            sys.exit("Virtual Desktop is publishing, but every face weight is 0: is Natural Facial Expressions on and "
                     "a VR app or SteamVR running?")
        face = threading.Thread(target=qplog.log, args=(src, out / "qp.csv", total, True), daemon=True)
    cam_rec = eyerec.EyeRecorder(cam, out).start()
    if face:
        face.start()
    sp = Speaker()
    t0 = time.time()
    with open(out / "prompts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_epoch", "seconds", "prompt"])
        sp.say("Starting.")
        time.sleep(1)
        for k in range(loops):
            for secs, text in ROUTINE:
                w.writerow([f"{time.time():.6f}", secs, text])
                sp.say(text)
                time.sleep(secs)
            if k + 1 < loops:
                sp.say(f"Round {k + 2} of {loops}.")
        sp.say("Done. Thank you.")
    if face:
        face.join(total + 5)
    print(cam_rec.stop())
    sp.close()
    print(f"recorded {time.time() - t0:.0f} s -> {out}")


def check(folder):
    import cv2
    import numpy as np
    import onnxruntime as ort
    folder = Path(folder)
    names =["CheekPuffLeft", "CheekPuffRight", "CheekSuckLeft", "CheekSuckRight", "JawOpen", "JawForward", "JawLeft",
             "JawRight", "NoseSneerLeft", "NoseSneerRight", "MouthFunnel", "MouthPucker", "MouthLeft", "MouthRight",
             "MouthRollUpper", "MouthRollLower", "MouthShrugUpper", "MouthShrugLower", "MouthClose", "MouthSmileLeft",
             "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight",
             "MouthUpperUpLeft", "MouthUpperUpRight", "MouthLowerDownLeft", "MouthLowerDownRight", "MouthPressLeft",
             "MouthPressRight", "MouthStretchLeft", "MouthStretchRight", "TongueOut", "TongueUp", "TongueDown",
             "TongueLeft", "TongueRight", "TongueRoll", "TongueBendDown", "TongueCurlUp", "TongueSquish", "TongueFlat",
             "TongueTwistLeft", "TongueTwistRight"]   # = ParameterSenderService.FaceExpressionMap order = model outputs

    # camera frames -> stock model outputs (+ mouth brightness, for the flicker)
    sess = ort.InferenceSession(str(FACE_MODEL), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    t_cam, outs, bright = [], [], []
    for row, jpg in eyerec.read_frames(folder):
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        x = cv2.resize(img, (224, 224)).astype(np.float32)[None, None] / 255.0
        outs.append(sess.run(None, {inp: x})[0][0])
        t_cam.append(float(row["t_epoch"]))
        h, w = img.shape
        bright.append(img[h * 3 // 8:h * 7 // 10, w * 3 // 10:w * 7 // 10].mean())
    t_cam, B, bright = np.array(t_cam), np.array(outs), np.array(bright)

    # Quest Pro face weights
    with open(folder / "qp.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    t_qp = np.array([float(r["t_epoch"]) for r in rows])
    Q = {k: np.array([float(r[k]) for r in rows]) for k in qplog.EXPR}

    dur = t_cam[-1] - t_cam[0]
    print(f"camera: {len(t_cam)} frames, {len(t_cam) / dur:.0f} fps · Quest Pro: {len(t_qp)} samples, "
          f"{len(t_qp) / (t_qp[-1] - t_qp[0]):.0f} Hz · face flags {sorted({r['face_flags'] for r in rows})}")
    jump = np.abs(np.diff(bright))
    print(f"mouth brightness: mean {bright.mean():.0f}, frame-to-frame jump avg {jump.mean():.1f} "
          f"(~0.2 = steady, ~20 = the Quest's IR flicker)")

    # common 100 Hz grid; find the camera-vs-Quest offset from the jaw
    grid = np.arange(max(t_cam[0], t_qp[0]) + 1, min(t_cam[-1], t_qp[-1]) - 1, 0.01)

    def on_grid(t, v, lag=0.0):
        return np.interp(grid, t + lag, v)

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else float("nan")

    jaw_qp = on_grid(t_qp, Q["JawDrop"])
    jaw_b = B[:, names.index("JawOpen")]
    lags = np.arange(-0.5, 0.501, 0.01)
    cs = [corr(jaw_qp, on_grid(t_cam, jaw_b, -lag)) for lag in lags]
    best = float(lags[int(np.nanargmax(cs))])
    print(f"time offset: camera frames arrive {best * 1000:+.0f} ms vs the Quest Pro (jaw r = {max(cs):.2f}; "
          f"positive = camera later)")
    if max(cs) < 0.4:
        print("  ⚠ weak jaw match: the offset is unreliable. Did the jaw-open part of the routine get recorded?")

    print(f"\n{'Babble output':20} {'Quest Pro':34} {'r':>6} {'r (L/R swapped)':>16}  Quest range  Babble range")
    for bname, qnames in PAIRS.items():
        q = on_grid(t_qp, np.mean([Q[n] for n in qnames], axis=0))
        b = on_grid(t_cam, B[:, names.index(bname)], -best)
        sw = ""
        if any(s in bname for s in SWAP):
            other = bname.replace("Left", "§").replace("Right", "Left").replace("§", "Right")
            sw = f"{corr(q, on_grid(t_cam, B[:, names.index(other)], -best)):16.2f}"
        print(f"{bname:20} {'+'.join(qnames)[:34]:34} {corr(q, b):6.2f} {sw:>16}  "
              f"{q.min():.2f}-{q.max():.2f}    {b.min():.2f}-{b.max():.2f}")
    print("\nr near 1 = the stock model already follows the Quest Pro; near 0 = it doesn't (room to teach). "
          "Quest range near 0 = that expression never happened in this recording, so its r means nothing.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--cam", default=CAM)
    r.add_argument("--loops", type=int, default=1, help="repeat the ~90 s routine this many times")
    r.add_argument("--no-qp", action="store_true", help="camera + prompts only (Quest Pro face tracking OFF)")
    c = sub.add_parser("check")
    c.add_argument("folder")
    a = ap.parse_args()
    if a.cmd == "record":
        record(a.out, a.cam, a.loops, a.no_qp)
    else:
        check(a.folder)


if __name__ == "__main__":
    main()
