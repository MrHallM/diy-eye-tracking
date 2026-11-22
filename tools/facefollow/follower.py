"""Face follower: keep Baballonia's face crop on the face when the mouth camera / headset shifts.

Needs the patched Baballonia (builds/baballonia-uncapped, control API on 127.0.0.1:8899 with
/face/frame + /face/latest). It never opens the camera itself.

    python tools/facefollow/follower.py set-reference      # face RELAXED, right after calibrating
    python tools/facefollow/follower.py run                # DRY RUN: logs what it would do
    python tools/facefollow/follower.py run --apply        # actually moves the crop

How it decides:
  1. polls /face/latest ~10x/s; "relaxed" = every watched shape within its calibration rest level + margin,
     held for RELAX_HOLD_S seconds
  2. while relaxed, ~1x/s grabs /face/frame and registers it against the reference (facereg.py)
  3. keeps the last MEDIAN_N good measurements (match quality cc >= MIN_CC); their median = where the face is now
  4. target crop = reference crop moved by that shift (and scaled by the size change)
  5. --apply: moves the live crop toward the target by at most STEP_PX per update, only if it differs by
     >= DEADBAND_PX, never further than MAX_OFFSET_PX from the reference crop
Everything goes to a CSV log. Live only: restarting Baballonia goes back to the saved crop.
"""
import argparse, json, time, csv, urllib.request, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from facereg import FaceRegistrar

API = "http://127.0.0.1:8899"
HERE = Path(__file__).parent
REF_PNG, REF_JSON = HERE / "reference.png", HERE / "reference.json"

RELAX_HOLD_S = 1.0
MEASURE_EVERY_S = 1.0
MIN_CC = 0.85
MEDIAN_N = 5
DEADBAND_PX = 2.0
STEP_PX = 3
MAX_OFFSET_PX = 25
SCALE_RANGE = (0.88, 1.12)

# Babble output order (faceModel.onnx / FaceExpressionMap) - same list as training/face/facemap.py
BABBLE = ["CheekPuffLeft", "CheekPuffRight", "CheekSuckLeft", "CheekSuckRight", "JawOpen", "JawForward", "JawLeft",
          "JawRight", "NoseSneerLeft", "NoseSneerRight", "MouthFunnel", "MouthPucker", "MouthLeft", "MouthRight",
          "MouthRollUpper", "MouthRollLower", "MouthShrugUpper", "MouthShrugLower", "MouthClose", "MouthSmileLeft",
          "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight", "MouthDimpleLeft", "MouthDimpleRight",
          "MouthUpperUpLeft", "MouthUpperUpRight", "MouthLowerDownLeft", "MouthLowerDownRight", "MouthPressLeft",
          "MouthPressRight", "MouthStretchLeft", "MouthStretchRight", "TongueOut", "TongueUp", "TongueDown",
          "TongueLeft", "TongueRight", "TongueRoll", "TongueBendDown", "TongueCurlUp", "TongueSquish", "TongueFlat",
          "TongueTwistLeft", "TongueTwistRight"]
# shapes that tell us the mouth is doing something; cheekSuck / jawForward are biased at rest on this setup
WATCH = ["JawOpen", "JawLeft", "JawRight", "MouthFunnel", "MouthPucker", "MouthLeft", "MouthRight",
         "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight", "CheekPuffLeft", "CheekPuffRight",
         "TongueOut", "MouthRollUpper", "MouthRollLower"]
MARGIN = 0.15


def get_json(path):
    with urllib.request.urlopen(API + path, timeout=2) as r:
        return json.loads(r.read())


def get_frame():
    with urllib.request.urlopen(API + "/face/frame", timeout=2) as r:
        return cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_GRAYSCALE)


def post_roi(roi):
    body = json.dumps({"camera": "face", "roi": roi}).encode()
    req = urllib.request.Request(API + "/transform", data=body, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())


def latest_calibration_rest():
    """Per-shape rest level from the newest Babble Calibrator run (raw model values)."""
    caps = sorted(Path("captures").glob("calibration_*.json"))  # your Babble Calibrator captures folder
    if not caps:
        return {}
    d = json.loads(caps[-1].read_text(encoding="utf-8-sig"))
    return {s["ConfigKey"]: (s["Rest"] or 0.0) for s in d["Shapes"]}


def is_relaxed(values, rest):
    for name in WATCH:
        if values[BABBLE.index(name)] > rest.get(name, 0.05) + MARGIN:
            return False
    return True


def set_reference(args):
    rest = latest_calibration_rest()
    print("Keep your face RELAXED for a few seconds...")
    frames, t0 = [], time.time()
    while time.time() - t0 < 6 and len(frames) < 5:
        v = get_json("/face/latest")["values"]
        if is_relaxed(v, rest):
            frames.append(get_frame())
        time.sleep(0.5)
    if len(frames) < 3:
        sys.exit("Face never looked relaxed long enough - try again (is the face camera running?)")
    ref = np.median(np.stack(frames), axis=0).astype(np.uint8)   # median of a few frames = less noise
    roi = get_json("/status")["face"]["roi"]
    cv2.imwrite(str(REF_PNG), ref)
    REF_JSON.write_text(json.dumps({"roi": roi, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print(f"Reference saved ({len(frames)} relaxed frames), crop {roi}")


def target_roi(ref_roi, dx, dy, scale):
    x, y, w, h = ref_roi["x"], ref_roi["y"], ref_roi["width"], ref_roi["height"]
    cx, cy = x + w / 2 + dx, y + h / 2 + dy
    nw, nh = w * scale, h * scale
    return dict(x=int(round(cx - nw / 2)), y=int(round(cy - nh / 2)), width=int(round(nw)), height=int(round(nh)))


def clamp_roi(roi, size=240):
    w = min(roi["width"], size); h = min(roi["height"], size)
    x = min(max(roi["x"], 0), size - w); y = min(max(roi["y"], 0), size - h)
    return dict(x=x, y=y, width=w, height=h)


def run(args):
    if not REF_PNG.exists():
        sys.exit("No reference yet - run: follower.py set-reference (face relaxed)")
    ref_roi = json.loads(REF_JSON.read_text())["roi"]
    reg = FaceRegistrar(cv2.imread(str(REF_PNG), cv2.IMREAD_GRAYSCALE))
    rest = latest_calibration_rest()
    log_path = HERE / f"follow_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    meas, relaxed_since, last_measure = [], None, 0.0
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"{mode} | reference crop {ref_roi} | log {log_path.name} | Ctrl+C stops")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "relaxed", "dx", "dy", "scale", "rot", "cc", "used", "med_dx", "med_dy", "med_scale",
                    "cur_x", "cur_y", "cur_w", "cur_h", "action"])
        while True:
            try:
                v = get_json("/face/latest")["values"]
            except Exception as e:
                print("waiting for Baballonia...", e); time.sleep(2); continue
            now = time.time()
            if is_relaxed(v, rest):
                relaxed_since = relaxed_since or now
            else:
                relaxed_since = None
            if not (relaxed_since and now - relaxed_since >= RELAX_HOLD_S and now - last_measure >= MEASURE_EVERY_S):
                time.sleep(0.1); continue
            last_measure = now
            try:
                m = reg.measure(get_frame())
            except Exception as e:
                print("frame failed:", e); continue
            good = m["ok"] and m["cc"] >= MIN_CC and SCALE_RANGE[0] <= m["scale"] <= SCALE_RANGE[1]
            if good:
                meas.append(m); meas = meas[-MEDIAN_N:]
            cur = get_json("/status")["face"]["roi"]
            action, md = "", None
            if len(meas) >= 3:
                md = {k: float(np.median([x[k] for x in meas])) for k in ("dx", "dy", "scale")}
                tgt = clamp_roi(target_roi(ref_roi, md["dx"], md["dy"], md["scale"]))
                off = max(abs(tgt["x"] - ref_roi["x"]), abs(tgt["y"] - ref_roi["y"]))
                gap = max(abs(tgt[k] - cur[k]) for k in ("x", "y", "width", "height"))
                if off > MAX_OFFSET_PX:
                    action = f"too far ({off}px) - recalibrate"
                elif gap >= DEADBAND_PX:
                    step = {k: cur[k] + int(np.clip(tgt[k] - cur[k], -STEP_PX, STEP_PX)) for k in cur}
                    action = f"move {cur['x']},{cur['y']},{cur['width']}x{cur['height']} -> " \
                             f"{step['x']},{step['y']},{step['width']}x{step['height']} (target {tgt['x']},{tgt['y']},{tgt['width']}x{tgt['height']})"
                    if args.apply:
                        post_roi(clamp_roi(step))
                    else:
                        action = "WOULD " + action
            w.writerow([f"{now:.2f}", 1, f"{m['dx']:.2f}", f"{m['dy']:.2f}", f"{m['scale']:.4f}", f"{m['rot_deg']:.2f}",
                        f"{m['cc']:.3f}", int(good)] + ([f"{md['dx']:.2f}", f"{md['dy']:.2f}", f"{md['scale']:.4f}"] if md else ["", "", ""])
                       + [cur["x"], cur["y"], cur["width"], cur["height"], action])
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} dx {m['dx']:+5.1f} dy {m['dy']:+5.1f} scale {m['scale']:.3f} cc {m['cc']:.2f}"
                  f"{'' if good else ' (rejected)'} {action}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["set-reference", "run"])
    ap.add_argument("--apply", action="store_true", help="actually move the crop (default: dry run, log only)")
    a = ap.parse_args()
    set_reference(a) if a.cmd == "set-reference" else run(a)
