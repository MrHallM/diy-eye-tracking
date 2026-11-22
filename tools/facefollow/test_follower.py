"""Offline replay test of follower.py's decisions using frames captured on 2026-11-19 (no Baballonia needed).
Reference = base_rest1 (crop 60,20,150x150). Live frames = timed_rest*/g9_rest*, where the face sat ~5 px higher.
Expected: crop walks up ~5 px in <=3 px steps; expression frames are never measured."""
import sys, json, time, itertools, shutil, tempfile
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent))
import follower as F

D = Path("captures/led_diffuse")
img = lambda n: cv2.imread(str(D / f"{n}.jpg"), cv2.IMREAD_GRAYSCALE)

tmp = Path(tempfile.mkdtemp())
F.REF_PNG, F.REF_JSON, F.HERE = tmp / "reference.png", tmp / "reference.json", tmp
cv2.imwrite(str(F.REF_PNG), img("base_rest1"))
F.REF_JSON.write_text(json.dumps({"roi": {"x": 60, "y": 20, "width": 150, "height": 150}}))
F.RELAX_HOLD_S = 0; F.MEASURE_EVERY_S = 0

relaxed = np.zeros(45); busy = np.zeros(45); busy[F.BABBLE.index("JawOpen")] = 0.9
script = [("hard_tongue", busy), ("base_open1", busy)] + [(n, relaxed) for n in ["timed_rest1", "timed_rest2", "g9_rest1", "g9_rest2"]] * 4
state = {"roi": {"x": 60, "y": 20, "width": 150, "height": 150}, "i": 0, "moves": []}

def get_json(path):
    if path == "/status":
        return {"face": {"roi": dict(state["roi"])}}
    if state["i"] >= len(script):
        raise KeyboardInterrupt
    name, vals = script[state["i"]]; state["cur"] = name; state["i"] += 1
    return {"values": vals.tolist()}

def get_frame():
    return img(state["cur"])

def post_roi(roi):
    state["moves"].append(roi); state["roi"] = roi

F.get_json, F.get_frame, F.post_roi = get_json, get_frame, post_roi
F.latest_calibration_rest = lambda: {}
try:
    F.run(type("A", (), {"apply": True})())
except KeyboardInterrupt:
    pass
print("crop moves:", [(m["x"], m["y"], m["width"], m["height"]) for m in state["moves"]])
final = state["roi"]
ok = 13 <= final["y"] <= 17 and abs(final["x"] - 59) <= 2 and all(
    abs(a["y"] - b["y"]) <= F.STEP_PX for a, b in zip([{"y": 20}] + state["moves"], state["moves"]))
print("PASS" if ok else "FAIL", "final crop", final)
shutil.rmtree(tmp)
