"""
calarchive.py - keep EVERY Baballonia eye calibration and train on several sessions at once.

Why: Baballonia throws each calibration away. Every run writes the step captures, merges them into
ModelData\\user_cal.bin (OVERWRITTEN next time) and trains a model from scratch on only that file
(repos/Baballonia ICalibrationRoutine.cs: MergeBinsStep + TrainerCalibrationStep, read 10-11). Keeping them gives
more training data, and because every session catches the headset sitting a little differently, a model trained on
several sessions should learn your natural re-seats (the Pupil Invisible recipe). Idea (10-11), after the
Quest Pro's stored calibration.

    python tools/calarchive.py watch                  leave running: archives each new calibration as it lands
    python tools/calarchive.py save                   archive the current user_cal.bin once
    python tools/calarchive.py list                   sessions in the archive (frames, dates, label ranges)
    python tools/calarchive.py train --last 5         join the 5 newest sessions, run Baballonia's own BabbleTrainer
         [--weight-latest 2]                          include the newest session twice (it matches today's fit best)
         [--install]                                  make Baballonia use the result (close Baballonia first;
                                                      settings backup kept). Or pick the .onnx in its model picker.
    python tools/calarchive.py --selftest             format + join check on fake data (no Baballonia needed)

Archive: EyeTrackingLab/data/calib_archive/ (git-ignored: these are images of your eyes). Each session =
<time>_user_cal.bin + <time>.json (frame count, label ranges, the crop settings Baballonia had at the time).
⚠️ If the crop changed a lot between sessions, old data looks different to the model: compare with and without them.

File format (repos/Baballonia CaptureFrame.cs): per frame a packed little-endian 100-byte header
(16 float32 labels: routine pitch/yaw/distance/convergence/fov, L/R eye pitch/yaw, lids, brow, widen, squint,
dilate; 3 uint64 timestamps ms; uint32 state flags; uint32 left/right JPEG lengths), then the left and right JPEGs.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
HDR = struct.Struct("<16f3Q3I")                      # 100 bytes
LABELS = ["pitch", "yaw", "dist", "conv", "fov", "lpitch", "lyaw", "rpitch", "ryaw",
          "llid", "rlid", "brow_raise", "brow_angry", "widen", "squint", "dilate"]
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "calib_archive"
APPDATA = Path(os.environ.get("APPDATA", ""))
LOCAL = Path(os.environ.get("LOCALAPPDATA", ""))
# Source (Utils.cs) puts ModelData in %APPDATA%\ProjectBabble\ModelData; older builds used the install folder.
MODELDATA_CANDIDATES = [APPDATA / "ProjectBabble" / "ModelData", LOCAL / "Baballonia"]
TRAINER = LOCAL / "Baballonia" / "Calibration" / "Windows" / "Trainer" / "BabbleTrainer.exe"
SETTINGS = APPDATA / "ProjectBabble" / "ApplicationData" / "LocalSettings.json"
MODEL_NAME = "tuned_temporal_eye_tracking_latest.onnx"


def frames(path):
    """Yield (labels dict, flags, left_len, right_len) per frame; stops at a malformed tail like Baballonia does."""
    with open(path, "rb") as f:
        while True:
            h = f.read(HDR.size)
            if len(h) < HDR.size:
                return
            v = HDR.unpack(h)
            ll, rl = v[20], v[21]
            if ll > 100 << 20 or rl > 100 << 20:
                return
            body = f.read(ll + rl)
            if len(body) < ll + rl:
                return
            yield dict(zip(LABELS, v[:16])), v[19], ll, rl, v[16]


def describe(path):
    n, lo, hi, t0, t1 = 0, {}, {}, None, None
    for lab, _flags, _ll, _rl, ts in frames(path):
        n += 1
        for k in ("pitch", "yaw", "llid", "rlid"):
            lo[k] = min(lo.get(k, lab[k]), lab[k]); hi[k] = max(hi.get(k, lab[k]), lab[k])
        t0 = ts if t0 is None else t0
        t1 = ts
    return {"frames": n, "ranges": {k: [round(lo[k], 2), round(hi[k], 2)] for k in lo},
            "span_s": round((t1 - t0) / 1000, 1) if n else 0}


def find_usercal():
    for d in MODELDATA_CANDIDATES:
        p = d / "user_cal.bin"
        if p.exists():
            return p
    return None


def crop_settings():
    try:
        s = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {k: v for k, v in s.items() if any(w in k.lower() for w in ("roi", "crop", "camera", "rotation", "flip"))}


def save(src=None, quiet=False):
    src = Path(src) if src else find_usercal()
    if not src or not src.exists():
        print("no user_cal.bin found (looked in: " + ", ".join(str(d) for d in MODELDATA_CANDIDATES) + ")")
        return None
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(src.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
    dst = ARCHIVE / f"{stamp}_user_cal.bin"
    if dst.exists():
        if not quiet:
            print(f"already archived: {dst.name}")
        return dst
    shutil.copy2(src, dst)
    meta = {"source": str(src), "archived": datetime.now().isoformat(timespec="seconds"), **describe(dst),
            "crop_settings": crop_settings()}
    model = src.parent / MODEL_NAME
    if model.exists():
        shutil.copy2(model, ARCHIVE / f"{stamp}_{MODEL_NAME}")
        meta["model_copied"] = True
    (ARCHIVE / f"{stamp}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"archived {dst.name}: {meta['frames']} frames, {meta['span_s']} s")
    return dst


def sessions():
    return sorted(ARCHIVE.glob("*_user_cal.bin"))


def cmd_list():
    S = sessions()
    if not S:
        print(f"archive empty ({ARCHIVE})"); return
    for p in S:
        m = json.loads((ARCHIVE / p.name.replace("_user_cal.bin", ".json")).read_text(encoding="utf-8"))
        r = m.get("ranges", {})
        print(f"  {p.name[:15]}  {m['frames']:>6} frames  {m['span_s']:>6} s  "
              f"yaw {r.get('yaw')} pitch {r.get('pitch')}")
    print(f"{len(S)} session(s)")


def combine(paths, out):
    with open(out, "wb") as o:
        for p in paths:
            with open(p, "rb") as i:
                shutil.copyfileobj(i, o, 1 << 20)


def cmd_train(last, weight_latest, install, dry):
    S = sessions()[-last:]
    if not S:
        sys.exit("archive empty: calibrate in Baballonia first (with `watch` running, or run `save` after)")
    parts = S + [S[-1]] * max(0, weight_latest - 1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_bin = ARCHIVE / f"combined_{stamp}.bin"
    out_onnx = ARCHIVE / f"combined_{stamp}.onnx"
    combine(parts, out_bin)
    n = sum(1 for _ in frames(out_bin))
    (ARCHIVE / f"combined_{stamp}.json").write_text(json.dumps(
        {"sessions": [p.name for p in parts], "frames": n, "weight_latest": weight_latest}, indent=1), encoding="utf-8")
    print(f"joined {len(S)} session(s) (+{max(0, weight_latest - 1)} extra copy of the newest): {n} frames -> {out_bin.name}")
    if dry:
        return
    if not TRAINER.exists():
        sys.exit(f"BabbleTrainer not found at {TRAINER}")
    print("training (Baballonia's own trainer, same as its calibration step)...")
    r = subprocess.run([str(TRAINER), str(out_bin), str(out_onnx)], cwd=TRAINER.parent)
    if r.returncode != 0 or not out_onnx.exists():
        sys.exit(f"trainer failed (exit {r.returncode})")
    print(f"model -> {out_onnx}")
    if install:
        install_model(out_onnx)


def install_model(onnx):
    # Baballonia's own path (OverlayCalibrationService.cs): the trained model goes into %APPDATA%\ProjectBabble\Models
    # with a timestamp and the "EyeHome_EyeModel" setting points at it. Settings are read at start-up and rewritten
    # by the app, so Baballonia must be CLOSED for this.
    if b"Baballonia" in subprocess.run(["tasklist"], capture_output=True).stdout:
        sys.exit(f"close Baballonia first, then: python tools/calarchive.py install {onnx}")
    models = APPDATA / "ProjectBabble" / "Models"
    models.mkdir(parents=True, exist_ok=True)
    dest = models / f"tuned_temporal_eye_tracking_{datetime.now():%Y%m%d_%H%M%S}_multi.onnx"
    shutil.copy2(onnx, dest)
    st = json.loads(SETTINGS.read_text(encoding="utf-8"))
    shutil.copy2(SETTINGS, SETTINGS.with_name(f"LocalSettings.backup_calarchive_{datetime.now():%Y%m%d_%H%M%S}.json"))
    old = st.get("EyeHome_EyeModel")
    st["EyeHome_EyeModel"] = str(dest)
    SETTINGS.write_text(json.dumps(st, indent=2), encoding="utf-8")
    print(f"eye model -> {dest}\n(was {old}; settings backup saved next to LocalSettings.json). Start Baballonia.")


def cmd_watch(interval):
    print(f"watching for new calibrations every {interval:.0f} s (Ctrl+C stops)")
    last = None
    while True:
        p = find_usercal()
        if p:
            m = p.stat().st_mtime
            if last is not None and m != last:
                time.sleep(3)                         # let the merge finish writing
                save(p)
            elif last is None:
                save(p, quiet=True)
            last = m
        time.sleep(interval)


def selftest():
    import tempfile
    global ARCHIVE
    tmp = Path(tempfile.mkdtemp())
    ARCHIVE = tmp / "archive"
    ok = True

    def fake(path, n, yaw0, t0):
        with open(path, "wb") as f:
            for i in range(n):
                lab = [0.0] * 16
                lab[0], lab[1] = 5.0, yaw0 + i
                jl, jr = b"\xff\xd8" + bytes(10 + i) + b"\xff\xd9", b"\xff\xd8\xff\xd9"
                f.write(HDR.pack(*lab, t0 + 33 * i, t0 + 33 * i, t0 + 33 * i, 0, len(jl), len(jr)) + jl + jr)

    assert HDR.size == 100
    a, b = tmp / "a" / "user_cal.bin", tmp / "b" / "user_cal.bin"
    a.parent.mkdir(); b.parent.mkdir()
    fake(a, 30, -20, 1000); fake(b, 20, 0, 5000)
    os.utime(b, (time.time() + 5, time.time() + 5))
    pa, pb = save(a), save(b)
    ok &= pa is not None and pb is not None and len(sessions()) == 2
    d = describe(pa)
    ok &= d["frames"] == 30 and d["ranges"]["yaw"] == [-20.0, 9.0]
    cmd_train(5, 2, False, dry=True)
    comb = sorted(ARCHIVE.glob("combined_*.bin"))[-1]
    n = sum(1 for _ in frames(comb))
    ok &= n == 30 + 20 + 20
    with open(comb, "ab") as f:                       # a torn tail must be ignored, like Baballonia does
        f.write(HDR.pack(*([0.0] * 16), 0, 0, 0, 0, 999, 0)[:60])
    ok &= sum(1 for _ in frames(comb)) == n
    print("selftest", "passed" if ok else "FAILED")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="archive Baballonia calibrations and train on several sessions")
    ap.add_argument("cmd", nargs="?", choices=["watch", "save", "list", "train", "install"])
    ap.add_argument("path", nargs="?", help="install: the .onnx to make Baballonia use")
    ap.add_argument("--last", type=int, default=5)
    ap.add_argument("--weight-latest", type=int, default=1)
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="train: only join the sessions")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "watch":
        cmd_watch(a.interval)
    elif a.cmd == "save":
        save()
    elif a.cmd == "list":
        cmd_list()
    elif a.cmd == "install":
        install_model(Path(a.path) if a.path else sys.exit("install needs the .onnx path"))
    elif a.cmd == "train":
        cmd_train(a.last, a.weight_latest, a.install, a.dry_run)
    else:
        ap.print_help()


if __name__ == "__main__":
    sys.exit(main())
