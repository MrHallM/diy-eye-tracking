"""Unattended: wait for fetch_vrgaze.py to finish, then pseudo-label -> pack -> pre-train -> few-shot test.
Each step is skipped if its output already exists, so it can be re-run after a stop.

    python training/run_pretrain_pipeline.py            (log: data/pipeline.log)
"""
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = LAB / "data" / "pipeline.log"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, done_marker=None):
    if done_marker and Path(done_marker).exists():
        log(f"skip (done): {' '.join(map(str, args))}")
        return
    log(f"RUN {' '.join(map(str, args))}")
    with open(LOG, "a", encoding="utf-8") as f:
        r = subprocess.run([PY, *map(str, args)], cwd=LAB, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        log(f"FAILED ({r.returncode}): {args[0]}")
        sys.exit(r.returncode)


def main():
    fetch_log = LAB / "data" / "fetch_vrgaze.log"
    log("waiting for the VRGaze download to finish ...")
    while "ALL DONE" not in (fetch_log.read_text(encoding="utf-8", errors="replace") if fetch_log.exists() else ""):
        time.sleep(60)
    log("download finished")
    d = LAB / "data"
    run(["training/prep/pseudolabel.py", d / "vrgaze", d / "prepped/vrgaze", "--split", "train"],
        d / "prepped/vrgaze/id513.pseudo.npz")
    run(["training/prep/pseudolabel.py", d / "vrgaze", d / "prepped/vrgaze_val", "--split", "val"],
        d / "prepped/vrgaze_val/id229.pseudo.npz")
    run(["training/pretrain/pack.py", d / "prepped/vrgaze", d / "packed/vrgaze_train"],
        d / "packed/vrgaze_train/subjects.txt")
    run(["training/pretrain/pack.py", d / "prepped/vrgaze_val", d / "packed/vrgaze_val_static", "--labels", "static"],
        d / "packed/vrgaze_val_static/subjects.txt")
    run(["training/pretrain/train.py", "--train", d / "packed/vrgaze_train", "--val", d / "packed/vrgaze_val_static",
         "--out", LAB / "runs/pre1", "--steps", "60000"], LAB / "runs/pre1/last.pth")
    run(["training/eval/fewshot.py", "--val", d / "prepped/vrgaze_val", "--pretrained", LAB / "runs/pre1/best.pth",
         "--out", LAB / "runs/fewshot_pre1", "--k", "50", "200", "1000", "--steps", "1500"],
        LAB / "runs/fewshot_pre1/fewshot.json")
    log("PIPELINE DONE")


if __name__ == "__main__":
    main()
