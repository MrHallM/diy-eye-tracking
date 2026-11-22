"""Early preview of the pre-training experiment on the subjects already downloaded, in a separate folder so the
main pipeline (run_pretrain_pipeline.py) is untouched. Smaller run: 12k steps.

    python training/preview_pipeline.py        (log: data/preview.log)
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
PY = sys.executable
D = LAB / "data"
PREV = D / "preview"
LOG = D / "preview.log"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args):
    log("RUN " + " ".join(map(str, args)))
    with open(LOG, "a", encoding="utf-8") as f:
        if subprocess.run([PY, *map(str, args)], cwd=LAB, stdout=f, stderr=subprocess.STDOUT).returncode:
            log("FAILED")
            sys.exit(1)


def main():
    src = D / "prepped" / "vrgaze"
    dst = PREV / "prepped"
    dst.mkdir(parents=True, exist_ok=True)
    # only files that reload cleanly: the downloader may be writing one right now
    import numpy as np
    for f in sorted(src.glob("id*.npz")):
        if f.name.endswith((".tmp.npz", ".pseudo.npz")) or (dst / f.name).exists():
            continue
        try:
            with np.load(f) as z:
                z["valid_left"]
        except Exception:
            continue
        shutil.copy2(f, dst / f.name)
    log(f"preview subjects: {len(list(dst.glob('id*.npz')))}")
    run(["training/prep/pseudolabel.py", D / "vrgaze", dst, "--split", "train"])
    run(["training/prep/pseudolabel.py", D / "vrgaze", D / "prepped" / "vrgaze_val", "--split", "val"])
    run(["training/pretrain/pack.py", dst, PREV / "packed_train"])
    run(["training/pretrain/pack.py", D / "prepped" / "vrgaze_val", PREV / "packed_val_static", "--labels", "static"])
    run(["training/pretrain/train.py", "--train", PREV / "packed_train", "--val", PREV / "packed_val_static",
         "--out", LAB / "runs" / "preview_pre", "--steps", "12000", "--eval-every", "2000"])
    run(["training/eval/fewshot.py", "--val", D / "prepped" / "vrgaze_val", "--pretrained",
         LAB / "runs" / "preview_pre" / "best.pth", "--out", LAB / "runs" / "preview_fewshot", "--k", "50", "200",
         "--steps", "1500"])
    log("PREVIEW DONE")


if __name__ == "__main__":
    main()
