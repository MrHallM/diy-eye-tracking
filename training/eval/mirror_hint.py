"""Two questions from 2026-10-22, answered on VRGaze before any hardware exists:

  MIRROR - can a mirrored right eye stand in for a left eye? (VRGaze stores the right image mirrored.)
    a) per person: calibrate on ONE eye, test on both. If the other eye scores like the calibrated one, a mirrored
       eye is interchangeable for that person.
    b) pre-training: train on left eyes only / mirrored right eyes only (same steps), score each on both eyes.
  HINT - The "colour the parts" idea: does a pupil-blob mask as an extra channel help? (train.py --hint)
    Same-budget pre-training grey vs hint, then the usual few-shot test (scratch and fine-tune, K = 50 / 200).

Error is against VRGaze's clean static points of the person's OTHER sessions (re-wear), in degrees, as fewshot.py.
Unattended + resumable (each stage skips if its output exists); log in data/mirror_hint.log.

    python training/eval/mirror_hint.py [--steps 15000]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pretrain"))
from fewshot import fit, load_subject, score, spread  # noqa: E402
from train import MicroChad, Packed, evaluate, stack4, stack_hint  # noqa: E402

LAB = Path(__file__).resolve().parents[2]
DATA = LAB / "data"
RUNS = LAB / "runs"
OUT = RUNS / "mirror_hint"
LOG = DATA / "mirror_hint.log"
PY = sys.executable
PRETRAIN = {   # run folder -> extra train.py args
    "exp_eyeL": ["--eye", "left"],
    "exp_eyeR": ["--eye", "right"],
    "exp_grey": [],
    "exp_hint": ["--hint"],
}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save(name, obj):
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(OUT / name)


def subjects():
    for path in sorted((DATA / "prepped/vrgaze_val").glob("id*.npz")):
        if not path.name.endswith((".tmp.npz", ".pseudo.npz")) and path.with_suffix(".pseudo.npz").exists():
            yield path


def split(path):
    """fewshot.py's split: calibrate in the session with most usable frames, test on other sessions' static points.
    Also returns which eye each row is (load_subject stacks all left images, then all mirrored right ones)."""
    imgs, cal_labs, cal_ok, test_labs, is_static, seq = load_subject(path)
    eye = np.repeat([0, 1], len(imgs) // 2)
    sessions = sorted(set(seq), key=lambda s: -(cal_ok & (seq == s)).sum())
    return imgs, cal_labs, test_labs, eye, (seq == sessions[0]) & cal_ok, (seq != sessions[0]) & is_static


def load(run, dev):
    m = MicroChad(out_count=2).to(dev)
    m.load_state_dict(torch.load(run, map_location=dev))
    return m


def mirror_per_person(dev, steps, k=200):
    """Calibrate on one eye only, score on each eye. Scratch (no pre-training) and fine-tune of runs/pre1."""
    pre = torch.load(RUNS / "pre1/last.pth", map_location=dev)
    rows = []
    for path in subjects():
        imgs, cal_labs, test_labs, eye, cal_pool, test = split(path)
        for cal_eye in (0, 1):
            pool = np.flatnonzero(cal_pool & (eye == cal_eye))
            cal = pool[spread(cal_labs[pool], k, np.random.default_rng(0))]
            torch.manual_seed(0)
            models = {"scratch": fit(MicroChad(out_count=2).to(dev), imgs[cal], cal_labs[cal], steps, 2e-3, dev, 0)}
            ft = MicroChad(out_count=2).to(dev)
            ft.load_state_dict(pre)
            models["finetune"] = fit(ft, imgs[cal], cal_labs[cal], steps, 3e-4, dev, 0)
            for name, m in models.items():
                r = {"subject": path.stem, "cal_eye": "LR"[cal_eye], "model": name}
                for test_eye in (0, 1):
                    t = test & (eye == test_eye)
                    r["test_" + "LR"[test_eye]] = score(m, imgs[t], test_labs[t], dev)[0]
                rows.append(r)
                log(f"  {json.dumps(r)}")
    return rows


def mirror_pretrain(dev):
    """Each one-eye pre-trained model (and pre1 = both eyes, 60k steps, for reference) scored on each val eye."""
    rows = []
    for run in ("exp_eyeL", "exp_eyeR", "exp_grey", "pre1"):
        m = load(RUNS / run / "last.pth", dev)
        r = {"run": run}
        for e in ("left", "right"):
            r["val_" + e] = evaluate(m, Packed(DATA / "packed/vrgaze_val_static", e), dev)[0]
        rows.append(r)
        log(f"  {json.dumps(r)}")
    return rows


def hint_fewshot(dev, steps):
    """grey vs hint, same budget: scratch (Baballonia-style) and fine-tune of the matching pre-trained model."""
    pre = {"grey": torch.load(RUNS / "exp_grey/last.pth", map_location=dev),
           "hint": torch.load(RUNS / "exp_hint/last.pth", map_location=dev)}
    inp = {"grey": stack4, "hint": stack_hint}
    rows = []
    for path in subjects():
        imgs, cal_labs, test_labs, eye, cal_pool, test = split(path)
        for k in (50, 200):
            pool = np.flatnonzero(cal_pool)
            cal = pool[spread(cal_labs[pool], k, np.random.default_rng(0))]
            r = {"subject": path.stem, "k": k}
            for v in ("grey", "hint"):
                zs = MicroChad(out_count=2).to(dev)
                zs.load_state_dict(pre[v])
                r[f"zero_shot_{v}"] = score(zs, imgs[test], test_labs[test], dev, inp[v])[0]
                torch.manual_seed(0)
                m = fit(MicroChad(out_count=2).to(dev), imgs[cal], cal_labs[cal], steps, 2e-3, dev, 0, inp[v])
                r[f"scratch_{v}"] = score(m, imgs[test], test_labs[test], dev, inp[v])[0]
                ft = MicroChad(out_count=2).to(dev)
                ft.load_state_dict(pre[v])
                ft = fit(ft, imgs[cal], cal_labs[cal], steps, 3e-4, dev, 0, inp[v])
                r[f"finetune_{v}"] = score(ft, imgs[test], test_labs[test], dev, inp[v])[0]
            rows.append(r)
            log(f"  {json.dumps(r)}")
    return rows


def stage(name, fn, *args):
    if (OUT / name).exists():
        log(f"skip (done): {name}")
        return
    log(f"STAGE {name}")
    save(name, fn(*args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000, help="pre-training steps per run (pre1's val error "
                    "plateaued by ~14k)")
    ap.add_argument("--fit-steps", type=int, default=1500, help="few-shot calibration steps, as fewshot.py")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    stage("mirror_per_person.json", mirror_per_person, dev, a.fit_steps)
    for run, extra in PRETRAIN.items():
        if (RUNS / run / "last.pth").exists():
            log(f"skip (done): pre-train {run}")
            continue
        args = ["training/pretrain/train.py", "--train", DATA / "packed/vrgaze_train",
                "--val", DATA / "packed/vrgaze_val_static", "--out", RUNS / run, "--steps", str(a.steps), *extra]
        log(f"RUN {' '.join(map(str, args))}")
        with open(LOG, "a", encoding="utf-8") as f:
            if subprocess.run([PY, *map(str, args)], cwd=LAB, stdout=f, stderr=subprocess.STDOUT).returncode:
                log(f"FAILED: pre-train {run}")
                sys.exit(1)
    stage("mirror_pretrain.json", mirror_pretrain, dev)
    stage("hint_fewshot.json", hint_fewshot, dev, a.fit_steps)
    log("ALL DONE")


if __name__ == "__main__":
    main()
