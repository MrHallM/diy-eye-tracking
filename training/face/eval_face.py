"""Score face models against the Quest Pro on recorded sessions. Read-only.

    python training/face/eval_face.py <session> [<session> ...] [--model tuned=path/faceModel.onnx] [--full-frame]
    python training/face/eval_face.py <session> --lr-check        is left/right the right way round?
    python training/face/eval_face.py <s1> <s2> <s3> --rest       how far do outputs move between re-wears?

The stock model is always included. Preprocessing = Baballonia's real settings unless --full-frame / --no-flip.

--rest (the PLACEMENT BASELINE): for each session it takes the frames from the "Relax ... keep still" prompts and
averages every output. The same resting face should give the same numbers each time the headset is put back on, so
the spread of those averages across sessions is the placement sensitivity. Needs no Quest Pro labels to be right,
only the prompts.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import facedata   # noqa: E402
import facemap    # noqa: E402
import facemodel  # noqa: E402

LR_PAIRS = [("JawLeft", "JawRight"), ("MouthLeft", "MouthRight"), ("MouthSmileLeft", "MouthSmileRight"),
            ("CheekPuffLeft", "CheekPuffRight"), ("MouthStretchLeft", "MouthStretchRight")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--model", action="append", default=[], help="name=path to a faceModel.onnx (repeatable)")
    ap.add_argument("--full-frame", action="store_true")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--lr-check", action="store_true")
    ap.add_argument("--rest", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    roi, hflip, vflip, gamma = facedata.baballonia_face_settings()
    if a.full_frame:
        roi = None
    if a.no_flip:
        hflip = False
    roi = facedata.valid_roi(roi)
    print(f"preprocessing: ROI {roi}, hflip {hflip}, vflip {vflip}, gamma {gamma}")

    models = {"stock": facemodel.load_torch()}
    for m in a.model:
        name, path = m.split("=", 1)
        models[name] = facemodel.load_torch(path)
    sessions = [facedata.Session(s, a.stride) for s in a.sessions]

    def xs(s, idx, flip=hflip, r=roi):
        return np.stack([facedata.preprocess(s.frames[i], r, flip, vflip, gamma) for i in idx])

    if a.lr_check:
        # Which way round does the model's left/right come out, for each way Baballonia could feed it?
        s = sessions[0]
        print("\nLeft/right check (stock model). 'same' = model Left follows the Quest's Left; 'swapped' = it follows "
              "the Quest's Right.\nThe bigger number tells you which way round that setting really is.")
        for label, flip, r in [("full frame, no flip", False, facedata.valid_roi(None)),
                               ("Baballonia ROI, no flip", False, roi), ("Baballonia ROI, flipped", True, roi)]:
            p = facemodel.predict(models["stock"], xs(s, s.keep, flip, r), dev)
            y = s.y[s.keep]
            same, swp = [], []
            print(f"  {label}")
            for L, R in LR_PAIRS:
                iL, iR = facemap.BABBLE.index(L), facemap.BABBLE.index(R)
                d = y[:, iL] - y[:, iR]                    # the Quest's left-minus-right signal
                if d.max() - d.min() < 0.3:
                    continue
                c1 = facemodel._corr(d, p[:, iL] - p[:, iR])
                same.append(c1)
                swp.append(-c1)
                print(f"      {L[:-4]:12} left-minus-right follows the Quest with r = {c1:+.2f}")
            v = np.nanmean(same)
            print(f"      => {'SAME way round' if v > 0.15 else 'SWAPPED' if v < -0.15 else 'unclear'} (mean {v:+.2f})")
        return

    if a.rest:
        print("\nResting-face output per session (mean over the 'relax' prompts), and the spread across sessions")
        means = {k: [] for k in models}
        for s in sessions:
            idx = s.rest()
            if not len(idx):
                print(f"  {s.name}: no 'Relax' prompt frames, skipped")
                continue
            x = xs(s, idx)
            for k, m in models.items():
                means[k].append(facemodel.predict(m, x, dev).mean(0))
        for k in models:
            M = np.array(means[k])
            if len(M) < 2:
                print(f"  {k}: needs 2+ sessions (re-wears) to measure a spread; got {len(M)}")
                continue
            sd = M.std(0)
            worst = np.argsort(-sd)[:8]
            print(f"  {k}: mean spread {sd[:33].mean():.3f} over the 33 non-tongue outputs; worst: "
                  + ", ".join(f"{facemap.BABBLE[i]} {sd[i]:.2f}" for i in worst))
        return

    X = np.concatenate([xs(s, s.keep) for s in sessions])
    Y = np.concatenate([s.y[s.keep] for s in sessions])
    print(f"\n{len(X)} frames from {len(sessions)} session(s), against the Quest Pro")
    facemodel.table({k: facemodel.predict(m, X, dev) for k, m in models.items()}, Y)


if __name__ == "__main__":
    main()
