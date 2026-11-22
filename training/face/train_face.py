"""Fine-tune Babble's face model on the user's own face, with the Quest Pro as the teacher.

    python training/face/train_face.py --train <session> [<session> ...] --val <session> [...] --out training/face/runs/<name>
    python training/face/train_face.py --dry-run tests/QPT_2026-10-17/raw/p0 --out training/face/runs/dryrun

A session = one qpteach.py recording folder = one way of wearing the headset. --val sessions are never trained on:
they are the held-out placements the result is judged by. --dry-run splits ONE session in time instead, which only
proves the pipeline works.

What it does
  * starts from the stock model's weights (so anything it already knows is kept)
  * 34 outputs learn from the Quest Pro; the 11 tongue outputs with no teacher are held at the stock model's answer
  * every training image is re-cropped a little differently (shift, zoom, tilt, brightness, blur, noise) around the
    ROI Baballonia really uses, so the result should care less about exactly how the camera sits
  * keeps the epoch that did best on the held-out sessions, exports faceModel.onnx with the stock input name/shape,
    and proves the exported file gives the same numbers as the trained model

It does NOT install anything. Install = copy the new faceModel.onnx over the one in the Baballonia build folder
(back the old one up first), then re-run Baballonia's calibration, because the output ranges change.
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import facedata   # noqa: E402
import facemap    # noqa: E402
import facemodel  # noqa: E402

FR = facedata.FRAME


def augment(frames, roi, hflip, vflip, gamma, gen, strength=1.0):
    """frames: uint8 [B, 240, 240] on the GPU -> float [B, 1, 224, 224], a randomly disturbed version of what
    Baballonia would feed the model. strength 0 = exactly Baballonia's crop (up to resampling)."""
    B, dev = frames.shape[0], frames.device
    x, y, w, h = roi
    r = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi, generator=gen)   # noqa: E731
    s = strength
    zoom = r(1 - 0.15 * s, 1 + 0.18 * s)                 # >1 = sees more around the mouth (camera further away)
    ang = r(-10 * s, 10 * s) * math.pi / 180
    dx, dy = r(-0.10 * s, 0.10 * s) * w, r(-0.10 * s, 0.10 * s) * h
    cx, cy = (x + w / 2 + dx) / FR * 2 - 1, (y + h / 2 + dy) / FR * 2 - 1
    hw, hh = zoom * w / FR, zoom * h / FR                # half-size of the crop in normalised frame units
    fx = -1.0 if hflip else 1.0
    fy = -1.0 if vflip else 1.0
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([hw * cos * fx, -hh * sin * fy, cx], 1),
                         torch.stack([hw * sin * fx, hh * cos * fy, cy], 1)], 1)
    img = frames[:, None].float() / 255.0
    if gamma != 1.0:
        img = (img * gamma).clamp(0, 1)
    grid = F.affine_grid(theta, (B, 1, facedata.SIZE, facedata.SIZE), align_corners=False)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if s > 0:
        gain = r(1 - 0.35 * s, 1 + 0.35 * s).view(B, 1, 1, 1)
        g = torch.exp(r(-0.35 * s, 0.35 * s)).view(B, 1, 1, 1)          # tone curve
        out = (out.clamp(1e-4, 1) ** g * gain)
        blur = r(0, 1) < 0.3 * s
        if blur.any():
            k = torch.tensor([1., 4., 6., 4., 1.], device=dev)
            k = (k[:, None] * k[None]) / 256
            out[blur] = F.conv2d(F.pad(out[blur], (2, 2, 2, 2), mode="replicate"), k[None, None])
        out = out + torch.randn(out.shape, device=dev, generator=gen) * r(0, 0.03 * s).view(B, 1, 1, 1)
        out = out.clamp(0, 1)
    return out


def exact(sess, idx, roi, hflip, vflip, gamma):
    """Baballonia's own preprocessing (cv2), for scoring: float32 [N, 224, 224]."""
    return np.stack([facedata.preprocess(sess.frames[i], roi, hflip, vflip, gamma) for i in idx]) if len(idx) \
        else np.zeros((0, 224, 224), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--dry-run", help="ONE session, split in time (pipeline test only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--stride", type=int, default=2, help="use every Nth camera frame (neighbours are near-copies)")
    ap.add_argument("--aug", type=float, default=1.0, help="augmentation strength, 0 = none")
    ap.add_argument("--mirror", action="store_true",
                    help="also train on mirrored images with Left/Right targets swapped. ONLY valid if the camera "
                         "sits on the face's midline; compare held-out results with and without before trusting it")
    ap.add_argument("--full-frame", action="store_true", help="ignore Baballonia's ROI/flip and use the whole frame")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    roi, hflip, vflip, gamma = facedata.baballonia_face_settings()
    if a.full_frame:
        roi, hflip, vflip, gamma = None, False, False, 1.0
    roi = facedata.valid_roi(roi)
    print(f"preprocessing = Baballonia's: ROI {roi}, hflip {hflip}, vflip {vflip}, gamma {gamma}")

    # ---- data
    parts = []                                            # (session, train idx, val idx)
    if a.dry_run:
        s = facedata.Session(a.dry_run, a.stride)
        tr, va = s.split_in_time()
        parts.append((s, tr, va))
        print("DRY RUN: one session split in time. This tests the pipeline, NOT placement robustness.")
    else:
        if not a.train or not a.val:
            sys.exit("give --train and --val sessions (or --dry-run <session>)")
        for f in a.train:
            s = facedata.Session(f, a.stride)
            parts.append((s, s.keep, np.array([], int)))
        for f in a.val:
            s = facedata.Session(f, a.stride)
            parts.append((s, np.array([], int), s.keep))
    for s, tr, va in parts:
        print(f"  {s.name}: train {len(tr)}, held-out {len(va)}, time offset {s.lag * 1000:+.0f} ms (jaw r {s.lag_r:.2f})")

    Xtr = torch.from_numpy(np.concatenate([np.asarray(s.frames)[tr] for s, tr, _ in parts if len(tr)]))
    Ytr = torch.from_numpy(np.concatenate([s.y[tr] for s, tr, _ in parts if len(tr)]))
    Xva = np.concatenate([exact(s, va, roi, hflip, vflip, gamma) for s, _, va in parts])
    Yva = np.concatenate([s.y[va] for s, _, va in parts if len(va)])
    print(f"train frames {len(Xtr)}, held-out frames {len(Xva)}")

    # ---- models
    stock = facemodel.load_torch().to(dev).eval()
    for p in stock.parameters():
        p.requires_grad_(False)
    model = facemodel.load_torch().to(dev)
    mapped = torch.from_numpy(facemap.MAPPED).to(dev)
    swap = torch.from_numpy(facemap.swap_index()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(Xtr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    def val_loss(m):
        p = facemodel.predict(m, Xva, dev)
        d = np.abs(p - Yva)[:, facemap.MAPPED]
        return float(np.where(d < 0.05, 0.5 * d * d / 0.05, d - 0.025).mean())

    pred_stock = facemodel.predict(stock, Xva, dev)
    best, best_state, hist = val_loss(model), copy.deepcopy(model.state_dict()), []
    print(f"epoch  0  held-out loss {best:.4f} (stock)")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = n = 0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            b = perm[i:i + a.batch]
            fr, y = Xtr[b].to(dev), Ytr[b].to(dev)
            x = augment(fr, roi, hflip, vflip, gamma, gen, a.aug)
            if a.mirror:
                m = torch.rand(len(b), device=dev, generator=gen) < 0.5
                x[m] = x[m].flip(-1)
                y[m] = y[m][:, swap]
            with torch.no_grad():
                ys = stock(x)
                if a.mirror:                              # the keep-as-stock target must be mirrored the same way
                    ys[m] = stock(x[m].flip(-1))[:, swap] if m.any() else ys[m]
            p = model(x)
            teach = F.smooth_l1_loss(p[:, mapped], y[:, mapped], beta=0.05)
            keep = F.mse_loss(p[:, ~mapped], ys[:, ~mapped])
            loss = teach + keep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += float(teach)
            n += 1
        v = val_loss(model)
        hist.append({"epoch": ep, "train": tot / max(n, 1), "held_out": v})
        flag = ""
        if v < best:
            best, best_state, flag = v, copy.deepcopy(model.state_dict()), "  <- best"
        print(f"epoch {ep:2d}  train {tot / max(n, 1):.4f}  held-out {v:.4f}{flag}  ({time.time() - t0:.0f} s)")
    model.load_state_dict(best_state)

    # ---- score on the held-out frames, with Baballonia's exact preprocessing
    pred_tuned = facemodel.predict(model, Xva, dev)
    print("\nHeld-out frames, against the Quest Pro (r: higher is better, err: lower is better)")
    res = facemodel.table({"stock": pred_stock, "tuned": pred_tuned}, Yva)
    un = ~facemap.MAPPED
    drift = float(np.abs(pred_tuned[:, un] - pred_stock[:, un]).mean())
    print(f"tongue outputs with no teacher moved by {drift:.4f} on average (should stay near 0)")

    # ---- export + prove
    onnx_path = out / "faceModel.onnx"
    worst = facemodel.export_onnx(model, onnx_path, Xva[:: max(1, len(Xva) // 32)])
    print(f"\nexported {onnx_path}  (ONNX vs trained model: max difference {worst:.1e})")
    report = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a),
              "preprocessing": {"roi": roi, "hflip": hflip, "vflip": vflip, "gamma": gamma},
              "sessions": [{"name": s.name, "train": int(len(tr)), "held_out": int(len(va)), "lag_s": s.lag,
                            "lag_r": s.lag_r} for s, tr, va in parts],
              "history": hist, "best_held_out_loss": best, "results": res,
              "untaught_output_drift": drift, "onnx_vs_torch_max_diff": worst,
              "dry_run": bool(a.dry_run)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"report  {out / 'report.json'}")


if __name__ == "__main__":
    main()
