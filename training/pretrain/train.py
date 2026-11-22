"""Pre-train Baballonia's eye model (MicroChad, imported unchanged from BabbleTrainer so the export stays
compatible) on packed VRGaze gaze data.

    python training/pretrain/train.py --train data/packed/vrgaze_train --val data/packed/vrgaze_val --out runs/pre1
    python training/pretrain/train.py ... --steps 300            smoke test

One shared model for both eyes (VRGaze's right image is already mirrored to look like a left eye), gaze head only
(pitch, yaw in Baballonia's 0-1 = (deg + 45) / 90). Lids are learned later, in the per-user fine-tune.
Input = the same frame repeated into Baballonia's 4-frame stack (VRGaze frames aren't consecutive).
Augmentation on the GPU per batch: shift (Baballonia's own ±22 px), rotation, scale, brightness/contrast, blur,
noise. No flips (see augment()). Right-eye samples are VRGaze's mirrored images, with yaw mirrored (pack.py).
Metric: angular error in degrees ≈ 90 · |(Δpitch, Δyaw)| on the 0-1 labels.
"""
import argparse
import json
import math
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repos" / "BabbleTrainer"))
from models import MicroChad  # noqa: E402  (BabbleTrainer's own architecture)

DEG = 90.0   # 1.0 of a normalized label = 90 degrees


class Packed:
    def __init__(self, folder, eye="both"):
        f = Path(folder)
        self.images = np.load(f / "images.npy", mmap_mode="r")
        self.labels = np.load(f / "labels.npy")
        self.subject = np.load(f / "subject.npy")
        self.eye = np.load(f / "eye.npy")
        # --eye left/right: train on one eye only (right = VRGaze's mirrored image), to test whether a mirrored
        # eye can stand in for the other one
        self.keep = None if eye == "both" else np.flatnonzero(self.eye == ("left", "right").index(eye))
        self.n = len(self.labels) if self.keep is None else len(self.keep)

    def batch(self, idx):
        if self.keep is not None:
            idx = self.keep[idx]
        idx = np.sort(idx)
        return torch.from_numpy(np.ascontiguousarray(self.images[idx])), torch.from_numpy(self.labels[idx])


def prefetch(data, batch, steps, seed):
    """Background thread: random batches from the memmap, so disk reads overlap with GPU work."""
    q = queue.Queue(maxsize=8)

    def run():
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            q.put(data.batch(rng.integers(0, data.n, batch)))
        q.put(None)
    threading.Thread(target=run, daemon=True).start()
    while (b := q.get()) is not None:
        yield b


def augment(x, y, gen):
    """x uint8 [B,128,128] on GPU -> float [B,1,128,128] in 0-1; y [B,2] (pitch, yaw), yaw mirrored on flips."""
    b = x.shape[0]
    x = x.float().unsqueeze(1) / 255.0
    dev = x.device
    # No horizontal flips: the camera sees the eye at an angle, so image x mixes yaw AND pitch (VRGaze val,
    # corr(pupil x, pitch) +0.45); a flip changes the pitch relation too, which no label edit can undo.
    ang =(torch.rand(b, device=dev, generator=gen) - 0.5) * math.radians(30)          # ±15°
    scale = 1 + (torch.rand(b, device=dev, generator=gen) - 0.5) * 0.2                 # ±10 %
    shift = (torch.rand(b, 2, device=dev, generator=gen) - 0.5) * (2 * 22 / 64)       # ±22 px of 128
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    blur = torch.rand(b, device=dev, generator=gen) < 0.3
    if blur.any():
        x[blur] = F.avg_pool2d(x[blur], 3, 1, 1)
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.6
    bright = (torch.rand(b, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.3
    x = (x - 0.5) * contrast + 0.5 + bright
    x = x + torch.randn(x.shape, device=dev, generator=gen) * 0.03
    return x.clamp(0, 1), y


def stack4(x):
    return x.expand(-1, 4, -1, -1)                 # Baballonia's 4-frame stack, one frame repeated


def _grow(x, k):
    return F.max_pool2d(F.pad(x, (k // 2,) * 4, mode="replicate"), k, 1)


def pupil_hint(x, k=31, r=14):
    """The 'colour the parts' idea, as a mask: 1 on the pupil blob, else 0. Computed AFTER augmentation, so it
    is as imperfect as a live detector would be (10-22 check on 48 val frames: on the pupil in ~41; misses are
    blown-out frames, lashes, border hair). A plain darkest-2% threshold FAILED: hist-eq makes the noisy background
    as dark as the pupil. So: black top-hat (closing - image = small dark blobs, not big dark areas), border
    ignored, strongest blob-sized spot, then only the dark pixels within r px of it."""
    b = F.avg_pool2d(x, 5, 1, 2, count_include_pad=False)
    th = -_grow(-_grow(b, k), k) - b
    th[..., :6, :] = th[..., -6:, :] = 0
    th[..., :, :6] = th[..., :, -6:] = 0
    at = F.avg_pool2d(th, 9, 1, 4, count_include_pad=False).flatten(1).argmax(1)
    w = x.shape[-1]
    cy, cx = (at // w).view(-1, 1, 1, 1), (at % w).view(-1, 1, 1, 1)
    yy = torch.arange(x.shape[-2], device=x.device).view(1, 1, -1, 1)
    xx = torch.arange(w, device=x.device).view(1, 1, 1, -1)
    peak = th.flatten(1).max(1).values.view(-1, 1, 1, 1)
    return (((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r) & (th >= 0.4 * peak)).float()


def stack_hint(x):
    return torch.cat([x, x, x, pupil_hint(x)], 1)   # 3 copies of the frame + the hint, same 4-channel model


@torch.no_grad()
def evaluate(model, data, dev, limit=None, batch=2048, inp=stack4):
    model.eval()
    n = data.n if limit is None else min(limit, data.n)
    idx = np.linspace(0, data.n - 1, n).astype(int)
    errs = []
    for i in range(0, n, batch):
        x, y = data.batch(idx[i:i + batch])
        x = inp(x.to(dev).float().unsqueeze(1) / 255.0)
        p = model(x)
        errs.append((torch.linalg.vector_norm(p - y.to(dev), dim=1) * DEG).cpu())
    model.train()
    e = torch.cat(errs)
    return float(e.mean()), float(e.median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eye", choices=["both", "left", "right"], default="both", help="train on one eye only")
    ap.add_argument("--hint", action="store_true", help="replace frame 4 of the stack with a pupil-blob mask")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    inp = stack_hint if a.hint else stack4
    train, val = Packed(a.train, a.eye), Packed(a.val)
    print(f"train {train.n} eye images, val {val.n}", flush=True)

    model = MicroChad(out_count=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    best, log, t0 = float("inf"), [], time.time()
    for step, (x, y) in enumerate(prefetch(train, a.batch, a.steps, a.seed), 1):
        x, y = augment(x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), gen)
        loss = F.smooth_l1_loss(model(inp(x)), y, beta=0.02)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            mean, median = evaluate(model, val, dev, limit=20000, inp=inp)
            log.append({"step": step, "loss": loss.item(), "val_mean_deg": mean, "val_median_deg": median,
                        "minutes": (time.time() - t0) / 60})
            print(f"step {step:6d} loss {loss.item():.4f} · val error mean {mean:.2f}° median {median:.2f}° · "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if mean < best:
                best = mean
                torch.save(model.state_dict(), out / "best.pth")
            (out / "log.json").write_text(json.dumps(log, indent=1))
    torch.save(model.state_dict(), out / "last.pth")
    mean, median = evaluate(model, val, dev, inp=inp)
    print(f"FINAL full val: mean {mean:.2f}° median {median:.2f}° (best checkpoint {best:.2f}° on 20k)")


if __name__ == "__main__":
    main()
