"""The key test before any hardware: does pre-training help a NEW person's calibration?

For each held-out VRGaze subject (never seen in pre-training): take K calibration frames from their FIRST session,
then train the eye model two ways with the same frames and the same budget:
  scratch   - random init, as Baballonia does today with its per-user model
  finetune  - start from the pre-trained weights
and also score the pre-trained model untouched (zero-shot). Test on all of that person's OTHER sessions (later
re-wears of the headset, so slip is included). Report angular error in degrees.

    python training/eval/fewshot.py --val data/prepped/vrgaze_val --pretrained runs/pre1/best.pth --out runs/fewshot1
    python training/eval/fewshot.py ... --k 50 200 1000 --steps 1500

Calibration frames are spread over the gaze range (farthest-point sampling on the labels), like a dot grid would.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pretrain"))
from train import DEG, MicroChad, augment, stack4  # noqa: E402


def norm(deg):
    return np.clip((deg + 45) / 90, 0, 1).astype(np.float32)


def load_subject(path):
    """Both eyes as separate samples. Calibration labels = the pseudo labels (clean, every frame, like a dot
    calibration); test labels = VRGaze's static points only (clean). Main labels (mostly targets) are not used."""
    ps = np.load(path.with_suffix(".pseudo.npz"))
    with np.load(path) as d:
        v = d["valid_left"]
        seqs = d["seq"][v]
        imgs = np.concatenate([d["left"][v], d["right"][v]])
    two = lambda a: np.concatenate([a[v], a[v]])
    # second half = the mirrored right images: yaw reversed (see pretrain/pack.py)
    mirror = lambda a: np.concatenate([a[v], -a[v]])
    cal_labs = np.stack([norm(two(ps["pseudo_pitch"])), norm(mirror(ps["pseudo_yaw"]))], 1)
    test_labs = np.stack([norm(two(ps["static_pitch"])), norm(mirror(ps["static_yaw"]))], 1)
    return imgs, cal_labs, two(ps["pseudo_ok"]), test_labs, two(ps["is_static"]), np.concatenate([seqs, seqs])


def spread(labels, k, rng):
    """Farthest-point sample k indices over the (pitch, yaw) labels: a calibration grid, not a random clump."""
    k = min(k, len(labels))
    chosen = [int(rng.integers(len(labels)))]
    d = np.linalg.norm(labels - labels[chosen[0]], axis=1)
    for _ in range(k - 1):
        i = int(d.argmax())
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(labels - labels[i], axis=1))
    return np.asarray(chosen)


@torch.no_grad()
def score(model, imgs, labs, dev, inp=stack4):
    model.eval()
    errs = []
    for i in range(0, len(imgs), 4096):
        x = inp(torch.from_numpy(imgs[i:i + 4096]).to(dev).float().unsqueeze(1) / 255.0)
        errs.append((torch.linalg.vector_norm(model(x) - torch.from_numpy(labs[i:i + 4096]).to(dev), dim=1) * DEG).cpu())
    model.train()
    e = torch.cat(errs)
    return float(e.mean()), float(e.median())


def fit(model, imgs, labs, steps, lr, dev, seed, inp=stack4):
    x_all = torch.from_numpy(imgs).to(dev)
    y_all = torch.from_numpy(labs).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator(device=dev).manual_seed(seed)
    batch = min(128, len(imgs))
    for _ in range(steps):
        idx = torch.randint(0, len(imgs), (batch,), device=dev, generator=gen)
        x, y = augment(x_all[idx], y_all[idx], gen)
        loss = F.smooth_l1_loss(model(inp(x)), y, beta=0.02)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[50, 200, 1000])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = torch.load(a.pretrained, map_location=dev)
    rows = []
    for path in sorted(Path(a.val).glob("id*.npz")):
        if path.name.endswith((".tmp.npz", ".pseudo.npz")) or not path.with_suffix(".pseudo.npz").exists():
            continue
        imgs, cal_labs, cal_ok, test_labs, is_static, seq = load_subject(path)
        # calibrate on the session with the most usable frames; test on every OTHER session's static points
        sessions = sorted(set(seq), key=lambda s: -(cal_ok & (seq == s)).sum())
        cal_pool = (seq == sessions[0]) & cal_ok
        test = (seq != sessions[0]) & is_static
        if test.sum() == 0:
            print(f"{path.stem}: no other-session static points, skipped")
            continue
        zs = MicroChad(out_count=2).to(dev)
        zs.load_state_dict(pre)
        for k in a.k:
            rng = np.random.default_rng(a.seed)
            pool_idx = np.flatnonzero(cal_pool)
            cal = pool_idx[spread(cal_labs[pool_idx], k, rng)]
            # same-session test = that session's static points not used for calibration (no re-wear). Mild leak:
            # those points also fitted the session's pseudo-label mapping. Other sessions = re-wear, no leak.
            same = (seq == sessions[0]) & is_static
            same[cal] = False
            torch.manual_seed(a.seed)
            scratch = fit(MicroChad(out_count=2).to(dev), imgs[cal], cal_labs[cal], a.steps, 2e-3, dev, a.seed)
            ft = MicroChad(out_count=2).to(dev)
            ft.load_state_dict(pre)
            ft = fit(ft, imgs[cal], cal_labs[cal], a.steps, 3e-4, dev, a.seed)
            r = {"subject": path.stem, "k": k, "test_images": int(test.sum()), "same_session_images": int(same.sum())}
            for name, m in (("zero_shot", zs), ("scratch", scratch), ("finetune", ft)):
                r[name] = score(m, imgs[test], test_labs[test], dev)[0]
                r[name + "_same"] = score(m, imgs[same], test_labs[same], dev)[0] if same.any() else float("nan")
            rows.append(r)
            print(f"{path.stem} K={k:5d} | other sessions: zero-shot {r['zero_shot']:.2f}° scratch {r['scratch']:.2f}° "
                  f"fine-tune {r['finetune']:.2f}° | same session: zero-shot {r['zero_shot_same']:.2f}° scratch "
                  f"{r['scratch_same']:.2f}° fine-tune {r['finetune_same']:.2f}°", flush=True)
        (out / "fewshot.json").write_text(json.dumps(rows, indent=1))
    print("\nmean over subjects (angular error, degrees; lower is better):")
    for k in a.k:
        rk = [r for r in rows if r["k"] == k]
        if rk:
            m = {m: np.mean([r[m] for r in rk]) for m in rk[0] if m not in ("subject", "k")}
            print(f"  K={k:5d} other sessions: zero-shot {m['zero_shot']:.2f} · scratch {m['scratch']:.2f} · "
                  f"fine-tune {m['finetune']:.2f} | same session: zero-shot {m['zero_shot_same']:.2f} · scratch "
                  f"{m['scratch_same']:.2f} · fine-tune {m['finetune_same']:.2f}  ({len(rk)} subjects)")


if __name__ == "__main__":
    main()
