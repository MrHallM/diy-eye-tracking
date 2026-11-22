"""Cleaner gaze labels for VRGaze. Its main labels are mostly the TARGET position (label_check.py: pupil explains
only R² 0.40-0.69), but each session has ~20 static points with trustworthy gaze. Per session and eye, fit a small
mapping from VRGaze's own per-frame eye features to gaze on those points, then label every frame with it.

    python training/prep/pseudolabel.py data/vrgaze data/prepped/vrgaze --split train
    python training/prep/pseudolabel.py data/vrgaze data/prepped/vrgaze_val --split val

Features (metadata_<split>_features.csv, VRGaze's detector): pupil centre, and the pupil minus the centroid of the
visible LED glints (the classic pupil-corneal-reflection vector, less sensitive to small headset slip). Per session
and eye, ridge regression on [pupil] and on [pupil, pcr], linear or quadratic (quadratic only with >= 15 points);
the variant with the lowest leave-one-out error on the static points wins. Labels of the two eyes are averaged when
both are usable (VRGaze's gaze is binocular).

Writes <prepped>/<subject>.pseudo.npz aligned row-for-row with <subject>.npz:
  pseudo_yaw, pseudo_pitch (deg, float32), pseudo_ok (bool), is_static (bool), static_yaw, static_pitch (deg, NaN
  if not a static point), loo_deg (float32: leave-one-out error of the mapping used for that row's session).
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_features(path, subjects):
    df = pd.read_csv(path, usecols=["image", "pupils", "glints", "glints_visibility"])
    df["image"] = df["image"].str.replace(r"^(train|val)/", "", regex=True).str.replace("_fliped", "_flipped")
    df = df[df["image"].str.split("/").str[0].isin(subjects)]
    pupil = np.array([[float(v) for v in NUM.findall(s)[:2]] for s in df["pupils"]], np.float64)
    glints = np.array([[float(v) for v in NUM.findall(s)][:20] for s in df["glints"]], np.float64).reshape(-1, 10, 2)
    vis = np.array([[float(v) for v in NUM.findall(s)][:10] for s in df["glints_visibility"]], np.float64) > 0.5
    n_vis = vis.sum(1)
    centroid = np.where(n_vis[:, None] > 0, (glints * vis[..., None]).sum(1) / np.maximum(n_vis, 1)[:, None], np.nan)
    pcr = np.where(n_vis[:, None] >= 4, pupil - centroid, np.nan)       # need >= 4 glints for a stable centroid
    return {name: (pupil[i], pcr[i]) for i, name in enumerate(df["image"])}


def design(f, quad):
    cols = [f, np.ones((len(f), 1))]
    if quad:
        cols.append(np.stack([f[:, i] * f[:, j] for i in range(f.shape[1]) for j in range(i, f.shape[1])], 1))
    return np.concatenate(cols, 1)


def ridge(X, y, alpha=1e-3):
    """Ridge on standardized features; the intercept column (all ones, from design()) is neither scaled nor
    penalized. Returns a predictor for new design rows."""
    icpt = np.all(X == 1, 0)
    mu = np.where(icpt, 0.0, X.mean(0))
    sd = np.where(icpt, 1.0, X.std(0) + 1e-9)
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + np.diag(np.where(icpt, 0.0, alpha * len(X))), Xs.T @ y)
    return lambda Z: ((Z - mu) / sd) @ w


def loo(F, Y, quad):
    """Mean leave-one-out error (degrees) of the mapping on the static points."""
    errs = []
    for i in range(len(F)):
        m = np.ones(len(F), bool)
        m[i] = False
        X = design(F[m], quad)
        if X.shape[0] <= X.shape[1] + 2:
            return np.inf
        pred = ridge(X, Y[m])(design(F[i:i + 1], quad))
        errs.append(float(np.linalg.norm(pred - Y[i])))
    return float(np.mean(errs))


def fit_session(static_feats, static_y, all_feats):
    """static_feats/all_feats: dict variant -> [n, d] arrays. Returns (pred [N,2] or None, loo_deg)."""
    best = (np.inf, None, None)
    for variant in ("pupil", "pupil+pcr"):
        F = static_feats[variant]
        ok = ~np.isnan(F).any(1)
        if ok.sum() < 6:
            continue
        for quad in (False, True):
            if quad and ok.sum() < 15:
                continue
            e = loo(F[ok], static_y[ok], quad)
            if e < best[0]:
                best = (e, variant, quad)
    e, variant, quad = best
    if variant is None:
        return None, np.inf
    F = static_feats[variant]
    ok = ~np.isnan(F).any(1)
    model = ridge(design(F[ok], quad), static_y[ok])
    A = all_feats[variant]
    pred = np.full((len(A), 2), np.nan)
    good = ~np.isnan(A).any(1)
    pred[good] = model(design(A[good], quad))
    return pred, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("prepped", type=Path)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--max-loo", type=float, default=4.0, help="sessions whose mapping is worse than this are dropped")
    a = ap.parse_args()

    npzs = sorted(p for p in a.prepped.glob("id*.npz") if not p.name.endswith((".tmp.npz", ".pseudo.npz")))
    subjects = {p.stem for p in npzs}
    static = pd.read_csv(a.src / f"metadata_{a.split}_static_points.csv")
    static["image_l"] = static["image_l"].str.replace("_fliped", "_flipped")
    static = static[static["image_l"].str.split("/").str[0].isin(subjects)]
    static_by_key = {r.image_l: (r.gaze_yaw_degrees, r.gaze_pitch_degrees) for r in static.itertuples()}
    feats_name = "metadata_train_features.csv" if a.split == "train" else "metadata_val_features.csv"
    print(f"reading {feats_name} for {len(subjects)} subjects ...", flush=True)
    feats = parse_features(a.src / feats_name, subjects)

    all_loo = []
    for p in npzs:
        with np.load(p) as d:
            seq, frame = d["seq"], d["frame"]
        n = len(seq)
        out = {k: np.full(n, np.nan, np.float32) for k in ("pseudo_yaw", "pseudo_pitch", "static_yaw", "static_pitch",
                                                           "loo_deg")}
        is_static = np.zeros(n, bool)
        per_eye = {}
        rows_by_seq = defaultdict(list)
        for i in range(n):
            rows_by_seq[seq[i]].append(i)
        for s, idx in rows_by_seq.items():
            idx = np.asarray(idx)
            names_l = [f"{p.stem}/{s}/{frame[i]}_L.jpg" for i in idx]
            keys = [static_by_key.get(nm) for nm in names_l]
            st = np.asarray([k is not None for k in keys])
            is_static[idx[st]] = True
            if st.any():
                sy = np.asarray([k for k in keys if k is not None], np.float64)
                out["static_yaw"][idx[st]] = sy[:, 0]
                out["static_pitch"][idx[st]] = sy[:, 1]
            preds, errs = [], []
            for eye, suffix in (("L", "_L.jpg"), ("R", "_R_flipped.jpg")):
                names = [f"{p.stem}/{s}/{frame[i]}{suffix}" for i in idx]
                f = [feats.get(nm, (np.full(2, np.nan), np.full(2, np.nan))) for nm in names]
                pupil = np.asarray([x[0] for x in f])
                pcr = np.asarray([x[1] for x in f])
                variants = {"pupil": pupil, "pupil+pcr": np.concatenate([pupil, pcr], 1)}
                if st.sum() < 6:
                    continue
                pred, e = fit_session({k: v[st] for k, v in variants.items()}, sy, variants)
                if pred is not None and e <= a.max_loo:
                    preds.append(pred)
                    errs.append(e)
                per_eye.setdefault(eye, []).append(e)
            if preds:
                avg = np.nanmean(np.stack(preds), 0)
                out["pseudo_yaw"][idx] = avg[:, 0]
                out["pseudo_pitch"][idx] = avg[:, 1]
                out["loo_deg"][idx] = float(np.mean(errs))
                all_loo.append(float(np.mean(errs)))
        ok = ~np.isnan(out["pseudo_yaw"]) & ~np.isnan(out["pseudo_pitch"])
        np.savez(p.with_suffix(".pseudo.npz"), pseudo_ok=ok, is_static=is_static, **out)
        loo_txt = ", ".join(f"{e}: {np.median(v):.2f}°" for e, v in per_eye.items()) or "no sessions fitted"
        print(f"{p.stem}: {len(rows_by_seq)} sessions, labelled {ok.mean() * 100:.1f}% of frames, "
              f"median leave-one-out error per eye ({loo_txt})", flush=True)
    if all_loo:
        print(f"\nALL: {len(all_loo)} sessions kept, leave-one-out error median {np.median(all_loo):.2f}°, "
              f"p90 {np.percentile(all_loo, 90):.2f}°")


if __name__ == "__main__":
    main()
