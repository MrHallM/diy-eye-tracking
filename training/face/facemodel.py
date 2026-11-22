"""Babble faceModel.onnx <-> PyTorch, and the shared scoring code.

The stock model is an EfficientNet-style net (5.9 M weights, batch-norm already folded into the convolutions),
input "x.1" [1, 1, 224, 224] gray 0..1, output [1, 45] with no final squashing.
"""
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch

import facemap

STOCK = Path(__file__).resolve().parents[2] / "builds" / "baballonia-uncapped" / "faceModel.onnx"
INPUT_NAME = "x.1"


def load_torch(path=STOCK):
    """ONNX -> trainable torch module. One fix-up: the file's Clip nodes carry an empty 'max' input, which
    onnx2torch reads as a dynamic value and refuses; dropping the empty slot is the same maths."""
    from onnx2torch import convert
    m = onnx.load(str(path))
    for n in m.graph.node:
        if n.op_type == "Clip":
            ins = list(n.input)
            while ins and ins[-1] == "":
                ins.pop()
            del n.input[:]
            n.input.extend(ins)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return convert(m)


def export_onnx(model, path, check_x=None):
    """torch -> ONNX with the stock file's input name and shape, then prove ONNX == torch on real frames."""
    import onnxruntime as ort
    model = model.eval().cpu()
    dummy = torch.rand(1, 1, 224, 224)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, dummy, str(path), input_names=[INPUT_NAME], output_names=["out"],
                          opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    i = sess.get_inputs()[0]
    assert i.name == INPUT_NAME and list(i.shape) == [1, 1, 224, 224], (i.name, i.shape)
    xs = check_x if check_x is not None else dummy.numpy()
    worst = 0.0
    with torch.no_grad():
        for x in xs[:32]:
            x = np.asarray(x, np.float32).reshape(1, 1, 224, 224)
            a = sess.run(None, {INPUT_NAME: x})[0]
            assert a.shape == (1, 45), a.shape
            worst = max(worst, float(np.abs(a - model(torch.from_numpy(x)).numpy()).max()))
    assert worst < 2e-3, f"exported ONNX differs from the trained model by {worst}"
    return worst


@torch.no_grad()
def predict(model, xs, device, batch=128):
    """xs: float32 [N, 224, 224] already preprocessed -> [N, 45]"""
    model = model.eval().to(device)
    out = []
    for i in range(0, len(xs), batch):
        out.append(model(torch.from_numpy(xs[i:i + batch, None]).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 45), np.float32)


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else float("nan")


def score(pred, y, min_range=0.15):
    """Per output: r and mean absolute error against the Quest Pro. An output only counts toward the summary if the
    Quest Pro actually moved it in these frames (range >= min_range); otherwise r is meaningless."""
    rows, rs, maes = [], [], []
    for i, n in enumerate(facemap.BABBLE):
        if not facemap.MAPPED[i]:
            continue
        q, p = y[:, i], pred[:, i]
        rng = float(q.max() - q.min()) if len(q) else 0.0
        r, mae = _corr(q, p), float(np.abs(q - p).mean())
        active = rng >= min_range
        rows.append((n, r, mae, rng, active))
        if active:
            rs.append(r)
            maes.append(mae)
    return rows, (float(np.nanmean(rs)) if rs else float("nan")), (float(np.mean(maes)) if maes else float("nan"))


def table(named_preds, y, min_range=0.15):
    """named_preds: {"stock": pred, "tuned": pred, ...}. Prints one row per taught output."""
    scored = {k: score(p, y, min_range) for k, p in named_preds.items()}
    names = list(named_preds)
    head = f"{'output':22}{'QP range':>9}  " + "  ".join(f"{'r ' + k:>9}{'err ' + k:>11}" for k in names)
    print(head)
    first = scored[names[0]][0]
    for j, (n, _, _, rng, active) in enumerate(first):
        line = f"{n:22}{rng:9.2f}  " + "  ".join(f"{scored[k][0][j][1]:9.2f}{scored[k][0][j][2]:11.3f}" for k in names)
        print(line + ("" if active else "   (not exercised)"))
    print(f"{'MEAN over exercised':22}{'':9}  " + "  ".join(f"{scored[k][1]:9.2f}{scored[k][2]:11.3f}" for k in names))
    return {k: {"mean_r": scored[k][1], "mean_abs_err": scored[k][2],
                "per_output": {n: {"r": r, "abs_err": e, "qp_range": g, "exercised": bool(a)}
                               for n, r, e, g, a in scored[k][0]}} for k in names}
