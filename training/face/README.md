# face/ — teach Babble's mouth model with the Quest Pro

Goal (10-17): a mouth tracker that cares less about exactly how the camera sits, and a better face model overall.
The Quest Pro's face tracking is the teacher; Babble's stock `faceModel.onnx` is the student. Built 2026-11-10, dry-run
on the 10-17 recording. Run everything with `training\.venv\Scripts\python.exe`.

| file | job |
|---|---|
| `facemap.py` | Quest Pro's 70 weights → Babble's 45 outputs. 34 have a teacher; 11 tongue shapes don't and are held at the stock model's answer. |
| `facedata.py` | Loads a `qpteach.py` recording folder (a *session* = one way of wearing the headset). Lines the camera up with the Quest in time, and applies Baballonia's own crop / flip / gamma, read from its real settings file. |
| `facemodel.py` | ONNX ↔ PyTorch, scoring table, export with a check that the exported file equals the trained model. |
| `train_face.py` | Fine-tunes from the stock weights with placement augmentation (shift, zoom, tilt, brightness, blur, noise), judges by held-out sessions, exports `faceModel.onnx`. Installs nothing. |
| `eval_face.py` | Scores any models on any sessions; `--lr-check` (is left/right the right way round?); `--rest` (how far a resting face moves between re-wears = the placement baseline). |

## Dry run, 10-17 recording (95 s, one wear, Quest IR on)

`train_face.py --dry-run tests/QPT_2026-10-17/raw/p0 --out training/face/runs/dryrun_1017 --epochs 10` — 54 s on the 5070.

| on held-out frames | stock | fine-tuned |
|---|---|---|
| mean r against the Quest Pro (25 exercised outputs) | 0.16 | 0.81 |
| mean absolute error | 0.129 | 0.029 |
| smile L / R (r) | 0.65 / 0.39 | 0.99 / 0.99 |
| jaw forward (r) | −0.05 | 0.87 |
| pucker / funnel (r) | 0.14 / −0.11 | 0.99 / 0.98 |
| untaught tongue outputs, average change | — | 0.014 |

**What this does and doesn't show.** It shows the whole chain works (pairing → labels → training → export, exported file
identical to 1e-6) and that the camera image carries the information: smile-right, jaw-forward and pucker are all
learnable from the tester's camera. It does NOT show placement robustness or a real-world gain: the held-out frames are the last
30 % of each prompt from the SAME wear, the SAME lighting, with the Quest's IR on. Expect lower numbers on a different
wear. Weak even here: JawRight 0.39, UpperUp 0.5, LowerDown 0.5, StretchRight 0.47 (little data for each).

## Side finding: Baballonia's flip setting swaps left and right

`eval_face.py tests/QPT_2026-10-17/raw/p0 --lr-check`: with the image as the camera sends it, the stock model's
left/right follows the Quest Pro's (r +0.44). With `FaceCamera.UseHorizontalFlip = true`, which is what
`%APPDATA%\ProjectBabble\ApplicationData\LocalSettings.json` has now (file dated 10-27), it follows the OPPOSITE side
(r −0.47). So with the stock model, flip should be OFF. A model trained here is correct either way, because it learns
left/right from the Quest on the image Baballonia really feeds it — but then the flip setting must not change after training.

## Filter day (IR-cut filter, due Nov 15) — the real dataset

1. Tape the filter flat on the mouth-camera lens. Quest Pro Natural Facial Expressions ON, VD "Forward tracking data to PC" on, a VR app running.
2. Flicker check: `tools/qpteach.py record --out tests/QPT_<date>/raw/flicker`, then `tools/qpteach.py check` on it. Want mouth brightness jump ≈ 0.2 (it was 36 without the filter). If the frame edges still flicker, stop and tell Claude (plan B = domain-adversarial training).
3. Decide Baballonia's crop and flip FIRST and don't change them afterwards.
4. Record 6+ sessions, taking the headset off and putting it back on between each, a couple with the hinge deliberately a little off:
   `tools/qpteach.py record --loops 3 --out tests/QPT_<date>/raw/w1` (… `w2`, `w3` …). About 4.5 min each.
5. Train on all but two, hold two out:
   `training/face/train_face.py --train …/w1 …/w2 …/w3 …/w4 --val …/w5 …/w6 --out training/face/runs/<date>`
6. Read the table. Then `eval_face.py …/w1 … …/w6 --model tuned=…/faceModel.onnx --rest` for the placement spread, stock vs tuned.
7. Only if the held-out sessions are clearly better: back up the build's `faceModel.onnx`, copy the new one in, re-run Baballonia's calibration (output ranges change), try it in VRChat.

Placement baseline, possible any time before the filter (Quest face tracking OFF, no labels needed):
`tools/qpteach.py record --no-qp --out tests/PLACE_<date>/raw/w1` × 4-5 re-wears, then `eval_face.py … --rest`.

## Known limits

- The Quest Pro is the teacher, not the truth. Its tongue is poor; whatever it gets wrong, the student copies.
- 11 tongue outputs have no teacher.
- Mount geometry differs between the Quest Pro mount and the Quest 3 mount (see `hardware/mount_compare/`); the zoom/tilt/shift augmentation covers small differences only. Plan: match the view before recording, short top-up on the Quest 3.
- `--mirror` (flip + swap L/R targets) is only valid if the camera is on the midline. Test with and without on held-out sessions.
- `runs/` is git-ignored: the exported model is Babble's weights plus changes, not ours to publish.
