# Eye-model pre-training (layout, started 2026-10-15)

**Goal:** a gaze/lid model that is PRE-TRAINED on large public near-eye IR datasets, then FINE-TUNED on the user's own
short calibration, and dropped into Baballonia in place of the per-user model it trains from scratch each time.
Baballonia's own figure is ~5°; research models trained on big near-eye sets reach ~2° (VRGaze 1.84°, NVGaze 2.06°).
Bigscreen's 2e is reported to do the same thing (pre-trained + per-user fine-tune).

**Success = measured, not assumed:** on the tester's recorded calibration footage (A1 grid, `tools/a1grid.py` +
`tools/eyerec.py`), the fine-tuned model beats Baballonia's stock per-user model on angular error and on stability
(jitter at fixation), at the same input. Same split, same frames, both models.

## The contract we must match (read from source 2026-10-15)
Baballonia loads `Documents\ProjectBabble\Models\eyeModel.onnx` (or a per-user tuned model) and feeds it:
- **input** `"input"`: float `[batch, 8, 128, 128]`. Per eye, the 4 most recent frames, newest first
  (`ImageCollector.cs`). Each frame is the eye crop (user's ROI in `CameraSettings`) → grayscale → 128×128 →
  `EqualizeHist`. Channels 0-3 are routed to the LEFT-eye heads and 4-7 to the RIGHT-eye heads
  (`MultiInputMergedMicroChad` conv1). Note: Baballonia swaps L/R before stacking ("inference requires them that way").
- **output** `"output"`: sigmoid values, grouped per eye. BabbleTrainer heads per side: gaze pitch, gaze yaw, lid,
  widen, squeeze, brow (`main.py` ~L230); Baballonia's eye pipeline uses 6 (pitch/yaw/lid per eye,
  `Utils.EyeRawExpressions`). ONNX metadata `blendshape_names` (JSON list) names them. Exported with opset 18,
  dynamic batch (`merge_models`, `main.py` L234-285).
- **architecture**: `MicroChad` (`BabbleTrainer/models.py`), 6 conv layers 4→14→21→32→47→70→106, max-pools,
  global max-pool, linear head + sigmoid. About 0.15 M parameters per eye per head-group: tiny, so pre-training
  needs a careful recipe (or a bigger teacher distilled into it, see `pretrain/`).
- ⚠️ To verify before training: the exact head grouping/sizes BabbleTrainer uses for eyes, the label ranges
  (0-1 ↔ which degrees), and the L/R swap direction, by running the stock trainer on one capture.

## Stages (folders)
| folder | does | status |
|---|---|---|
| `DATASETS.md` | registry: what each dataset is, size, access, licence, how close to our cameras | written |
| `prep/` | dataset → our contract (128×128, hist-eq, labels mapped to Baballonia's 0-1 ranges) | **VRGaze done** (`vrgaze.py`, 2 subjects prepped) |
| `pretrain/` | train MicroChad (and optionally a larger teacher → distil) on the prepped sets | layout only |
| `finetune/` | read Baballonia's calibration capture for one user, fine-tune the pre-trained weights | layout only |
| `export/` | merge L/R heads → ONNX matching the contract (+ `blendshape_names`), verify it loads in Baballonia | layout only |
| `eval/` | A1-grid angular error + fixation jitter, stock vs ours, on recorded footage | layout only |

Data lives in `EyeTrackingLab/data/` (gitignored, ~100 GB budget). Code + results tables are tracked.

## Order of work
1. Install PyTorch (CUDA, RTX 5070) + onnx + onnxruntime in a venv for `training/`.
2. Download ONE VRGaze subject (~2 GB) and build `prep/` against it end to end. Look at the crops.
3. Run the STOCK BabbleTrainer on a Baballonia calibration capture, to pin down the exact head layout and label ranges.
4. Pre-train on VRGaze (all 61 train subjects, 68.5 GB); evaluate on its 7 val subjects (8.3 GB).
5. When the eye cameras exist: record calibration + A1 grid, fine-tune, export, A/B against stock in Baballonia.
6. Optional: add OpenEDS2020 / TEyeD for lids and robustness, if licences allow.

## Domain gap to plan for
VRGaze (sample viewed 10-15) is off-axis near-eye IR from BELOW, with a ring of LED glints. That's close to the ETVR
Quest 3 mount (camera at the temple side, mid-height, 4 LEDs). Differences: camera angle (below vs side), resolution
and blur of our OV2640 at 240×240 JPEG, 130° lens distortion, glint pattern. Plan: augment (rotation, flips for the
side angle, blur/JPEG/noise, brightness; Baballonia's ±22 px shift) and always finish with the per-user fine-tune.

## Progress
- 2026-10-15: venv `training/.venv` (torch 2.11 cu128, RTX 5070 CUDA ok; onnx 1.23, onnxruntime 1.31, cv2 5.0).
- 2026-10-15: VRGaze format: per pair `image_l`/`image_r` (R stored mirrored), ONE binocular `gaze_yaw/pitch_degrees` (±23.5°), 400×400; ~1800 labelled pairs per session sampled every ~24 raw frames (NOT consecutive → pre-training stacks repeat one frame).
- 2026-10-15: VRGaze labels blinks with a gaze too. Per-eye blink detection by pupil blob was unreliable, so pre-training keeps only pairs where BOTH eyes show a clear pupil (78-84% of frames; 64 random kept pairs checked by eye: all open). Lids are left to the per-user fine-tune.
- Label mapping (BabbleTrainer data.py): gaze 0-1 = (deg + 45) / 90; lid 0.3 open / 1 closed / 0.7 squint / 0 wide. Signs + L/R orientation still to verify against a real Baballonia calibration.
- 2026-10-15: pipeline smoke-tested (pack.py memmap, train.py ~11 steps/s at batch 512 on the RTX 5070, fewshot.py). Calibrated on 200 own frames, error stayed ~18° even WITHIN the same session (= no better than always guessing centre).
- 🔴 2026-10-15 ROOT CAUSE: VRGaze's main labels are mostly the TARGET position, not true gaze (the authors: fixation "cannot be guaranteed"). Pupil position (their features file) explains only R² 0.40-0.69 of metadata_train labels (residual 5-8°), vs R² 0.81-0.98 (residual 1.5-3.8°) for metadata_*_static_points.csv. Static points are only ~85 per subject (~5 k train pairs total). `prep/label_check.py` reproduces this.
  - So: never score against the main labels (use static points), and don't pre-train on them as-is. Options: pseudo-labels (per-session fit of pupil/glint features → gaze on its static points, applied to every frame), clean-only (static points), or GazeShift-style unsupervised.
- ✅ 2026-10-15 PREVIEW (pre-trained on 16 subjects, 12k steps; `preview_pipeline.py`, `runs/preview_fewshot`), 7 held-out subjects, error vs clean static points:

  | K calibration frames | test | scratch (Baballonia-style) | pre-trained + fine-tune |
  |---|---|---|---|
  | 50 | other sessions (re-wear) | 4.14° | **2.78°** (−33%) |
  | 200 | other sessions (re-wear) | 3.83° | **2.69°** (−30%) |
  | 50 | same session | 2.49° | **2.10°** |
  | 200 | same session | 1.85° | **1.64°** |

  Fine-tune beat scratch for 7/7 subjects on re-wear at both K. Zero-shot (no calibration) 6.5°. Pre-trained K=50 beats scratch K=200. Same-session has a mild leak (those static points also fitted the session's pseudo-label mapping). Pre-train val error plateaued ~6.6° from step 2k (16 subjects). The full run (61 subjects, 60k steps) is running via run_pretrain_pipeline.py.
- ✅ 2026-10-16 FULL RUN (61 subjects, 1.59 M eye images, 60k steps, `runs/pre1`; `runs/fewshot_pre1/fewshot.json`), 7 held-out subjects, error vs clean static points:

  | K | test | zero-shot (no calibration) | scratch (Baballonia-style) | pre-trained + fine-tune |
  |---|---|---|---|---|
  | 50 | re-wear | 4.70° | 4.36° | **2.54°** (−42%) |
  | 200 | re-wear | 4.70° | 3.79° | **2.46°** (−35%) |
  | 1000 | re-wear | 4.70° | 3.75° | **2.46°** (−34%) |
  | 50 | same session | 4.54° | 2.83° | **2.11°** |
  | 200 | same session | 4.52° | 1.82° | **1.69°** |
  | 1000 | same session | 4.51° | 1.71° | **1.58°** |

  Fine-tune beat scratch on re-wear for 7/7 subjects at every K. Vs the 16-subject preview: zero-shot 6.54 → 4.70°, re-wear K=50 2.78 → 2.54°. Fine-tune plateaus at ~2.46° from K=200 (re-wear floor = headset slip).
- 🪞 2026-10-22 MIRROR + HINT TESTS (`eval/mirror_hint.py`, results `runs/mirror_hint/*.json`; `train.py --eye`, `--hint`). Pre-training runs 15k steps each (pre1's val plateaued by ~14k), one seed, 7 held-out people: gaps under ~0.5° may be noise.
  - **Per person, a mirrored eye does NOT replace the other eye's calibration** (K=200, calibrate one eye, test both on re-wear):

    | | same eye | other (mirrored) eye |
    |---|---|---|
    | scratch | 3.6-4.0° | **15.0°** |
    | fine-tune of pre1 | 2.3-2.5° | **4.9-5.9°** |

    Varies by person: 2 of 7 carry over well, 2 of 7 are 3-4× worse. Faces + headset fit are asymmetric → keep one calibration per eye (as Baballonia does).
  - **As pre-training data, mirrored eyes help partly** (zero-shot val, no calibration): left-only 5.32° L / 5.77° R; right-only 6.60° L / 4.59° R; both 4.84° L / 5.17° R. Adding mirrored rights helped the left eyes, but right eyes did best on a right-only model. Keep training on both (as pre1 does).
  - **The colour-the-parts idea as a pupil-mask channel (`--hint`): no gain.** Mean over 7 people:

    | | K=50 grey | K=50 hint | K=200 grey | K=200 hint |
    |---|---|---|---|---|
    | zero-shot | 4.94° | 4.93° | 4.94° | 4.93° |
    | scratch | 4.13° | 4.72° (worse 6/7) | 3.72° | 3.95° |
    | fine-tune | 2.70° | 2.60° | 2.49° | 2.51° |

    The mask is right on ~41/48 frames (a plain darkest-2% threshold failed: hist-eq makes the background as dark as the pupil, see `pupil_hint`). Here it costs one of the 4 frame channels. A CNN can find a dark pupil by itself, so a hint only pays where the model CAN'T see the part itself; not worth changing the ONNX contract for. Untested: hints for lids/glints, or as a 5th channel.
