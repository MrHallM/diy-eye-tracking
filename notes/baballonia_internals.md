# Baballonia internals + community field notes (2026-10-10)

Read the source and community reports for our exact stack: DIY ETVR on a Beyond 1 (beymod), Baballonia, VRChat.
Every claim below cites a source. Lines marked (inference) are my reasoning, not something read.

## 1. The eye model is a tiny per-user CNN trained from scratch, not a pretrained general network
Source: `repos\BabbleTrainer` (a0b7641, 2026-05-30), `main.py` + `models.py` + `babble_data\babble_data.cpp`.
- **Network:** `MicroChad`, 6 conv layers (4→14→21→32→47→70→106 channels), max-pool after each, then a GLOBAL max pool and one linear layer (`models.py` l.18-43).
- **Fresh weights on every calibration.** No pretrained checkpoint is loaded; `torch.load` only reloads the temp `.pth` files it just wrote, for the ONNX merge (`main.py` l.250-257, 467).
- **Training schedule:** gaze 1000 steps per eye, batch 128 (`main.py` l.476/480); blink 1600 steps, batch 16, lr 5e-5, with noise augmentation (l.484/488); brow 1600 steps (l.492/496).
- **Input:** 4 channels = the current frame + 3 previous frames, 128×128, grayscale, histogram-equalized (`babble_data.cpp` l.199-210; export shape `main.py` l.264). Gaze labels are mapped as (deg+45)/90, so the output range is ±45° (l.364-365).
- **It is trained to tolerate headset shift.** In 30% of training samples it applies ±22 px shift (about 17% of the 128 px crop), ±12° rotation and ±10% scale (`babble_data.cpp` l.222-241). It also randomizes brightness ±0.2 and contrast ±0.6, and adds Gaussian blur. `data.py` names the shift augmentation "simulate headset movement" (l.270).
- **Consequences for us** (inference):
  - The literature's "calibration-free NN ~5°" comparison (Pupil Invisible) does not apply. Baballonia is a per-user fit trained on a few minutes of data.
  - The shift augmentation plus the global max pool make the model partly shift-invariant BY DESIGN. Plain crop re-centering (goal 1) may therefore gain less than the pye3d 5.8°/mm figure suggests. It could also fight the model if the model already reads pupil position relative to the lids/corners.
  - **Test S2** (shift the crop by known px, measure the gaze change) measures exactly this invariance. Run it BEFORE building goal 1.
  - **Re-running calibration = a new model.** That is the user-side slip fix today.
- **Where "~5°" came from:** README line 66 cites "Babble's figure". It has not been measured on our setup.

## 2. Slip: the Babble team's position (issue #221, open since 2026-02-21)
- **The user report:** "even a slight shift in VR headset position can offset eye tracking"; the eyes look up when the user looks straight ahead. Frooxius (Resonite dev) sees the same with a Vive Pro Eye feeding Baballonia, and says shifts happen "a lot".
- **SummerSigh (Babble member) rejects recentering:** "we don't intend to implement recentering but rather solve this on the model level." Their reason is DYNAMIC RANGE: an OUTPUT offset clips part of the model's 0-1 range. They are "working on an eye model that's resistant to headset shift".
  - Status 10-10: not in any release. The latest is v1.1.1.0rc6 (2026-03-12). No branch name matches either.
- **Erimelowo (contributor):** the initial calibration itself is hard to get centred, and shift compensation won't fix a bad start.
- **Our plan avoids their objection** (inference): our goal 1 moves the INPUT crop, not the output, so it does not clip the output range. It does inherit the limit that if part of the eye leaves the 240×240 frame, nothing recovers it.
- **Dead setting:** `AppSettings_RecenterAddress` (`/avatar/parameters/etvr_recenter`) is a UI field with NO consumer in `src\` (grep 10-10). It does nothing.
- **Official tip** (BabbleDocs PR #95): "the quality and reliability of eye tracking can be improved by slightly shifting the headset around during calibration". That is real slip variety in the training data. Also: recenter SteamVR facing forward first, calibrate in the dark, leave cameras uncropped in most cases, and alternate relaxed/tight closes during the blink step.

## 3. Goal 2 already has an upstream prototype: branch `feat/autocalibration`
- 2 commits by SummerSigh, 2026-02-27. Unmerged, 30 commits behind main, untouched since.
- `CalibrationService.OnExpressionChanged`: when enabled, every frame expands each expression's `Lower`/`Upper` to the running min/max (face expressions + `LeftEyeLid`/`RightEyeLid`), seeded at 0.5.
- **Weaknesses** (from reading the diff):
  - It only ever EXPANDS, with no decay. One glitch frame, a squint or a hand over the camera widens the range for good, until a manual reset.
  - It calls `SaveAsync()` on EVERY frame.
- Legacy ProjectBabble `osc_calibrate_filter.py` used slow decay. Blink19 uses the median of the last n peaks. Both are sturdier.
- **Our goal-2 work has a natural upstream target** (inference): a robust percentile/median-with-decay version of this same hook. Mind the license: Babble Software Distribution License 1.0 = non-commercial.

## 4. Other upstream activity worth knowing
- `expr-opt` branch (2026-09-01): "Use camera timing for independent One Euro filters" (#268).
- `BabbleCalibration` repo (updated 2026-09-09): a new Godot OpenVR/OpenXR eye-tracking calibrator app.
- Issue #260 (open): export calibration settings.

## 5. Beyond 1 + beymod specifics
- **beymod README (Lenrul, archived 2025-08-08):**
  - "if your eye is not exactly center from the optic, a small difference makes a big impact on tracking quality."
  - "may or may not work for everyone depending on your face structure".
  - A flat face can make the camera stick out past the cushion.
  - "it can get hot so prepare some eyedrops".
  - BOM: OV2640 130° 75 mm + XIAO ESP32S3 Sense + V4 IR mini kit.
- **Bigscreen's EyeTrackVR blog:** a ~$50 BOM, plugs into the Beyond's USB-C accessory port, "a few bugs and quirks". No accuracy figures.
- **No published accuracy, latency or blink-rate number exists for DIY ETVR on a Beyond 1 or 2** (searched 10-10). Our test plan would produce the first.
- **Fit note:** our Beyond 1 came with NO cushion (bought used 10-06). Slip and centring both depend on the cushion, so run the slip tests with the cushion that will actually be worn (inference).

## Sources
- https://github.com/Project-Babble/Baballonia/issues/221
- https://github.com/Project-Babble/Baballonia/compare/main...feat/autocalibration
- https://github.com/Project-Babble/BabbleTrainer (cloned to repos\BabbleTrainer)
- https://github.com/Project-Babble/BabbleDocs/pull/95/files
- https://github.com/Lenrul/beymod
- https://store.bigscreenvr.com/blogs/beyond/eye-tracking-for-bigscreen-beyond-with-eyetrackvr
