# facefollow: dynamic calibration for the Babble mouth tracker

Goal (2026-11-19): tracking that stays right when the camera / headset shifts, without recalibrating.

## Why it's needed (measured 2026-11-19)
- Between frame sets a few minutes apart, with nothing moved on purpose, the relaxed face sat **~5 px higher**
  (22:29-22:30 vs 22:31). Inside one 3-minute Calibrator run the resting "lower lip down" crept 0.05 -> 0.35.
- Shifting Babble's crop by 5 px changes the most-affected shape by ~0.1 (up to 0.2); 8 px by up to 0.38
  (tongue, pucker, jaw forward, jaw open). That crosses the calibration minimums (0.3-0.5) -> shapes fire at rest.

## Layer 1: follow the face (this folder)
- `facereg.py` - registers a RELAXED frame against the relaxed reference (CLAHE + high-pass, OpenCV ECC affine,
  coarse-to-fine) -> dx, dy, scale, rotation, match quality. Selftest on known warps: worst error 0.29 px.
  Same-position relaxed frames match at cc 0.94-0.97. Expressions fool it (by design it only runs when relaxed).
  Big deliberate re-mounts (scale 0.6-0.8) fail -> recalibrate + new reference for those.
- `nosetrack.py` - first attempt (nose template). Dropped: the nose sits at the frame's top edge (can't see it move
  up) and is the darkest area. Kept for reference.
- `follower.py` - the service. `set-reference` (face relaxed, right after calibrating), then `run` (dry run, logs
  only) or `run --apply`. Relaxed = every watched shape within calibration rest + 0.15 for 1 s. Median of 5 good
  measurements, deadband 2 px, steps <= 3 px, max 25 px from the reference crop, scale 0.88-1.12. CSV log per run.
- `test_follower.py` - offline replay with the 11-19 frames: crop walks 20 -> 17 -> 15 (the measured 5 px), then holds;
  expression frames are never measured. PASS.

Needs the patched Baballonia (`builds/baballonia-uncapped`, patch 0007 adds `GET /face/frame` + `GET /face/latest`
to the control API) so the follower never opens the camera itself (a 2nd stream choked the ESP32 on 11-19).

## Layer 2: re-learn the resting face (not built yet)
Slowly track each shape's resting level during relaxed moments and adjust the calibration minimums (like the old
BabbleApp's adaptive min/max). Planned as an OSC relay between Baballonia and VRCFaceTracking.

## Next
1. First live session: patched Baballonia + `follower.py set-reference` + `run` (DRY RUN) -> read the log: how often
   it measures, how much the face really moves, would its moves have been right?
2. Then `--apply`, compare tracking at rest with vs without.
