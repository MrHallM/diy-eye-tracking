# Tools

Python tools used for the tests and the build. Run each with `--help` for its options. Paths like `captures/` are
examples: change them for your setup. Several tools have a `--selftest` that checks their maths without hardware.

- `framelog.py`: frame / fps / freeze logging for a camera stream.
- `latency.py`: camera-to-PC latency measurement.
- `setup_eye_board.bat`: runs the OpenIris setup menu (name, Wi-Fi, mode) on a board over USB.
- `session_stats.py`: statistics over a logged session.

## Calibration and logging tools

All on one clock (`time.time()`), so their CSVs line up with no sync step:
  - `tools/a1grid.py`: the A1 target grid (7x7, ±25×±20°, 2.3→0.57° dot, score 1.0-1.5 s; `--grid 3x3` for a
    9-point validation). `--display vr` = head-locked SteamVR overlay at 1 m (the real test); `--display screen` =
    monitor, for dry runs only (the grid shrinks to fit). `--record/--serve/--osc` run the two tools below in the
    same process, into one run folder, and tag every frame with its target + phase.
  - `tools/eyerec.py`: raw footage (the board's own JPEGs, no re-encode) → `frames.mjpeg` + `frames.csv`.
    `--serve 8090` re-serves the stream so Baballonia (camera address `http://127.0.0.1:8090/`) sees the very
    frames we record. The board serves one client, so this is the only way to record while Baballonia tracks.
  - `tools/osclog.py`: Baballonia's OSC output (eye X/Y/lid, mouth shapes) → `osc.csv`. Port 8888 is shared with
    the VRCFT Babble module: close VRCFT, or move Baballonia's send port and use `--forward`.
  - `tools/poselog.py` (10-11): SteamVR headset pose → `pose.csv` (250 Hz: yaw/pitch/roll, head-frame angular
    speed). For VOR / head-speed gating. The camera is ~42 ms behind: shift the HEAD data later by the measured lag
    in analysis; nothing live is slowed. `--selftest` checks the angle maths. 🔴 Not run with a headset yet.
  - `tools/headcal.py` (10-11): one-dot head-movement calibration (CalibMe17-style). A dot FIXED in the room at
    2 m; keep looking at it and move the head slowly for ~25 s → `headcal.csv` = the dot's direction in head space
    (the eye-in-head truth) per pose sample, plus coverage of ±25×±20° (aim > 85%). Learns SCALE + vertical offset.
    Same `--record/--serve/--osc` options as a1grid. `--selftest` checks the geometry. 🔴 Not run with a headset yet.
  - `tools/calarchive.py` (10-11): keeps EVERY Baballonia eye calibration (Baballonia overwrites `user_cal.bin`
    each time) in `data/calib_archive/` (git-ignored: eye images), and `train --last N` runs Baballonia's own
    BabbleTrainer on several sessions joined (natural re-seats → slip robustness). Run `watch` before calibrating.
    `--install` points Baballonia at the result (Baballonia closed). `--selftest` = format/join check. 🔴 No real data yet.
  - `tools/qplog.py` (10-11): Quest Pro eye/face tracking straight from Virtual Desktop's shared memory
    (`VirtualDesktop.BodyState`, the same data VRCFT's VD module reads) → csv: per-eye gaze yaw/pitch + quaternion,
    validity, confidence, lid/brow expressions (`--all` = all 70). Passive: VRCFT/VRChat untouched. Pair with
    poselog.py for head pose. 🔴 Gaze signs/frame unchecked live: look far left then up on the first run.
