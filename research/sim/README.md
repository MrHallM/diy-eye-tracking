# Simulations on public data

## Centre prior (target-free offset) — 2026-10-11
Question: can a slow, robust estimate of where the eyes usually sit in the head catch tracker drift
or slip without any targets? Simulated on REAL head-free gaze: Burlingham24 (Meta, CC-BY 4.0), 169 recordings,
~30 people, 9 everyday tasks, 26.6 h, Tobii glasses 50 Hz + head IMU. Tobii gaze = truth; our tracker = truth +
offset + noise (white 3 + AR(1) 4, tau 1 s = 5 deg RMS per axis, a GUESS at Baballonia's error shape).

Data: `data/burlingham24/data.mat` (git-ignored, 1.34 GB, sha256 d0145383...baafa) from
github.com/facebookresearch/Reality-Labs-Egocentric-ET. Scripts: `b24_load.py`, `center_sim.py`, `center_followup.py`.
Raw outputs: `out_noise0.txt`, `out_noise5.txt`, `out_followup.txt`.

### Results
1. **Tracker noise barely matters; normal viewing does.** With 0 vs 5 deg tracker noise the windowed estimate
   wanders almost the same amount (5-min window, 2-D: median 1.09 vs 1.33, 95th 4.07 vs 4.50 deg). The limit is how
   much people's own gaze centre moves, not our ~5 deg error.
2. **Horizontal works, vertical doesn't.** Same person, different tasks: horizontal centre moves median 1.5 deg
   (95th 4.9); vertical moves median 4.5 deg (95th 12.4). Per-task vertical means span -4.1 (pokemon) to +6.4
   (driving). Head orientation explains little of it (best row r = -0.28; vertical spread 6.2 -> 6.0 deg).
3. **Per-axis floor (5 deg noise, gate = fixations, head < 20 deg/s, >= 0.5 s), median / 95th:**
   horizontal 0.89/3.28 (2 min), 0.66/2.39 (5 min), 0.45/1.58 (10 min); vertical 1.23/4.65, 0.94/3.39, 0.57/3.23.
4. **Horizontal step detection** (threshold = that window's 95th floor; 77 recordings >= 10 min):
   - 2-min window: 5 deg found in 85 s median (83%), 3 deg 105 s (66%), 2 deg 112 s (44%).
   - 5-min window: 5 deg in 157 s (94%), 3 deg 220 s (82%), 2 deg 275 s (71%).
5. **Gating helps only a little.** Long steady fixations (>= 0.5 s, head still) gave the lowest 95th floor at
   30-300 s (e.g. 300 s: 4.07 vs 5.12 deg for all fixations), but no gate changes the picture.

### What it means for the design
- Use the centre prior for **horizontal offset only**, as a slow (2-10 min) anchor that catches big slips/drift
  (>= 3-5 deg) within a few minutes. Don't use it vertically, or only with a very wide tolerance.
- Small or fast slips need a faster target-free signal: **eye-region image registration** (crop re-centering).
- The vertical offset and the scale need explicit help: the one-dot head-movement calibration (CalibMe-style).
- Caveats: real-world tasks, not VRChat (VR fixations are tighter, Aizenman23 ±5 deg, so VR may do better);
  Tobii's own error is part of the "truth"; the noise model is a guess. Re-run on our own footage when we have it.

## Estimation layer (blink hold, jump gate, smoothing) — 2026-11-10
Question: how much of a ~5 deg tracker's error can design item 1 (the estimation layer) remove, and does item 2
("raw minus estimated" as a slip monitor) work? Script `estim_sim.py`, raw output `out_estim.txt`. Truth = the same
Burlingham24 gaze (70 recordings × 4 min; half the people tune the filter, the other half test it, 2.04 h).
Simulated tracker = truth + noise (5 deg per axis, three SHAPES) + glitch frames (0.5 % of frames, 10-40 deg) + a
blink artefact (gaze dragged ~20 deg down while the lid moves). The public file has no blink marks, so blink TIMES
are synthetic (~13/min, 120-280 ms); the lid signal is one frame late and misses 10 % of blinks.
Errors are 2-D, in degrees (5 deg per axis = 7.3 deg 2-D).

| tracker noise shape (white + slow) | output | error, all frames | in fixations | around blinks | false darts / min | delay |
|---|---|---|---|---|---|---|
| white-heavy 5 + 0 | raw | 8.74 | 7.32 | 20.9 | 233 | 0 |
| | Baballonia's one-euro defaults | 6.46 | 4.12 | 19.6 | 302 | 0 |
| | one-euro, tuned | 5.83 | 3.48 | 16.9 | 135 | 0 |
| | full layer | 5.12 | 3.66 | 9.6 | 94 | 20 ms |
| mixed 3 + 4 | raw | 8.74 | 7.31 | 20.9 | 345 | 0 |
| | Baballonia's one-euro defaults | 7.99 | 6.30 | 19.9 | 113 | 0 |
| | one-euro, tuned | 7.68 | 6.20 | 16.5 | 38 | 0 |
| | full layer | 7.03 | 6.08 | 10.9 | 6 | 20 ms |
| slow-heavy 1 + 4.9 | raw | 8.75 | 7.33 | 20.9 | 293 | 0 |
| | one-euro, tuned | 8.40 | 7.05 | 18.2 | 41 | 0 |
| | full layer | 7.77 | 7.03 | 11.5 | 9 | 20 ms |

("False dart" = the output moves faster than 150 deg/s while the real eye is still: what an avatar's eye twitch is.)

### Results
1. **Blink hold is the sure win.** Error around blinks halves (21 → 10-11 deg) in every noise shape, and it carries
   most of the layer's overall gain. What is left is the 10 % of blinks the lid signal missed and its one-frame delay,
   so the quality of the LID signal sets the ceiling here.
2. **Smoothing only pays if the noise is fast.** In fixations: 7.3 → 3.5 deg if the noise is all white, 7.3 → 6.1
   mixed, 7.3 → 7.0 if it is mostly slow wander. No causal filter touches the slow part (it looks like real eye
   movement); only calibration does. **We have never measured which shape Baballonia's error has, so this is now the
   most useful number to get**: stare at one dot for 30 s, split the error into fast and slow parts.
3. **The jump gate trades error for calm.** Waiting one frame before believing a >16 deg jump removes most false
   darts once smoothing has done its part (mixed: 38 → 6 per minute) but adds 20 ms and slightly raises overall error
   when the noise is white. Worth it for avatar eyes (twitches are what people see); the 0.5 % glitch rate is a guess.
4. **Design item 2 does not work as written.** After a 5 deg slip was added to the raw stream, "raw minus estimated"
   averaged +0.12 deg in the first 5 s and −0.04 deg afterwards (spread between recordings ±0.17): nothing. A filter
   follows its input, so the slip moves both streams together. Divergence shows spikes (glitches, blinks) only.
   Slip needs an estimate that is ANCHORED to something else: the horizontal centre prior (above) or eye-region image
   registration. Keep the two-stream LOGGING (hardware tests must not measure our own guesses); drop it as a slip monitor.
5. Baballonia's own one-euro defaults (0.5, 3) made MORE false darts than raw on white noise (302 vs 233/min). The
   conversion of its −1..1 signal to degrees (÷45) is my assumption, so treat that row as approximate.

### Limits
Noise shapes, glitch rate and blink artefact are all guesses, not measurements. 50 Hz data (ours ~66). One eye, so
nothing here tests the two-eye ideas (Hering's law, one-eye-lost). No head-rotation compensation (the file has head
speed but not direction).
