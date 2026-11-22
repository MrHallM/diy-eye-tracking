# QPT 2026-10-17 — Quest Pro as teacher for the Babble mouth camera, step 1

Question: can the mouth camera and the Quest Pro's face tracking (Virtual Desktop shared memory) be recorded together
and lined up, and how far is Babble's stock face model from the Quest Pro on the tester's face?

Recorded with `python tools/qpteach.py record --out tests/QPT_2026-10-17/raw/p0` (spoken ~90 s routine), checked with
`python tools/qpteach.py check tests/QPT_2026-10-17/raw/p0`. Raw data is git-ignored (`raw/`).

Setup: mouth tracker on the 29 mm arm, LEDs diffused with tape, full-frame crop, patched firmware (gain 6);
Quest Pro Natural Facial Expressions ON, VD "Forward tracking data to PC" on, VRChat running.

## Result

- 95 s: 6224 camera frames (65 fps, 0 freezes, 0 bad JPEGs), 7666 Quest Pro samples (81 Hz).
- Time offset: camera −10 ms vs Quest Pro (jaw r = 0.75) → effectively in sync. Left/right NOT mirrored.

| Babble output | Quest Pro | r |
|---|---|---|
| JawOpen | JawDrop | 0.75 |
| JawLeft / JawRight | JawSideways L / R | 0.53 / 0.67 |
| MouthLeft / MouthRight | MouthLeft / Right | 0.54 / 0.44 |
| MouthSmile L / R | LipCornerPuller L / R | 0.16 / 0.13 |
| MouthPucker | LipPucker L+R | 0.17 |
| MouthFunnel | LipFunneler ×4 | −0.09 |
| JawForward | JawThrust | −0.10 |
| MouthFrown L / R | LipCornerDepressor L / R | −0.05 / −0.03 |
| MouthStretch L / R | LipStretcher L / R | 0.13 / 0.07 |
| CheekPuff L / R | CheekPuff L / R | 0.14 / −0.02 |
| MouthClose | LipsToward | 0.28 |
| TongueOut | TongueOut | 0.04 |

The stock model follows the jaw and side-to-side movement, and misses smiles, pucker/funnel, jaw forward, frown,
stretch and cheek puff: exactly the tester's complaints, so there's room to teach. Caveat: the Quest Pro isn't ground
truth (its tongue especially), and the stock model was running on IR-lit images here.

## Blocker: the Quest Pro's IR

With Quest Pro face tracking on, mouth brightness alternates frame to frame between two clusters (~190 and ~230;
jump avg 36). With it off, it's ~140 and steady (jump 0.2). No frame comes near the clean level, so picking one
flicker phase doesn't give clean images. A model trained on these would learn IR-lit images that the Quest 3 will
never produce.

Fix ordered: IRC650 absorbing IR-cut filter, ø16.5 × 1.1 mm (arrives November), taped flat over the lens. It passes
the white LEDs and blocks the Quest's 850 nm IR. Next: re-run the flicker check (centre and edges separately), then
record the real dataset (20–30 min, 2+ re-wears) while the Quest Pro is still here.
