# DIY eye tracking: notes, tools and firmware patches

A reference collection from building EyeTrackVR-style eye tracking for VR: firmware fixes, tools, measurements and
notes. It isn't a step-by-step guide. Take whatever is useful to you.

**Target headset: Meta Quest 3.** Some earlier tests were done on a Bigscreen Beyond 1.

**Hardware:** 2× Seeed XIAO ESP32-S3 Sense, 2× OV2640 cameras with 130° lenses (IR filter removed), IR LEDs,
Wi-Fi streaming. **Software:** [OpenIris](https://github.com/EyeTrackVR/OpenIris-ESPIDF) firmware on the boards and
[Baballonia](https://github.com/Project-Babble/Baballonia) (Project Babble) on the PC, feeding VRChat through VRCFaceTracking.

> **Status:** both camera boards stream at about 66 fps over Wi-Fi with no freezes (with the patches below), and
> the mount has been fit-tested. The full in-headset build is **not finished yet**. Expect rough edges.
>
> **Shared as-is.** No support is promised, and things may change or stay unfinished.

## What's here

| Folder | What |
|---|---|
| [`firmware-patches/`](firmware-patches) | Fixes and features for **OpenIris** (camera settings that were never applied, exposure, several stream clients, a REST API that never parsed requests and could crash the board, sensor zoom, a newer camera driver) and **Baballonia** (frames processed as they arrive instead of on a 10 ms timer, a memory leak fixed, faster OSC). Each folder has a README with measurements. [`LANDSCAPE.md`](firmware-patches/LANDSCAPE.md) maps related upstream work. |
| [`hardware/`](hardware) | Notes on the Quest 3 lens-ring mount we use, plus `eye-rig-v0/`: FreeCAD scripts for a desk test rig (chin rest + lens bosses that hold the rings) and the print-fit test coupons. |
| [`tools/`](tools) | Python tools: frame / fps / freeze logging, latency, OSC and pose logging, a calibration-grid recorder and scorer for camera placement, head calibration, face-tracking helpers. |
| [`training/`](training) | Experiments pre-training a small gaze / eyelid model on public near-eye IR datasets, then fine-tuning per user. Code only: no datasets or trained models are included. |
| [`research/sim/`](research/sim) | Simulations of a calibration-free drift correction (where the eyes usually sit in the head). |
| [`notes/`](notes) | How Baballonia works inside (per-user model, calibration, where camera transforms are applied). |
| [`tests/`](tests) | Write-ups of hardware tests: Wi-Fi vs wired (UVC) latency, a 10-minute Wi-Fi soak, two clients at once. |

## If you use any of this

- **Read [SAFETY.md](SAFETY.md).** The IR LEDs shine at your eyes, so keep them within safe limits, and the
  cameras stream over Wi-Fi with no password, so only use them on a network you trust.
- Paths in the scripts are examples (for instance `captures/`). Change them to fit your setup.
- This is a personal project, **not affiliated** with EyeTrackVR, Project Babble or Meta. Use at your own risk.

## Credits and licences

Built on the work of the EyeTrackVR and Project Babble communities and others. See **[CREDITS.md](CREDITS.md)**.

- Code and docs here: **MIT** ([LICENSE](LICENSE)).
- `hardware/eye-rig-v0/`: **CC BY-SA 4.0**, because it fits (and uses an outline measured from) BasilAldwin's
  CC BY-SA Quest 3 ring. See [`hardware/eye-rig-v0/LICENSE.md`](hardware/eye-rig-v0/LICENSE.md).
- `firmware-patches/`: changes to other projects. Use them under each project's own terms (see CREDITS.md).
  Two patches by other people are not copied here. The README links their upstream pull requests.
