# Credits

This project stands on other people's work. Thank you to all of them.

## Projects this builds on or patches

| Project | What it is | Licence | Used for |
|---|---|---|---|
| [OpenIris-ESPIDF](https://github.com/EyeTrackVR/OpenIris-ESPIDF) (EyeTrackVR) | Camera firmware for the XIAO ESP32-S3 | no licence file (all rights reserved by default) | Firmware on the boards. Our changes are in `firmware-patches/openiris-*`, offered as patches for people who already use OpenIris. |
| [Baballonia](https://github.com/Project-Babble/Baballonia) (Project Babble) | Eye and face tracking app | Babble Software Distribution License 1.0 (non-commercial) | The tracking app. Our changes are in `firmware-patches/baballonia-uncapped/`. |
| [BabbleTrainer](https://github.com/Project-Babble/BabbleTrainer), [ProjectBabble](https://github.com/Project-Babble/ProjectBabble) | Babble's model trainer and legacy app | Babble Software Distribution License 1.0 | Studied for `notes/` and `training/`. |
| [EyeTrackVR-Docs](https://github.com/EyeTrackVR/EyeTrackVR-Docs) | EyeTrackVR documentation | MIT | Build guidance and the IR safety guide linked in SAFETY.md. |
| [EyeTrackVR-Hardware](https://github.com/EyeTrackVR/EyeTrackVR-Hardware) | LED boards and mounts | CC BY-NC-SA 4.0 | Reference only. None of its files are included. |
| [ETVRTrackingModule](https://github.com/EyeTrackVR/ETVRTrackingModule) | VRCFaceTracking module | MIT | Reference for how eye data reaches VRChat. |
| [pye3d-detector](https://github.com/pupil-labs/pye3d-detector) (Pupil Labs) | 3D eye model | LGPL-3.0 | Idea source for drift / slip correction (`research/sim/`). |
| [beymod](https://github.com/Lenrul/beymod) (Lenrul) | Bigscreen Beyond mounts | Apache-2.0 | Reference for the earlier Beyond tests. |

## Hardware

- **Quest 3 lens ring**: "Quest 3 eyetrackvr Mount for v4 hardware" by **BasilAldwin** (with a ring by Dentohn),
  [Thingiverse 6439510](https://www.thingiverse.com/thing:6439510), CC BY-SA. The test rig in `hardware/eye-rig-v0/`
  fits its v13 ring. The ring's files are not included: get them from Thingiverse.

## Patches by other people (not copied here)

The OpenIris patch set builds on two upstream pull requests. Get them from their sources:

- [OpenIris-ESPIDF PR #40](https://github.com/EyeTrackVR/OpenIris-ESPIDF/pull/40) by **misonyah**: diagnostic frame,
  camera auto-retry, LED recovery.
- [OpenIris-ESPIDF PR #39](https://github.com/EyeTrackVR/OpenIris-ESPIDF/pull/39) by **webdevbrian**: Wi-Fi TX power,
  mesh access-point choice, camera clock by mode.

## Datasets

`training/` refers to public near-eye IR datasets. See [`training/DATASETS.md`](training/DATASETS.md) for each one's
licence. No dataset files or models trained on them are included.
