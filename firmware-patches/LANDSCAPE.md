# Camera firmware landscape (surveyed 2026-10-15)

Base everything on **EyeTrackVR/OpenIris-ESPIDF** (`main`, active). The old Arduino `EyeTrackVR/OpenIris`
is ARCHIVED. No independent alternative firmware turned up: everything active is a fork of OpenIris-ESPIDF.
Baballonia also ships Babble's own builds (`%LOCALAPPDATA%\Baballonia\Firmware\Binaries\BabbleFirmware1.0-1.3.bin`,
plus `1.2_TrackingLossFix`). Their source is presumably the `Project-Babble/OpenIris-ESPIDF` fork (unverified).

## Worth taking (open upstream PRs)
| PR | What | Relevance |
|---|---|---|
| #39 webdevbrian "Thermal + WiFi reliability" | `esp_wifi_set_max_tx_power()` was NEVER called (TX power setting dead, always 20 dBm); `CAMERA_WIFI_XCLK_FREQ` dead on UVC-enabled builds (XIAO runs the USB XCLK in Wi-Fi mode); mesh scan picks the first node, not the strongest; XIAO → 160 MHz. Measured 155°F → 110°F. Maintainer (lorow) intends to merge. | HIGH: same "saved but never applied" bug class as ours; heat matters next to the face |
| #40 misonyah "diagnostic frame when camera init fails" | auto-retry camera init, recover live streams, fix LED stuck in error | MEDIUM: robustness |
| #15 RamesTheGeneric (Babble dev) "VSync Illuminator Strobe" | pulses the external IR LED in sync with the camera VSYNC (modified `esp32s3/ll_cam.c`) | the "Meta varies the IR" idea, already built. Needs the LEDs on a GPIO (hardware change; V4.1 Lite is on VBUS) |
| #35 OrendCross "Quest Pro IR LED flicker" | XCLK 10.5 MHz removes the Quest Pro's IR flicker from Babble cams, but caps ~30 fps | Quest Pro only |
| #22 | switch_mode_and_restart command | low |

`misonyah/OpenIris-ESPIDF` branch `integration/all-open-prs` = upstream + #15 #22 #35 #39 #40 merged (ready-made combo to diff against).

## Interesting forks / experiments
- **RamesTheGeneric** (Babble dev): `802.11-TX(-RS)` raw Wi-Fi frames + Reed-Solomon/zfec FEC, `udp-steam-poc` → lower-latency
  wireless streaming experiments (not merged).
- **PhosphorosVR** (FaceFocusVR, commercial): per-sensor camera profiles, OV3660 tuning, 320p tests, fan PWM, upstream sync docs.
- **bwmp**: "better contrast" camera settings (brightness/contrast/sat 0, aec 600, gainceiling 2, raw gamma OFF, BPC/WPC off, q6, 20 MHz XCLK).
- **webdevbrian/xiao-ov3660**: OV3660 on XIAO fixes.
- upstream branches: `feature/add-jpeg-ls` (lossless JPEG-LS, initial code), `feature/facefocus-duty-cycle-poc`.
- **celesrabbit-lab/TrackerBridge**: Android app that makes a Meta Quest a WIRED USB bridge for OpenIris trackers (Quest 3 relevance: no Wi-Fi hop).

## Broader search (2026-10-15, round 2)
- Upstream issues: #30 "feed randomly freezes" (maintainer: HTTP MJPEG, plans TCP/UDP streaming); #34 setup port check; #14 Linux.
- **esp32-camera driver:** OpenIris locked 2.0.15; latest 2.1.8 has cam_hal fixes (recursive cam_take → bounded loop,
  SOI/EOI bounds, DMA overflow guard, PSRAM-DMA). Taken in `openiris-combo/0010`.
- **arXiv 2604.24331** (Zimmer, Abdrabou, Kasneci, Apr 2026) "An Affordable, Wearable Stereo-Eye-Tracking Platform": per
  search snippets, an ESP32-S3 + OV2640 per camera, EyeTrackVR firmware "with minor changes", ~45 fps at 240×240 (= the
  16.5 MHz XCLK rate?), per-frame timestamps; hardware + docs open (repo URL not on the abstract page). Worth reading.
- OV2640 sensor windowing is error-prone by hand (OpenMV forum: timeouts/green images); use the driver's `set_res_raw`.
- No non-OpenIris eye/face camera firmware exists in active use (old Futurabeast face-cam firmware in IndexEyeTrackVR is stale).

## Our own patches
`openiris-combo/`: the full current series (PRs #39 #40 + 6 of ours), with measurements.
`openiris-camera-config/`: apply saved camera config (gain works, measured) + exposure field.
