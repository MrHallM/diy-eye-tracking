# F2 wired (UVC) mode, 2026-10-10

Setup: same board as F1 (XIAO ESP32-S3 Sense + OV2640 160°, OpenIris-ESPIDF, who_am_i version 0.0.1), switched to UVC
with the serial command switch_mode uvc + restart_device (no reflash). In UVC mode the board enumerates as
USB\VID_303A&PID_8000 (not 1001 as in Wi-Fi mode): composite device with a serial port (COM28) plus a camera named
ETVR-Right (MI_02). Driver advertises 240x240 at 60 fps (OpenCV MSMF, uvc:1). Baballonia 1.1.1.0. Windows 11 26200.
Board and a Logitech C615 webcam on the same Intel USB 3.20 xHCI controller (PCI DEV_7A60).

Measured:
- Wired baseline (uvc_baseline.csv, 300 s): 59.8 fps mean, frame interval median 16.2 ms, p99 32.9 ms, max 48 ms,
  0 freezes. Wi-Fi on the same board (F1, 60 s): 66.4 fps, median 15.1 ms, p99 29.7 ms. Wired is capped at the
  advertised 60 fps, about 10 percent below Wi-Fi.
- Glass-to-glass lag (lat_uvc.csv, lat_wifi.csv): wired (UVC, Media Foundation) median 52.6 ms, Wi-Fi median
  41.7 ms. Decision 10-10: the build will use Wi-Fi only, so the DirectShow lag was not chased further.
- Side effect: while Baballonia streamed ETVR-Right, a brief OpenCV open of the Logitech C615 (camera index 0) left
  the C615's camera and mic in Code 10 (CM_PROB_FAILED_START). Same xHCI controller; consistent with the
  controller running out of resources (see the USB endpoint notes). Fixed by unplugging the C615. Lesson for the
  Beyond build: two eye cameras + mouth camera over UVC on one controller may hit this (test C0059 in F8).

| Claim | Outcome | Match | Helpful | What we saw |
|---|---|---|---|---|
| C0174 | confirmed | diff | + | Baballonia 1.1.1.0 on Windows showed the OpenIris board's live feed in wired (UVC) mode (watched it in real time). The claim names Babble v1.3, hence diff |
| C0140 | partly | same | 0 | The XIAO board works wired on the current release (Baballonia 1.1.1.0, OpenIris-ESPIDF); we did not test the old stable release the claim says failed |
| C0037 | wrong | diff | - | Wired is supported now: the firmware runs UVC mode and the desktop app shows its feed (C0174). The claim described an older state, hence diff |
| C0030 | confirmed | same | + | While Baballonia held ETVR-Right, a second app (OpenCV MSMF) could OPEN the camera but received 0 frames in 5 s and 3 s attempts. Note it does not fail loudly: it opens and stays silent |
| C0006 | wrong | diff | - | Baballonia 1.1.1.0 has no fps or resolution setting anywhere in its UI (searched by hand), so the advice to set them cannot be followed; and our camera offers one mode, 240x240 at 60 fps, measured 59.8 fps, so the default is already the max. The docs are written for the official Babble tracker, hence diff |
| C0057 | wrong | diff | - | On our board wired was neither faster nor lower-lag. Frame rate: UVC 59.8 fps vs Wi-Fi 66.4 fps (both 0 freezes). Glass-to-glass lag (tools/latency.py, full-screen black/white flips, same camera position, includes monitor delay): UVC via Media Foundation median 52.6 ms (p10 42.8, p90 66.5, 59/60 flips) vs Wi-Fi median 41.7 ms (p10 26.8, p90 53.9, 60/60). Baballonia reads UVC through DirectShow, which runs 60.2 fps here, but its lag could not be measured: DirectShow plus a full-screen OpenCV window in one process deadlocked in the driver (twice). The claim is about the old wired SERIAL mode, ours is UVC, hence diff |
| C0008 | partly | same | + | Replugged the board while Baballonia 1.1.1.0 streamed it. Baballonia tried ETVR-Right once (log 23:03:48: No data was received from ETVR-Right, with OpenCvCapture, closing), probably while the board was still booting, and never retried. Re-selecting was impossible: ETVR-Right no longer appeared in the camera list, because the list is built once at startup (the dropdown refresh handlers in HomePageView.axaml.cs L217-233 are commented out at 5dbd332). Restarting Baballonia fixed it with no other action: ETVR-Right auto-selected and streaming. So: fails to re-recognize = yes; restart fixes = yes; re-select fixes = no |
