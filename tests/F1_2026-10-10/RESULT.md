# F1 board bench (Wi-Fi), 2026-10-10 (part 1: what could be run from the PC alone)

Setup: Babble DIY mouth-tracker board = Seeed XIAO ESP32-S3 Sense + OV2640 160° 850 nm · OpenIris-ESPIDF (get_who_am_i:
board xiao_esp32s3, version 0.0.1) · mode WiFi, hostname ETVR-Right, 240x240 (framesize 5), JPEG quality 7,
brightness 2, tx power 52, 1 stored network (home router, 2.4 GHz) · Windows 11 26200, PC wired to the
router · Baballonia closed · board on a desk ~1-2 m from the router, IP 192.168.8.192.

Measured:
- Wi-Fi baseline (wifi_baseline.csv, 60 s): 66.4 fps mean, frame interval median 15.1 ms, p99 29.7 ms, max 66 ms,
  0 freezes, 0 bad JPEGs, board-to-PC delay spread (jitter) 5.8 ms, frame size median 3.3 KB.
- Ping to the board: 89-104 ms (high for a LAN; the firmware has Wi-Fi power save on), yet the stream itself is
  steady at 66 fps.
- Opening COM4 does NOT reset the board (0 bytes waiting right after open, then normal answers). A boot log seen
  in the first read came from the board having just been plugged in.

| Claim | Outcome | Match | Helpful | What we saw |
|---|---|---|---|---|
| C0153 | confirmed | same | 0 | Device Manager: USB Composite Device USB\VID_303A&PID_1001, serial port COM4 (MI_00) plus a USB JTAG/serial debug unit (MI_02); no camera device in Wi-Fi mode |
| C0038 | confirmed | same | + | Client A (framelog) streamed at 66.4 fps for 30 s with no gap; client B connected at t=8 s and got 0 frames, timing out after 10 s. The first client keeps the stream, the newcomer gets nothing; the LRU purge in the code did not kick client A |
| C0012 | confirmed | same | 0 | The board's own Wi-Fi scan listed 11 networks, all on channels 1, 6 and 11 (2.4 GHz); the router's 5/6 GHz networks (e.g. its -6G SSID) never appeared. A join attempt to a 5 GHz SSID was not made |
| C0025 | partly | same | + | Camera aimed straight at a running base station from about 1-2 m, 20 s, 1333 frames at 66.6 fps (raw/basestation, basestation_samples.png). Scene very dark (mean 16.8/255): the claim's worst case of weak IR. No frame-wide flicker or banding: mean brightness std 0.78, frame-to-frame change median 0.11, row ripple flat. The base station shows only as one small dot (about 90 px, 0.16 percent of the frame) whose peak ripples 5-15 percent (median change 11/255, never a drop over 30 percent) at a clean 16.7 Hz. That is likely a 50 Hz rotor aliased by 66.6 fps (66.6 - 50 = 16.6), or an artifact every 4 frames; separating them needs a different fps. In the headset the eye camera faces the eye, not the base station, so the effect should be smaller still |
| C0035 | wrong | diff | - | On our firmware the camera settings are stored but never applied. update_camera saved quality 40 and brightness 0 (get_config confirmed), then a restart; then back to quality 7 and brightness 2, then a restart; same scene, 30 s each: 66.7 vs 66.5 fps, JPEG median 2057 vs 2053 bytes, image mean 9.9 vs 10.3/255, identical within noise (a 7 to 40 quality change should shrink frames several times). Why: CameraManager::loadConfigData(), the only code that applies quality and brightness (agc_gain), is never called anywhere in OpenIris-ESPIDF b64b0fa; the sensor always starts with hard-coded JPEG quality 8, brightness 2, 240x240. The mapping the docs describe exists in that dead function, but the docs are for the old Arduino firmware, hence diff. Settings restored to quality 7 / brightness 2 afterwards |
