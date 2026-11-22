# OpenIris "combo" firmware (2026-10-15)

Base: EyeTrackVR/OpenIris-ESPIDF `main` @ `b64b0fa`. Local branch `combo` in `repos/OpenIris-ESPIDF`.
Patches 0001-0004 are **other people's work** and are not included here: apply upstream
[PR #40](https://github.com/EyeTrackVR/OpenIris-ESPIDF/pull/40) (misonyah) and
[PR #39](https://github.com/EyeTrackVR/OpenIris-ESPIDF/pull/39) (webdevbrian) first, for example with
`git fetch origin pull/40/head:pr40 pull/39/head:pr39` and `git cherry-pick` (or `git merge`), then apply ours in order with
`git am 00*.patch` (0005 onwards). Board: `python tools\switchBoardType.py --board seed_studio_xiao_esp32s3`, ESP-IDF v5.4.2.

| # | Patch | Source |
|---|---|---|
| 0001-0003 | diagnostic frame + camera auto-retry + LED recovery | upstream PR #40 (misonyah), not included: get it from the PR |
| 0004 | Wi-Fi TX power actually applied, mesh picks the strongest node, XCLK by real mode, XIAO CPU 160 MHz | upstream PR #39 (webdevbrian), not included: get it from the PR |
| 0005 | saved camera settings (gain) actually applied | ours |
| 0006 | `exposure` setting | ours (see below: capped at 240×240) |
| 0007 | one failed capture no longer closes the stream | ours: bug is in upstream `main` too |
| 0008 | XIAO Wi-Fi XCLK pinned at 23 MHz | ours: PR #39 alone drops 66 → 44 fps |
| 0009 | several stream clients at once (async httpd + one capture task fanning out) | ours |
| 0010 | esp32-camera driver 2.0.15 → 2.1.8 (cam_hal robustness fixes) | ours |
| 0011 | REST API: bodies were never parsed (every REST command saw no fields), ~2 s per request (1 s sleep between polls), RequestContext leak, 4 KB task stack (typo `2024*2`) | ours: all in upstream `main` |
| 0013 | REST: wrong-method request CRASHED the board (bare string to a `%m` → InstructionFetchError); replies no longer used as format strings | ours: in upstream `main` |
| 0014 | zoom survives `update_camera` | ours |
| 0015 | **debug only**: `sensor_reg` / `set_xclk` (serial, `POST /api/debug/reg/` `/api/debug/xclk/`), not saved | ours |
| 0012 | sensor zoom: `setVieWindow` implemented + `set_zoom` (serial / `POST /api/set/zoom/`), not saved | ours |


> ⚠️ **Network privacy.** OpenIris's camera stream and REST API have **no password**: anyone on the same network
> who finds the board can watch the camera and send commands. With **0009** several viewers can connect at once, so
> someone could watch without cutting off your own connection. **0015** adds debug commands that write the camera
> chip's registers and clock: they can't damage it, but they can freeze or garble the image. Only use these boards on
> a home network you trust, never on public or shared Wi-Fi, and leave 0015 out unless you need it for debugging.

## Measurements (XIAO ESP32-S3 Sense, OV2640 160°, 240×240 JPEG, Wi-Fi, same scene)
| build / setting | fps | mean brightness /255 |
|---|---|---|
| old 0.2.1rc0 (Dec 2025) | 67.9 | 22.3 |
| + PR #39 as-is (16.5 MHz XCLK) | **44.5** | 21.3 |
| + XCLK 23 MHz (CPU still 160) | 64.4 | - |
| + multi-client, 1 client | 66.7 | - |
| + multi-client, 2 clients at once | 66.5 + 66.3 | - |
| + driver 2.1.8, 1 / 2 clients | 66.8 / 66.8 + 66.8 | 21.0 |
| driver 2.1.8, gain 16 | 66.7 | 61.6 |

## Findings
- **Exposure is capped by design at 240×240:** `set_framesize` puts the OV2640 in CIF mode (≤296 lines/frame,
  `ov2640.c`), and exposure is counted in lines, so `aec_value` above ~300 can't add light at this fps. Gain is the knob.
- **Sensor zoom is possible:** the driver's `set_res_raw()` → `set_window(mode, offset, total, output)` can read an eye-sized
  window in SVGA mode at 2× the detail and still output 240×240 (OpenIris's `setVieWindow` is an empty stub). The fps cost is untested.
- Upstream issue #30 ("feed randomly freezes", fps windows dipping to 17-33): the maintainer blames HTTP MJPEG and plans TCP/UDP.
  With 0009 the capture no longer waits on the socket, so a slow send skips frames instead of queuing stale ones (unmeasured
  against #30's setup).

## 10-minute soak (combo + driver 2.1.8, 1 client, home Wi-Fi)
39,982 frames in 600 s = 66.6 fps, 0 reconnects. Frame gap median 15.2 ms, p99 34.1 ms, max 107 ms;
stalls >100 ms: 1, >250 ms: 0, >1 s: 0.


## REST API before/after (0011)
Before: every request ~2000 ms; `PATCH /api/update/led_duty_cycle/ {"dutyCycle":100}` → 400; `PATCH /api/update/camera/` hung the REST task.
After: 18-300 ms, commands apply.

## Zoom (0012)
Full view 66.8 fps → zoom (SVGA window 300 px = 2×, or 240 px = 2.5×) **33.3-33.5 fps**: SVGA mode reads twice the lines.
Gain now survives mode switches (45.8 → 46.6 /255 around a zoom on/off). Recovering 60+ fps while zoomed would need
sensor-level row windowing (REG17-1A; error-prone per the OpenMV forum), untested.


## Getting the frame rate back (2026-10-15, steady bright target, gain 16)
| mode | XCLK | fps | picture |
|---|---|---|---|
| normal | 23 MHz | 67 | ok |
| normal | 30 MHz | 80 | ok |
| normal | **36 MHz** | **~100** (5-min soak 99.2 fps, 0 reconnects, 1 stall 159 ms) | ok, ~same brightness |
| zoom 2× | 23 MHz | 33.5 | ok |
| zoom 2× | 30 MHz | 40 | ok |
| zoom 2× | **36 MHz** | **~50** | ok; darker at exp 300, exp 600 → brighter than normal (zoom frames are ~600 lines, so exposure works there) |
| zoom 2× | 40 MHz requested | 49.9 | same as 36 (clock likely quantized) |
| zoom + CLKRC doubler (0x111=0x80) | 23 MHz | 67 | **CORRUPT** (vertical streaks), unusable |
| zoom + VEND cut to 0x25 | 23 MHz | - | **stream stalls** (DSP expects 600 rows); needed a reboot |

OV2640 XCLK spec tops out around 36 MHz (verify in the datasheet). Heat at 36 MHz NOT measured (PR #39 cut the clock for heat).

## Heat test (2026-10-15, ESP32 die temperature via get_temperature, streaming continuously, 12 min per phase)
| XCLK | settled chip temp |
|---|---|
| 23 MHz (~67 fps) | 63.1 °C (61 → 63 over the first 7 min, then flat) |
| 36 MHz (~100 fps) | 67.9 °C (+4.8 °C; flat 67-68 over the last 5 min) |

The CHIP, not the camera module (unmeasured). The sensor reads in 1 °C steps. (thermal.py's fps column was a byte-counting bug; ignore it.)

## Clock policy (decided 2026-10-15)
Full view stays at 23 MHz (~66 fps, coolest; extra fps there is past what VRChat can show). Zoom switches XCLK to 36 MHz automatically (~50 fps vs 33), back to 23 MHz on zoom-off. Measured 66 ? 50 ? 66 fps.

