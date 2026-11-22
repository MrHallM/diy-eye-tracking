# OpenIris camera-config patches (2026-10-15)

Base: EyeTrackVR/OpenIris-ESPIDF `b64b0fa` (2026-09-30). Local branch `fix/apply-camera-config`.
Apply with `git am 000*.patch`. Built with ESP-IDF v5.4.2, board `seed_studio_xiao_esp32s3`.

- `0001`: `loadConfigData()` was never called, so `update_camera` only wrote NVS and the sensor
  always ran hardcoded gain 2. Now applied at boot and right after `update_camera`; gain clamped to 0-30.
- `0002`: new `exposure` field (manual AEC value, default 300 = old hardcoded value).

## Measured on the Babble mouth board (OV2640 160°, same scene, 240×240, Wi-Fi)

| firmware / setting | mean brightness /255 | fps |
|---|---|---|
| old firmware (0.2.1rc0, Dec 2025) | 22.3 | 67.9 |
| patched, defaults (exp 300, gain 2) | 19.5 | 66.0 |
| exp 600, gain 2 | 21.4 | 64.6 |
| exp 1200, gain 2 | 21.8 | 65.5 |
| exp 300, gain 8 | 37.6 | 64.3 |
| exp 300, gain 16 | 64.1 | 62.7 |
| exp 300, gain 30 | 79.5 | 62.5 |

- **Gain works** (about 3.6x brightness at 30), and fps drops only ~5%. Gain 30 is visibly grainier than 16.
- **Exposure above 300 does ~nothing.** Probably capped by the frame time at this fps (inference,
  not checked against the OV2640 timing registers).
- The board was left at defaults (exp 300, gain 2) so mouth tracking is unchanged.
- Full flash backup of the board BEFORE flashing: `firmware-backups/` (gitignored, contains Wi-Fi NVS).
  Restore: `python -m esptool --chip esp32s3 -p COM4 write_flash 0 <file>.bin`.
