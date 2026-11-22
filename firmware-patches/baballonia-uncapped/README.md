# Baballonia: uncapped processing + fixes (2026-10-15)

Base: Project-Babble/Baballonia `main` @ `5dbd332` (2026-08-31, v1.1.1). Local branch `perf/uncapped-processing`
in `repos/Baballonia`. Apply with `git am 000*.patch`. Personal use (Babble licence is non-commercial).

Build (needs the .NET 10 SDK and submodules `src/HyperText.Avalonia` + `src/VRCFaceTracking`):
```
cd src/Baballonia.Desktop
dotnet publish -r win-x64 -c Release --self-contained -f net10.0
```
Use **publish**, not `build -o`: the post-publish target moves the capture plugins into `Modules\`. Without them the
app starts but no camera works. Output was copied to `EyeTrackingLab\builds\baballonia-uncapped\`. It shares settings
and models with the installed app (`%APPDATA%\ProjectBabble`, `Documents\ProjectBabble`).

| # | Change |
|---|---|
| 0001 | process each frame as it arrives (Capture.AnyFrameArrived → one coalesced UI-thread update), instead of a 10 ms DispatcherTimer that really ticks every ~15.6 ms on Windows |
| 0002 | OSC sender wakes per result (was a Task.Delay(10) poll); stops losing messages enqueued between ToArray() and Clear() |
| 0003 | IP camera decodes exactly the frame bytes (TrimEnd scanned the whole 1 MB buffer per frame) |
| 0004 | per-frame Mat leaks (SingleCameraSource colour convert, FaceProcessingPipeline frame, DualCameraSource) + DualCameraSource Start/Stop were swapped |
| 0005 | fall back to auto-detecting the capture provider: renamed provider names ("IpCameraCapture" → "Wireless/IP Camera") stopped the camera starting with settings from older versions |

## Measured (Babble mouth board at XCLK 36 MHz = 99.7 fps, face pipeline, OSC on UDP 8888, `/jawOpen`, 15 s)
| | updates/s | median gap | p95 gap |
|---|---|---|---|
| installed Baballonia 1.0.0 (Mar 2026) | 55.7 | 15.8 ms | 31.9 ms |
| patched 1.1.1 | **98.3** | **9.9 ms** | 20.3 ms |

Working set steady (487 → 486 MB over 30 s). Caveat: not a pure A/B, since the installed version is older; but the 15.8 ms
gaps are the Windows timer tick, and the patched gaps follow the camera's 10 ms frames.

## Remaining caps NOT touched
- Baballonia's own VRCFT module (`src/VRCFaceTracking.Baballonia/BabbleVRC.cs`) has a `Thread.Sleep(10)` loop (~64 Hz on Windows).
- VRCFaceTracking's own update loop, and VRChat's network sync (~10 Hz for remote players), are outside Baballonia.
- Eye pipeline: the speed-up is the same code path, but untested (no eye cameras yet).

## 0006 control API (2026-10-15)
`GET http://127.0.0.1:8899/status` (face/left/right CameraSettings) and `POST /transform` `{"camera":"left","roi":{"x":40}}` (any of roi/rotationRadians/gamma/useRedChannel/useHorizontalFlip/useVerticalFlip). Loopback only; port via BABALLONIA_CONTROL_PORT (0 = off). Live only, not saved; dragging the crop in the UI overwrites it. Tested: status read, face roi + gamma change reflected, malformed JSON gives 400.


| 0007 | control API: `GET /face/frame` (latest raw face frame, image/jpeg, X-Frame-Age-Ms) + `GET /face/latest` (latest face outputs, before calibration) - for tools/facefollow (2026-11-19) |
