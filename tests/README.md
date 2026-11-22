# Hardware tests

Measurements on real hardware (XIAO ESP32-S3 Sense + OV2640 + OpenIris + Baballonia; the 10-10 tests ran on a
Bigscreen Beyond 1 setup). Each folder is one test session with its `RESULT.md`. Raw logs and recordings are not
included.

| Folder | What was tested |
|---|---|
| [`F1_2026-10-10`](F1_2026-10-10/RESULT.md) | Wi-Fi baseline, two stream clients at once, IR interference from base stations |
| [`F2_2026-10-10`](F2_2026-10-10/RESULT.md) | Latency: Wi-Fi vs wired (UVC) |
| [`F3_2026-10-10`](F3_2026-10-10/RESULT.md) | 10-minute Wi-Fi soak |
| [`QPT_2026-10-17`](QPT_2026-10-17/README.md) | Babble's face model vs Quest Pro face tracking |

Rows in the RESULT files look like `| C0153 | confirmed | same | + | what we saw |`: a claim ID (the claim list itself is not
public), whether the test confirmed it, whether our hardware matched, and whether the claim was helpful. The claim
texts are not included, but the "what we saw" column explains each one.
