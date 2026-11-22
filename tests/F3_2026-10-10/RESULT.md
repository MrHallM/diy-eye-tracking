# F3 soak, Wi-Fi, 2026-10-10/11

Setup: same board as F1/F2 (XIAO ESP32-S3 Sense + OV2640 160°, OpenIris-ESPIDF, who_am_i version 0.0.1, hostname
ETVR-Right, 192.168.8.192, 240x240), Wi-Fi mode, on the main router (not the second one). Powered from a PC USB port for
power only. Only client: `tools/framelog.py http://192.168.8.192/ --seconds 7200 --freeze 0.5`, unattended,
23:40:31 → 01:40:31. No Baballonia, no browser. The wired 2 h soak in the plan was skipped on purpose: the build is
Wi-Fi only (decision 10-10), so a wired freeze would not affect it.

Measured (wifi_soak.csv / .summary.txt; per-window numbers from the CSV):
- 479,873 frames in 7200 s = **66.6 fps**, and 66.6-66.7 fps in every 10-minute window.
- **0 freezes ≥ 0.5 s, 0 bad JPEGs.** Longest gap 135 ms. 5 gaps ≥ 100 ms (at 556, 1220, 1505, 4680, 4897 s),
  181 ≥ 50 ms. Frame interval median 15.0 ms, p99 25.8 ms.
- The summary's "delay spread 95.4 ms" was NOT jitter: the board clock ran 27 ppm slow against the PC's (about 0.2 s
  over 2 h). With a straight-line fit removed, jitter (p99 - median) = **14.2 ms**, and 13.9-18.5 ms per 10-minute
  window, with no trend. framelog.py now removes the drift itself (fixed 10-11); a 20 s check afterwards gave 12.8 ms.
  F1's 5.8 ms was a 60 s run, so p99 picked up fewer of the rare gaps.

| Claim | Outcome | Match | Helpful | What we saw |
|---|---|---|---|---|
| C0148 | inconclusive | diff | 0 | The claim is about WIRED XIAO boards. Our board in Wi-Fi mode streamed 2 h with 0 freezes ≥ 0.5 s (longest gap 135 ms), 66.6 fps steady in every 10-minute window, 0 bad JPEGs. The USB-side freeze it describes could not show up over Wi-Fi, and we skipped the wired soak because the build is Wi-Fi only |

Not applicable this run: C0149 (laptop vs desktop), C0151 (replug vs app restart) and C0152 (COM/zadig/JTAG fixes)
need a freeze to test, and there was none. C0150 (log text when a frozen feed returns) also needs a freeze, plus
Baballonia running, which it wasn't.
