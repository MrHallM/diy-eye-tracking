"""
latency.py - glass-to-glass lag of a camera stream: a full-screen window flips black/white, the camera looks at the
screen, and we time each flip until the camera's frames show it (claim-register F2 / C0057).

    python tools/latency.py uvc:1 --out tests/F2_<date>/lat_uvc.csv
    python tools/latency.py http://<board-ip>/ --out tests/F2_<date>/lat_wifi.csv

Point the camera anywhere at the screen (10-20 cm is fine). It shows 8 s of black to aim, then 30 flips. Esc quits.
The number INCLUDES the monitor's own delay (render + scan-out, typically 10-30 ms), so compare modes with each
other, not with a spec sheet. Frames are timestamped when the PC receives them.
"""
import argparse
import csv
import os
import statistics
import sys
import threading
import time

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
samples, stop = [], threading.Event()          # (pc_time, mean_brightness)


def reader_uvc(index, backend):
    # The capture is OPENED by main() on the main thread and never released here: on 10-10 DirectShow opened in this
    # thread plus the full-screen window hung the whole run (the process exits with os._exit instead).
    cap = index
    while not stop.is_set():
        ok, f = cap.read()
        if ok:
            samples.append((time.perf_counter(), float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())))


def reader_http(url):
    import http.client
    from urllib.parse import urlsplit
    u = urlsplit(url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
    c.request("GET", u.path or "/")
    r, buf = c.getresponse(), b""
    while not stop.is_set():
        buf += r.read1(65536)
        while True:
            h = buf.find(b"Content-Length:")
            e = buf.find(b"\r\n\r\n", h)
            if h < 0 or e < 0:
                break
            n = int(buf[h:e].split(b"Content-Length:")[1].split(b"\r\n")[0])
            if len(buf) < e + 4 + n:
                break
            t = time.perf_counter()
            img = cv2.imdecode(np.frombuffer(buf[e + 4:e + 4 + n], np.uint8), cv2.IMREAD_GRAYSCALE)
            buf = buf[e + 4 + n:]
            if img is not None:
                samples.append((t, float(img.mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flips", type=int, default=30)
    ap.add_argument("--backend", choices=["dshow", "msmf"], default="dshow",
                    help="UVC backend; Baballonia uses DirectShow on Windows (OpenCvCapture.cs), so dshow is the default")
    a = ap.parse_args()

    def watchdog():                                   # never leave a black full-screen window behind
        print("watchdog: 100 s passed, exiting", flush=True)
        os._exit(2)
    wd = threading.Timer(100, watchdog)
    wd.daemon = True
    wd.start()

    if a.source.startswith("uvc:"):
        cap = cv2.VideoCapture(int(a.source[4:]), {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}[a.backend])
        if not cap.isOpened():
            print(f"{a.source} ({a.backend}) would not open"); os._exit(1)
        th = threading.Thread(target=reader_uvc, args=(cap, a.backend), daemon=True)
    else:
        th = threading.Thread(target=reader_http, args=(a.source,), daemon=True)
    th.start()

    win = "latency test (Esc quits)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    black, white = np.zeros((90, 160), np.uint8), np.full((90, 160), 255, np.uint8)
    flips, t_end = [], time.perf_counter() + 8
    while time.perf_counter() < t_end:               # aiming time
        cv2.imshow(win, black)
        if cv2.waitKey(30) == 27:
            stop.set(); os._exit(0)
    for i in range(a.flips * 2):
        img = white if i % 2 == 0 else black
        cv2.imshow(win, img)
        cv2.waitKey(1)                                # pumps the window: the frame is handed to the compositor here
        flips.append((time.perf_counter(), "white" if i % 2 == 0 else "black"))
        t_next = time.perf_counter() + 0.45 + 0.1 * np.random.rand()   # jitter so flips don't lock to the frame rate
        while time.perf_counter() < t_next:
            if cv2.waitKey(5) == 27:
                break
    time.sleep(0.5)
    stop.set()
    cv2.destroyAllWindows()

    t = np.array([s[0] for s in samples])
    b = np.array([s[1] for s in samples])
    if len(b) < 50:
        print(f"only {len(b)} frames received: is another app holding the camera?"); os._exit(1)
    lo, hi = np.percentile(b, 5), np.percentile(b, 95)
    mid = (lo + hi) / 2
    print(f"frames {len(b)} · brightness black {lo:.0f} / white {hi:.0f} (needs a clear gap to be valid)")
    lat = []
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["flip", "to", "t_flip", "t_seen", "latency_ms"])
        for k, (tf, to) in enumerate(flips):
            after = np.nonzero(t > tf)[0]
            hit = next((j for j in after if (b[j] > mid if to == "white" else b[j] < mid)), None)
            if hit is None or t[hit] - tf > 0.4:
                w.writerow([k, to, f"{tf:.4f}", "", ""]); continue
            ms = (t[hit] - tf) * 1000
            lat.append(ms)
            w.writerow([k, to, f"{tf:.4f}", f"{t[hit]:.4f}", f"{ms:.1f}"])
    if not lat:
        print("no flips detected: is the camera pointed at the screen?"); os._exit(1)
    s = (f"source {a.source}{' (' + a.backend + ')' if a.source.startswith('uvc:') else ''} · {len(lat)}/{len(flips)} flips seen · glass-to-glass ms: median "
         f"{statistics.median(lat):.1f} · p10 {np.percentile(lat, 10):.1f} · p90 {np.percentile(lat, 90):.1f} · "
         f"min {min(lat):.1f} · max {max(lat):.1f} · black {lo:.0f} / white {hi:.0f}")
    print(s, flush=True)
    open(a.out.replace(".csv", ".summary.txt"), "w", encoding="utf-8").write(s + "\n")
    os._exit(0)                                       # skip camera teardown (see reader_uvc)


if __name__ == "__main__":
    main()
