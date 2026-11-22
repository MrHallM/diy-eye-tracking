"""
framelog.py - log every frame an OpenIris camera delivers, so a test can MEASURE fps, freezes and bad frames
instead of eyeballing them (claim-register sessions F1-F3, see analysis/claims/TEST_PLANS.md).

    python tools/framelog.py http://<board-ip>/ --seconds 600 --out tests/F1_2026-10-12/wifi.csv
    python tools/framelog.py uvc:1 --seconds 600 --out tests/F2_2026-10-12/uvc.csv    (needs: pip install opencv-python)
    python tools/framelog.py uvc:list                                                  (which uvc index is which)

Wi-Fi mode reads the board's MJPEG stream (port 80, "/"). The board serves ONE client, so close Baballonia and
any browser tab first; zero frames = something else holds the stream. Each frame's X-Timestamp header (board
clock) is logged next to the PC clock: the board-to-PC delay is NOT absolute (two clocks), but its SPREAD is the
network jitter. Ctrl+C stops early and still writes the summary.

Outputs: <out>.csv (one row per frame) and <out>.summary.txt (paste its numbers into RESULT.md).
A freeze = no frame for --freeze seconds (default 0.5). A bad frame = a JPEG with no start/end marker
(Wi-Fi only: OpenCV hands over decoded UVC frames, so they can't be checked).
"""
import argparse
import csv
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
SOI, EOI = bytes([0xFF, 0xD8]), bytes([0xFF, 0xD9])


def http_frames(url, timeout):
    """Yield (n_bytes, jpeg_ok, board_timestamp or None) from a multipart/x-mixed-replace stream."""
    # http.client read1() returns whatever has ARRIVED. requests' iter_content(8192) waits for a full 8 KB and
    # iter_content(None) waits for the end of the stream: both batch frames and hide the real gaps (tested 10-10).
    import http.client
    from urllib.parse import urlsplit
    u = urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    conn.request("GET", u.path or "/")
    r = conn.getresponse()
    if r.status != 200:
        sys.exit(f"{url} answered HTTP {r.status}")
    buf = b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            raise ConnectionError("stream closed by the board")
        buf += chunk
        while True:
            h = buf.find(b"Content-Length:")
            if h < 0:
                break
            end = buf.find(b"\r\n\r\n", h)
            if end < 0:
                break
            head = buf[h:end].decode("latin-1")
            n = int(head.split("Content-Length:")[1].split("\r\n")[0])
            ts = None
            if "X-Timestamp:" in head:
                try:
                    ts = float(head.split("X-Timestamp:")[1].split("\r\n")[0])
                except ValueError:
                    pass
            if len(buf) < end + 4 + n:
                break
            jpg = buf[end + 4:end + 4 + n]
            yield len(jpg), jpg[:2] == SOI and jpg.rstrip(b"\0")[-2:] == EOI, ts
            buf = buf[end + 4 + n:]


def uvc_frames(index):
    try:
        import cv2
    except ImportError:
        sys.exit("uvc needs OpenCV: pip install opencv-python")
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        sys.exit(f"uvc:{index} would not open (held by another app? wrong index? try uvc:list)")
    print(f"uvc:{index} reports {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps (what the driver CLAIMS; the log measures the real rate)")
    while True:
        ok, frame = cap.read()
        yield (frame.nbytes, True, None) if ok else (None, False, None)


def uvc_list():
    import cv2
    for i in range(8):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            print(f"uvc:{i} {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
        cap.release()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="http://<board>/  or  uvc:<index>  or  uvc:list")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--out", help="CSV path (default: framelog_<time>.csv here)")
    ap.add_argument("--freeze", type=float, default=0.5, help="gap in seconds that counts as a freeze")
    a = ap.parse_args()
    if a.source == "uvc:list":
        uvc_list(); return
    out = Path(a.out or f"framelog_{datetime.now():%Y%m%d_%H%M%S}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = uvc_frames(int(a.source[4:])) if a.source.startswith("uvc:") else http_frames(a.source, 10)

    t0 = last = time.perf_counter()
    wall0 = datetime.now()
    gaps, sizes, delays, freezes, bad, n = [], [], [], [], 0, 0
    next_print = t0 + 5
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "t_pc_s", "gap_ms", "bytes", "board_ts_s", "ok_jpeg"])
        try:
            for size, ok, ts in frames:
                now = time.perf_counter()
                if size is None:                     # uvc read failed: counts toward a freeze, no row
                    if now - t0 > a.seconds:
                        break
                    continue
                gap = now - last
                if n and gap >= a.freeze:
                    freezes.append((round(last - t0, 2), round(gap, 2)))
                    print(f"  ⚠️ FREEZE {gap:.2f} s at t={last - t0:.1f} s")
                bad += not ok
                if n:
                    gaps.append(gap * 1000)
                sizes.append(size)
                if ts is not None:
                    delays.append((now - t0, now - ts))
                w.writerow([n, f"{now - t0:.4f}", f"{gap * 1000:.2f}" if n else "", size,
                            f"{ts:.6f}" if ts is not None else "", int(ok)])
                n, last = n + 1, now
                if now >= next_print and gaps:
                    print(f"t={now - t0:6.0f} s  frames {n}  fps(last) {1000 / statistics.mean(gaps[-200:]):5.1f}"
                          f"  freezes {len(freezes)}  bad {bad}")
                    next_print += 5
                if now - t0 > a.seconds:
                    break
        except KeyboardInterrupt:
            print("stopped early (Ctrl+C)")
        except Exception as e:                       # noqa: BLE001 - a dropped stream IS a result: log it
            print(f"stream ended: {type(e).__name__}: {e}")
            freezes.append((round(last - t0, 2), f"stream ended: {type(e).__name__}"))

    dur = last - t0
    # The two clocks drift apart (F3 10-11: the board ran 27 ppm slow = 0.2 s over 2 h, which read as "95 ms jitter"),
    # so subtract a straight-line fit first. The first 2 frames carry a pre-reset board clock: skip them.
    jit = None
    if len(delays) > 10:
        xs, ys = [x for x, _ in delays[2:]], [y for _, y in delays[2:]]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sum((x - mx) ** 2 for x in xs) or 1)
        res = [y - my - k * (x - mx) for x, y in zip(xs, ys)]
        jit = (pct(res, 99) - pct(res, 50)) * 1000
    lines = [
        f"source {a.source} · started {wall0:%Y-%m-%d %H:%M:%S} · {dur:.0f} s logged",
        f"frames {n} · mean fps {n / dur:.1f}" if dur > 0 else f"frames {n}",
        (f"frame interval ms: median {statistics.median(gaps):.1f} · p1 {pct(gaps, 1):.1f} · p99 {pct(gaps, 99):.1f}"
         f" · max {max(gaps):.0f}") if gaps else "frame interval: no data",
        f"frame bytes: median {statistics.median(sizes):.0f}" if sizes else "",
        f"board->PC delay spread, clock drift removed (p99 - median = network jitter): {jit:.1f} ms" if jit is not None else "",
        f"freezes (>= {a.freeze} s): {len(freezes)} {freezes[:20]}",
        f"bad JPEG frames: {bad}",
    ]
    summary = "\n".join(l for l in lines if l)
    print("\n" + summary)
    out.with_suffix(".summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
