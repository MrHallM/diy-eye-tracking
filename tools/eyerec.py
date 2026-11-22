"""
eyerec.py - record the RAW eye-camera stream (every JPEG exactly as the board sent it, no re-encode) with a timestamp
per frame, and optionally re-serve it on localhost so Baballonia watches the very same frames while we record.
Plan item 7 ("record raw eye footage early") in the build plan; footage for offline work on slip/blink/drift.

    python tools/eyerec.py http://192.168.8.192/ --out tests/A1_<date>/run1 --seconds 120
    python tools/eyerec.py http://192.168.8.192/ --out ... --serve 8090      then in Baballonia set this camera's
                                                                             address to http://127.0.0.1:8090/
    python tools/eyerec.py extract tests/A1_<date>/run1 --every 30           dump every 30th frame as .jpg to look at
    python tools/eyerec.py left=http://<ip-L>/ right=http://<ip-R>/ mouth=http://192.168.8.192/ --out tests/vrc_<date>
                                                                             several cameras at once -> <out>/<name>/

Why --serve: the stock board firmware streams to ONE client (F1, C0038), so Baballonia and a recorder can't both
read it. With --serve this script is that one client and hands each frame on. Baballonia's IpCameraCapture just
scans for JPEG start/end markers, so any http:// URL that streams JPEGs works. The patched firmware
(firmware-patches/openiris-combo, 0009) serves several clients at once, so with it just record the board directly
next to Baballonia; --serve is only needed for stock boards.

Output folder: frames.mjpeg (the JPEGs back to back) + frames.csv (n, t_epoch, board_ts, offset, bytes, ok_jpeg,
target, phase) + frames.summary.txt + meta.json (the board's camera settings, chip temperature at start/end and the
stream drops, read over its REST API on port 81 when the firmware has one). t_epoch = time.time() when the frame
ARRIVED, the same clock as osclog.py and a1grid.py; with several cameras all share it. target/phase are filled only
when a1grid.py runs this recorder itself (--record); otherwise join on t_epoch. A dropped stream is reconnected
(the gap is listed in meta.json "drops"). Read it back with read_frames(folder), or:
ffmpeg -f mjpeg -i frames.mjpeg out_%05d.jpg
"""
import argparse
import csv
import http.client
import json
import statistics
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def http_jpegs(url, timeout=10):
    """Yield (jpeg_bytes, board_ts or None) from the board's multipart stream. read1 = no batching (see framelog.py)."""
    u = urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    conn.request("GET", u.path or "/")
    r = conn.getresponse()
    if r.status != 200:
        raise ConnectionError(f"{url} answered HTTP {r.status}")
    buf = b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            raise ConnectionError("stream closed by the board")
        buf += chunk
        while True:
            h = buf.find(b"Content-Length:")
            end = buf.find(b"\r\n\r\n", h) if h >= 0 else -1
            if end < 0:
                break
            head = buf[h:end].decode("latin-1")
            n = int(head.split("Content-Length:")[1].split("\r\n")[0])
            if len(buf) < end + 4 + n:
                break
            ts = None
            if "X-Timestamp:" in head:
                try:
                    ts = float(head.split("X-Timestamp:")[1].split("\r\n")[0])
                except ValueError:
                    pass
            yield buf[end + 4:end + 4 + n], ts
            buf = buf[end + 4 + n:]


def board_info(url, timeout=3):
    """Camera settings + chip temperature from the board's REST API (port 81), or {} on stock/old firmware.
    Wi-Fi networks and passwords are left out on purpose: recordings may be shared."""
    host = urlsplit(url).hostname
    info = {}
    for key, path in (("who_am_i", "/api/get/who_am_i/"), ("config", "/api/get/config/"),
                      ("temperature_c", "/api/get/temperature/")):
        try:
            conn = http.client.HTTPConnection(host, 81, timeout=timeout)
            conn.request("GET", path)
            data = json.loads(conn.getresponse().read())["result"]["data"]
        except Exception:                          # noqa: BLE001 - optional metadata
            continue
        if key == "config":
            cfg = json.loads("{" + data + "}") if isinstance(data, str) and not data.startswith("{") else \
                json.loads(data) if isinstance(data, str) else data
            info["camera_config"] = cfg.get("camera_config")
            info["mdns"] = (cfg.get("mdns_config") or {}).get("hostname")
            info["wifi_tx_power"] = cfg.get("wifi_tx_power")
        elif key == "temperature_c":
            info[key] = float(data)
        else:
            info[key] = data
    return info


class _Relay:
    """Latest frame + a condition, shared by the recorder and every localhost viewer."""
    def __init__(self):
        self.cv, self.frame, self.seq, self.viewers = threading.Condition(), None, 0, 0

    def put(self, jpg):
        with self.cv:
            self.frame, self.seq = jpg, self.seq + 1
            self.cv.notify_all()


def _serve(port, relay):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            seen = 0
            relay.viewers += 1
            try:
                while True:
                    with relay.cv:
                        relay.cv.wait_for(lambda: relay.seq != seen, timeout=2)
                        if relay.seq == seen:
                            continue
                        jpg, seen = relay.frame, relay.seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg)
                                     + jpg + b"\r\n")
            except (ConnectionError, OSError):
                pass                                  # viewer closed
            relay.viewers -= 1

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class EyeRecorder:
    """Background recorder. tag() -> (target, phase) labels each frame; a1grid.py passes its current target."""

    def __init__(self, source, folder, serve=None, tag=None, freeze=0.5, reconnect=True, t0=None):
        self.source, self.dir, self.tag, self.freeze = source, Path(folder), tag or (lambda: ("", "")), freeze
        self.stop_ev, self.relay, self.srv = threading.Event(), _Relay(), None
        self.n, self.bad, self.gaps, self.freezes, self.err = 0, 0, [], [], None
        self.serve, self.reconnect, self.drops, self.t0 = serve, reconnect, [], t0
        self.th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.serve:
            self.srv = _serve(self.serve, self.relay)
        self.t0 = self.t0 or time.time()
        self.meta = {"source": self.source, "t0_epoch": self.t0, "start": board_info(self.source)}
        self.th.start()
        return self

    def _run(self):
        last, off = None, 0
        with open(self.dir / "frames.mjpeg", "wb") as fm, open(self.dir / "frames.csv", "w", newline="") as fc:
            w = csv.writer(fc)
            w.writerow(["n", "t_epoch", "board_ts", "offset", "bytes", "ok_jpeg", "target", "phase"])
            while not self.stop_ev.is_set():
                try:
                    for jpg, ts in http_jpegs(self.source):
                        t = time.time()
                        self.relay.put(jpg)
                        ok = jpg[:2] == SOI and jpg.rstrip(b"\0")[-2:] == EOI
                        fm.write(jpg)
                        tgt, ph = self.tag()
                        w.writerow([self.n, f"{t:.6f}", f"{ts:.6f}" if ts is not None else "", off, len(jpg), int(ok), tgt, ph])
                        off += len(jpg)
                        if last is not None:
                            self.gaps.append(t - last)
                            if t - last >= self.freeze:
                                self.freezes.append((round(last - self.t0, 2), round(t - last, 2)))
                        last, self.n, self.bad = t, self.n + 1, self.bad + (not ok)
                        if self.stop_ev.is_set():
                            break
                except Exception as e:                # noqa: BLE001 - a dropped stream is a result: record it
                    self.err = f"{type(e).__name__}: {e}"
                    if not self.reconnect:
                        break
                    self.drops.append({"t": round(time.time() - self.t0, 2), "error": self.err})
                    self.stop_ev.wait(0.5)            # then reconnect; the gap shows up as a freeze
        self.t_end = last or time.time()

    def stop(self):
        self.stop_ev.set()
        self.th.join(3)
        if self.srv:
            self.srv.shutdown()
        self.meta["end"] = {"temperature_c": board_info(self.source).get("temperature_c")}
        self.meta["drops"] = self.drops
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return self.summary()

    def summary(self):
        dur = getattr(self, "t_end", time.time()) - self.t0
        g = sorted(x * 1000 for x in self.gaps)
        lines = [f"eyerec {self.source} · started {datetime.fromtimestamp(self.t0):%Y-%m-%d %H:%M:%S} · {dur:.0f} s"
                 + (f" · served on http://127.0.0.1:{self.serve}/" if self.serve else ""),
                 f"frames {self.n} · mean fps {self.n / dur:.1f}" if dur > 0 and self.n else "NO FRAMES: is another "
                 "client (Baballonia, a browser tab, framelog) holding the board's stream?",
                 f"frame interval ms: median {statistics.median(g):.1f} · p99 {g[int(.99 * (len(g) - 1))]:.1f} · max {g[-1]:.0f}" if g else "",
                 f"freezes (>= {self.freeze} s): {len(self.freezes)} {self.freezes[:20]}",
                 f"bad JPEG frames: {self.bad}",
                 f"stream drops (reconnected): {len(self.drops)}" if self.drops else "",
                 f"stream ended early: {self.err}" if self.err and not self.reconnect else ""]
        s = "\n".join(l for l in lines if l)
        (self.dir / "frames.summary.txt").write_text(s + "\n", encoding="utf-8")
        return s


def read_frames(folder):
    """Yield (row dict, jpeg bytes) for every recorded frame. Decode with cv2.imdecode(np.frombuffer(b, np.uint8), 0)."""
    folder = Path(folder)
    with open(folder / "frames.csv", newline="") as fc, open(folder / "frames.mjpeg", "rb") as fm:
        for row in csv.DictReader(fc):
            fm.seek(int(row["offset"]))
            yield row, fm.read(int(row["bytes"]))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        ap = argparse.ArgumentParser(prog="eyerec.py extract")
        ap.add_argument("cmd"); ap.add_argument("folder"); ap.add_argument("--every", type=int, default=30)
        a = ap.parse_args()
        out = Path(a.folder) / "jpg"
        out.mkdir(exist_ok=True)
        k = 0
        for row, jpg in read_frames(a.folder):
            if int(row["n"]) % a.every == 0:
                (out / f"{int(row['n']):06d}_{row['target'] or 'x'}_{row['phase'] or 'x'}.jpg").write_bytes(jpg); k += 1
        print(f"{k} frames -> {out}")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="http://<board-ip>/, or several as name=http://<ip>/")
    ap.add_argument("--out", help="output folder (default: eyerec_<time> here)")
    ap.add_argument("--seconds", type=float, default=0, help="0 = until Ctrl+C")
    ap.add_argument("--serve", type=int, help="single camera only: also re-serve on http://127.0.0.1:<port>/ "
                                              "(stock one-client firmware)")
    ap.add_argument("--no-reconnect", action="store_true", help="stop at the first dropped stream")
    a = ap.parse_args()
    out = Path(a.out or f"eyerec_{datetime.now():%Y%m%d_%H%M%S}")
    named = [s.split("=", 1) if "=" in s.split("://")[0] else (None, s) for s in a.sources]
    if len(named) > 1 and a.serve:
        ap.error("--serve works with one camera")
    t0 = time.time()
    recs = [EyeRecorder(url, out / name if name else out, a.serve, reconnect=not a.no_reconnect, t0=t0).start()
            for name, url in named]
    for rec in recs:
        print(f"recording {rec.source} -> {rec.dir}" + (f" · serving http://127.0.0.1:{a.serve}/" if a.serve else ""))
    print("Ctrl+C stops")
    try:
        while (not a.seconds or time.time() - t0 < a.seconds) and any(r.th.is_alive() for r in recs):
            time.sleep(1)
            if int(time.time() - t0) % 5 == 0:
                print(f"t={time.time() - t0:6.0f} s  " + "  |  ".join(
                    f"{r.dir.name}: {r.n} fr, {len(r.freezes)} frz, {len(r.drops)} drops" for r in recs))
    except KeyboardInterrupt:
        pass
    for rec in recs:
        print("\n" + rec.stop())


if __name__ == "__main__":
    main()
