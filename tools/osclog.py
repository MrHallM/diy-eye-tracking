"""
osclog.py - record every OSC message Baballonia sends (gaze, eyelids, mouth shapes) with a timestamp, so tests can
score its OUTPUT offline (A1/S1-S4/B1/D1 in analysis/library/chapters/evaluation.md).

    python tools/osclog.py --out tests/A1_<date>/run1/osc.csv                  (listens on 8888, Baballonia's default)
    python tools/osclog.py --port 8890 --forward 127.0.0.1:8888 --out ...      (logs AND passes the data on to VRCFT)

What Baballonia sends (ParameterSenderService.cs, repos/Baballonia @5dbd332), one UDP packet per value, every
processed frame, to 127.0.0.1:8888 by default (Settings > OSC send port):
  /LeftEyeX /LeftEyeY /LeftEyeLid /RightEyeX /RightEyeY /RightEyeLid    eye values after its calibration remap
  /jawOpen /mouthFunnel ... (45 face shapes)                            mouth tracker
  /tracking/eye/LeftRightPitchYaw, /tracking/eye/EyesClosedAmount        only with "native VRChat eye tracking" on
                                                                         (X * 45 deg, Y * -45 deg)
Port 8888 is also where the VRCFaceTracking Babble module listens. If VRCFT is running the bind fails: close VRCFT,
or set Baballonia's send port to 8890 and run with --port 8890 --forward 127.0.0.1:8888.

Timestamps are time.time() (wall clock, 0.1 us on Windows 11 / Python 3.13), the SAME clock a1grid.py and eyerec.py
write, so the three CSVs line up without any sync step. Ctrl+C or --seconds stops it and writes <out>.summary.txt.
"""
import argparse
import csv
import socket
import struct
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
NV = 4                                               # value columns; LeftRightPitchYaw carries 4 floats


def _str(b, i):
    """OSC string at i -> (text, next index). Strings are NUL-terminated and padded to a multiple of 4."""
    j = b.index(b"\0", i)
    return b[i:j].decode("utf-8", "replace"), (j + 4) & ~3


def parse(b):
    """Yield (address, [values]) from one OSC packet (a message or a #bundle of them)."""
    if b.startswith(b"#bundle\0"):
        i = 16                                       # "#bundle\0" + 8-byte timetag
        while i + 4 <= len(b):
            n = struct.unpack(">i", b[i:i + 4])[0]
            yield from parse(b[i + 4:i + 4 + n])
            i += 4 + n
        return
    addr, i = _str(b, 0)
    tags, i = _str(b, i) if i < len(b) else (",", i)
    vals = []
    for t in tags[1:]:
        if t == "f":
            vals.append(struct.unpack(">f", b[i:i + 4])[0]); i += 4
        elif t == "i":
            vals.append(struct.unpack(">i", b[i:i + 4])[0]); i += 4
        elif t == "d":
            vals.append(struct.unpack(">d", b[i:i + 8])[0]); i += 8
        elif t == "s":
            s, i = _str(b, i); vals.append(s)
        elif t in "TF":
            vals.append(int(t == "T"))
        else:                                        # a type we don't know: stop rather than misread the rest
            vals.append(f"?{t}"); break
    yield addr, vals


class OscLogger:
    """Background UDP listener -> CSV rows (t_epoch, address, v0..v3). Usable from other tools (a1grid.py)."""

    def __init__(self, port, out, forward=None, host="127.0.0.1"):
        self.out, self.stop_ev = Path(out), threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((host, port))
        except OSError as e:
            sys.exit(f"can't listen on {host}:{port} ({e}). VRCFaceTracking's Babble module probably holds it: close "
                     f"VRCFT, or set Baballonia's OSC send port to 8890 and use --port 8890 --forward 127.0.0.1:8888")
        self.sock.settimeout(0.2)
        self.fwd = None
        if forward:
            h, p = forward.rsplit(":", 1)
            self.fwd = (h, int(p))
        self.port, self.n, self.bad = port, 0, 0
        self.stats = defaultdict(lambda: [0, None, None, 0.0, float("inf"), float("-inf")])  # n, first, last, maxgap, min, max
        self.th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.th.start()
        return self

    def _run(self):
        with open(self.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_epoch", "address"] + [f"v{k}" for k in range(NV)])
            while not self.stop_ev.is_set():
                try:
                    pkt = self.sock.recv(65535)
                except socket.timeout:
                    continue
                except OSError:                      # Windows: ICMP port-unreachable from a dead --forward target
                    continue
                t = time.time()
                if self.fwd:
                    self.sock.sendto(pkt, self.fwd)
                try:
                    msgs = list(parse(pkt))
                except (ValueError, struct.error, IndexError):
                    self.bad += 1
                    continue
                for addr, vals in msgs:
                    w.writerow([f"{t:.6f}", addr] + [f"{v:.6g}" if isinstance(v, float) else v for v in vals[:NV]])
                    s = self.stats[addr]
                    if s[2] is not None:
                        s[3] = max(s[3], t - s[2])
                    s[0] += 1; s[1] = s[1] or t; s[2] = t
                    if vals and isinstance(vals[0], (int, float)):
                        s[4] = min(s[4], vals[0]); s[5] = max(s[5], vals[0])
                    self.n += 1

    def stop(self):
        self.stop_ev.set()
        self.th.join(2)
        self.sock.close()
        return self.summary()

    def summary(self):
        dur = time.time() - self.t0
        lines = [f"osclog port {self.port} · started {datetime.fromtimestamp(self.t0):%Y-%m-%d %H:%M:%S} · {dur:.0f} s"
                 f" · {self.n} messages · {len(self.stats)} addresses · unparsable packets {self.bad}",
                 "address | count | rate Hz | max gap ms | v0 min..max"]
        for addr, (n, t1, t2, gap, lo, hi) in sorted(self.stats.items()):
            rate = (n - 1) / (t2 - t1) if n > 1 and t2 > t1 else 0
            lines.append(f"{addr} | {n} | {rate:.1f} | {gap * 1000:.0f} | {lo:.3g}..{hi:.3g}")
        if not self.stats:
            lines.append("NO MESSAGES: is Baballonia running and sending to this port (Settings > OSC)?")
        s = "\n".join(lines)
        self.out.with_suffix(".summary.txt").write_text(s + "\n", encoding="utf-8")
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--forward", help="host:port to re-send every packet to (e.g. 127.0.0.1:8888 for VRCFT)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    ap.add_argument("--out", help="CSV path (default: osc_<time>.csv here)")
    a = ap.parse_args()
    lg = OscLogger(a.port, a.out or f"osc_{datetime.now():%Y%m%d_%H%M%S}.csv", a.forward).start()
    print(f"listening on 127.0.0.1:{a.port}" + (f", forwarding to {a.forward}" if a.forward else "") + " · Ctrl+C stops")
    last, next_print = 0, lg.t0 + 5
    try:
        while not a.seconds or time.time() - lg.t0 < a.seconds:
            time.sleep(0.2)
            if time.time() >= next_print:
                print(f"t={time.time() - lg.t0:6.0f} s  messages {lg.n}  (+{lg.n - last} in 5 s)  addresses {len(lg.stats)}")
                last, next_print = lg.n, next_print + 5
    except KeyboardInterrupt:
        pass
    print("\n" + lg.stop())


if __name__ == "__main__":
    main()
