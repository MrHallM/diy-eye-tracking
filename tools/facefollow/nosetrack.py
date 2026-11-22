"""Find how far the mouth camera has moved, using the nose as an anchor.

The nose and nostrils barely move when you make mouth expressions, so if the
nose sits somewhere else in the image than it did at calibration, the CAMERA
moved (or the headset slipped). This is the measuring part of "follow the face":
the follower later shifts Baballonia's crop by the same amount.

    tracker = NoseTracker(reference_frame)            # gray uint8, 240x240
    dx, dy, score = tracker.locate(frame)            # px the nose moved; score 0..1

Method: lighting-normalised edge image (CLAHE + high-pass) so LED brightness
changes don't fool it, then normalised template matching of the nose box
inside a search window, refined to sub-pixel with a parabola fit.
"""
import numpy as np
import cv2

# nose box in the 240x240 camera frame: nostrils + nose tip, above the upper lip
DEFAULT_BOX = (90, 2, 90, 68)       # x, y, w, h
SEARCH = 40                          # px searched in every direction


def prep(gray):
    """Even out lighting and keep shape: CLAHE, then subtract a heavy blur (high-pass)."""
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)).apply(gray)
    g = g.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 9)
    return cv2.GaussianBlur(hp, (0, 0), 1.2)


def _subpix(r, x, y):
    """Parabola fit around the integer peak (x, y) of score map r."""
    def fit(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
    ox = fit(r[y, x - 1], r[y, x], r[y, x + 1]) if 0 < x < r.shape[1] - 1 else 0.0
    oy = fit(r[y - 1, x], r[y, x], r[y + 1, x]) if 0 < y < r.shape[0] - 1 else 0.0
    return x + ox, y + oy


class NoseTracker:
    def __init__(self, reference, box=DEFAULT_BOX, search=SEARCH):
        self.box, self.search = box, search
        x, y, w, h = box
        self.tmpl = prep(reference)[y:y + h, x:x + w]

    def locate(self, frame):
        """(dx, dy, score): how far the nose moved vs the reference, + match quality 0..1."""
        x, y, w, h = self.box
        H, W = frame.shape
        x0, y0 = max(0, x - self.search), max(0, y - self.search)
        x1, y1 = min(W, x + w + self.search), min(H, y + h + self.search)
        win = prep(frame)[y0:y1, x0:x1]
        r = cv2.matchTemplate(win, self.tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(r)
        sx, sy = _subpix(r, mx, my)
        return x0 + sx - x, y0 + sy - y, float(score)


def selftest(frame):
    """Shift a frame by known amounts and check the tracker recovers them."""
    t = NoseTracker(frame); worst = 0.0
    for dx, dy in [(0, 0), (5, 0), (-7, 3), (12, -4), (-15, 10), (3.5, -2.5)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        moved = cv2.warpAffine(frame, M, frame.shape[::-1], borderMode=cv2.BORDER_REFLECT)
        ex, ey, s = t.locate(moved)
        worst = max(worst, abs(ex - dx), abs(ey - dy))
    return worst
