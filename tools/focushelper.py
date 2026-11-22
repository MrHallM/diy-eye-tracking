"""Live focus helper for the OV2640 cameras: the stream with a big sharpness score (variance of the Laplacian in
the centre of the image; higher = sharper) and the best score seen. Turn the lens until the number peaks.

    python tools/focushelper.py [http://192.168.8.192/]       q or Esc quits, r resets the best score

Works next to Baballonia on the patched firmware (several viewers at once). Focus on a target with fine detail
(printed text, a coin) placed where the lips / eye will be, not on a moving face.
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from eyerec import http_jpegs  # noqa: E402

url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.8.192/"
best, recent = 0.0, []
win = "focus helper - turn the lens until the score peaks (q = quit, r = reset best)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 720, 820)
for jpg, _ in http_jpegs(url):
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    h, w = img.shape
    centre = img[h // 4:3 * h // 4, w // 4:3 * w // 4]
    score = float(cv2.Laplacian(centre, cv2.CV_64F).var())
    recent = (recent + [score])[-10:]
    smooth = sum(recent) / len(recent)
    best = max(best, smooth)
    view = cv2.cvtColor(cv2.resize(img, (720, 720), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(view, (180, 180), (540, 540), (0, 200, 255), 1)
    panel = np.zeros((100, 720, 3), np.uint8)
    good = smooth >= 0.9 * best
    cv2.putText(panel, f"sharpness {smooth:6.0f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (0, 255, 0) if good else (0, 200, 255), 3)
    cv2.putText(panel, f"best {best:6.0f}   ({'near best' if good else 'keep turning'})", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.imshow(win, np.vstack([view, panel]))
    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
        break
    if k == ord("r"):
        best, recent = 0.0, []
cv2.destroyAllWindows()
