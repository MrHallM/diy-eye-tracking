"""Measure how the mouth camera moved, by registering a RELAXED face to the relaxed
face saved at calibration time.

Why the whole relaxed face, not the nose: in this camera view the nose sits at the very
top edge of the frame (it can't be seen moving up) and is the darkest part of the image.
The follower only adjusts while the face is relaxed anyway, so expressions don't need to
be tolerated here; the full lower face gives far more texture to lock on to.

    reg = FaceRegistrar(reference_relaxed_frame)       # gray uint8 240x240
    m = reg.measure(relaxed_frame)                      # dict: dx, dy, scale, rot_deg, cc, ok

dx/dy = px the face moved in the image (+x right, +y down), scale = face size ratio,
rot_deg = in-image rotation, cc = correlation 0..1 after alignment (quality).
Method: CLAHE + high-pass so LED/gain changes matter less, then OpenCV ECC with an affine
model on a coarse-to-fine pyramid, restricted to the face area (mask).
"""
import numpy as np
import cv2

FACE_MASK_BOX = (45, 0, 165, 215)   # x, y, w, h: lower face, avoids the bright headset edge at left


def prep(gray):
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)).apply(gray).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 12)
    hp = cv2.GaussianBlur(hp, (0, 0), 1.5)
    return (hp - hp.mean()) / (hp.std() + 1e-6)


class FaceRegistrar:
    def __init__(self, reference, mask_box=FACE_MASK_BOX, levels=3):
        self.levels = levels
        x, y, w, h = mask_box
        self.mask = np.zeros(reference.shape, np.uint8); self.mask[y:y + h, x:x + w] = 255
        self.ref = prep(reference)

    def _pyr(self, img):
        p = [img]
        for _ in range(self.levels - 1): p.append(cv2.pyrDown(p[-1]))
        return p

    def measure(self, frame):
        src = prep(frame)
        R, S, M = self._pyr(self.ref), self._pyr(src), self._pyr(self.mask)
        W = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
        cc, ok = 0.0, True
        for lv in range(self.levels - 1, -1, -1):
            Wl = W.copy(); Wl[:, 2] /= 2 ** lv
            try:
                cc, Wl = cv2.findTransformECC(R[lv], S[lv], Wl, cv2.MOTION_AFFINE, crit,
                                              (M[lv] > 0).astype(np.uint8), 3)
            except cv2.error:
                ok = False; break
            W = Wl.copy(); W[:, 2] *= 2 ** lv
        # W maps reference coords -> frame coords. Describe it at the face centre.
        x, y, w, h = FACE_MASK_BOX; c = np.array([x + w / 2, y + h / 2, 1.0])
        moved = W @ c
        a, b = W[0, 0], W[1, 0]
        return dict(dx=float(moved[0] - c[0]), dy=float(moved[1] - c[1]),
                    scale=float(np.sqrt(abs(np.linalg.det(W[:, :2])))),
                    rot_deg=float(np.degrees(np.arctan2(b, a))), cc=float(cc), ok=ok, W=W)


def selftest(frame):
    """Warp a frame by known shift/scale/rotation and check they are recovered."""
    reg = FaceRegistrar(frame); worst = 0.0
    x, y, w, h = FACE_MASK_BOX; cx, cy = x + w / 2, y + h / 2
    for dx, dy, s, r in [(0, 0, 1, 0), (6, -4, 1, 0), (-10, 8, 1, 0), (3, 3, 1.08, 0), (-4, 2, 0.93, 3), (8, -6, 1.05, -4)]:
        A = cv2.getRotationMatrix2D((cx, cy), r, s); A[:, 2] += (dx, dy)
        warped = cv2.warpAffine(frame, A, frame.shape[::-1], borderMode=cv2.BORDER_REFLECT)
        m = reg.measure(warped)
        worst = max(worst, abs(m['dx'] - dx), abs(m['dy'] - dy))
    return worst
