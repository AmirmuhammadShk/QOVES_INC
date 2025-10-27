import base64
import io
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import svgwrite

# ---------- decoding / encoding ----------

def b64_to_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw))

def pil_to_np_rgba(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGBA"))

def b64_to_mask_gray(b64_str: str) -> np.ndarray:
    pil_img = b64_to_pil(b64_str)
    np_rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    return gray

def svg_to_b64(svg_text: str) -> str:
    return base64.b64encode(svg_text.encode("utf-8")).decode("utf-8")

# ---------- geometry helpers ----------

def resize_mask_nn(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize label mask with nearest-neighbor to preserve label ids."""
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

def unique_labels(mask_gray: np.ndarray) -> List[int]:
    vals = np.unique(mask_gray)
    return [int(v) for v in vals if v != 0]

def extract_contours_per_label(mask_gray: np.ndarray) -> Dict[int, List[np.ndarray]]:
    """
    Returns {label: [contours]}, contours are Nx1x2 arrays (OpenCV format).
    - Multi-label masks: label == pixel value.
    - Binary masks: label 1 = foreground.
    """
    labels = unique_labels(mask_gray)
    out: Dict[int, List[np.ndarray]] = {}

    if not labels and np.any(mask_gray > 0):
        labels = [1]

    if labels == [1] and len(np.unique(mask_gray)) == 2:
        # binary
        work = (mask_gray > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            out[1] = contours
        return out

    # multi-label
    for lab in labels:
        sel = (mask_gray == lab).astype(np.uint8) * 255
        if sel.max() == 0:
            continue
        contours, _ = cv2.findContours(sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            out[lab] = contours
    return out

def cnt_to_xy_list(cnt: np.ndarray) -> List[Tuple[float, float]]:
    return [(float(x), float(y)) for [[x, y]] in cnt.astype(float)]

# ---------- optional upright rotation ----------

def compute_eye_angle(landmarks: List[Tuple[float, float]]) -> Optional[float]:
    """
    Try to estimate tilt angle from two eye-like points.
    Strategy:
      - choose two farthest landmarks in the upper half as eye candidates,
        or fall back to the farthest pair overall.
      - angle is atan2(dy, dx) in degrees.
    """
    if len(landmarks) < 2:
        return None

    pts = np.array(landmarks, dtype=float)
    # prefer upper half
    median_y = np.median(pts[:,1])
    upper = pts[pts[:,1] <= median_y + 5]
    if len(upper) < 2:
        upper = pts

    # farthest pair
    dists = np.sum((upper[None, :, :] - upper[:, None, :])**2, axis=-1)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    p1, p2 = upper[i], upper[j]
    dy, dx = (p2[1] - p1[1]), (p2[0] - p1[0])
    angle = np.degrees(np.arctan2(dy, dx))
    return float(angle)

def rotate_image_and_mask(np_rgba: np.ndarray, mask_gray: np.ndarray, angle_deg: float):
    """Rotate both arrays about the center; keep size and fill empty with transparent/0."""
    h, w = mask_gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_img = cv2.warpAffine(np_rgba, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    rotated_mask = cv2.warpAffine(mask_gray, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_img, rotated_mask, M

def rotate_points(pts: List[Tuple[float, float]], M) -> List[Tuple[float, float]]:
    if M is None:
        return pts
    arr = np.hstack([np.array(pts, dtype=float), np.ones((len(pts),1))])
    xy = (arr @ M.T)
    return [(float(x), float(y)) for x, y in xy]

# ---------- SVG ----------

DEFAULT_LABEL_COLORS = {
    # (adjust as you like)
    1: "#FFDE59",  # forehead / hairline
    2: "#66D3FA",  # eyes
    3: "#9EE493",  # nose
    4: "#C88BFA",  # cheeks
    5: "#FA8072",  # mouth/chin
}

def contours_to_svg(width: int, height: int,
                    contours_map: Dict[int, List[np.ndarray]],
                    landmarks: List[Tuple[float, float]] = None) -> str:
    dwg = svgwrite.Drawing(size=(width, height), profile="tiny")
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="none"))

    for label, cnts in contours_map.items():
        color = DEFAULT_LABEL_COLORS.get(label, "#000000")
        group = dwg.g(id=f"label-{label}", class_=f"mask label-{label}",
                      fill="none", stroke=color, stroke_width=2)
        for cnt in cnts:
            pts = cnt_to_xy_list(cnt)
            if not pts:
                continue
            # simple path
            d = [f"M {pts[0][0]},{pts[0][1]}"] + [f"L {x},{y}" for x,y in pts[1:]] + ["Z"]
            group.add(dwg.path(d=" ".join(d)))
        dwg.add(group)

    if landmarks:
        lm = dwg.g(id="landmarks", class_="landmarks", fill="none", stroke="#000000")
        for k, (x, y) in enumerate(landmarks):
            lm.add(dwg.circle(center=(x, y), r=1.5))
            # index label (tiny)
            lm.add(dwg.text(str(k), insert=(x+3, y+3), font_size="8px"))
        dwg.add(lm)

    return dwg.tostring()
# --- NEW: robust tilt estimation and rotation helpers ---

from typing import List, Tuple, Optional
import numpy as np
import cv2

def _angle_from_two_points(p1: np.ndarray, p2: np.ndarray) -> float:
    dy, dx = (p2[1] - p1[1]), (p2[0] - p1[0])
    return float(np.degrees(np.arctan2(dy, dx)))

def estimate_tilt_angle(landmarks: List[Tuple[float, float]]) -> Optional[float]:
    """
    Estimate face roll (tilt) in degrees.
    Strategy:
      1) Prefer the farthest pair of landmarks in the *upper half* of the face
         (works even if we don't know eye indices).
      2) Fallback to PCA major axis of all landmarks.
    Return angle in degrees where 0≈level; positive = slope up to the right.
    To upright, rotate by -angle.
    """
    if len(landmarks) < 2:
        return None

    P = np.asarray(landmarks, dtype=float)
    y_med = np.median(P[:, 1])
    top = P[P[:, 1] <= y_med + 5]  # soft upper band

    if len(top) >= 2:
        D = np.sum((top[None, :, :] - top[:, None, :]) ** 2, axis=-1)
        i, j = np.unravel_index(np.argmax(D), D.shape)
        angle = _angle_from_two_points(top[i], top[j])
    else:
        mu = P.mean(axis=0)
        C = np.cov((P - mu).T)
        vals, vecs = np.linalg.eigh(C)
        v = vecs[:, np.argmax(vals)]
        angle = float(np.degrees(np.arctan2(v[1], v[0])))

    # normalize to a reasonable range
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    return float(np.clip(angle, -45.0, 45.0))

def rotate_image_and_mask(np_rgba: np.ndarray, mask_gray: np.ndarray, angle_deg: float):
    """
    Rotate both arrays about the center; keep canvas size.
    Use BILINEAR for image and NEAREST for mask to preserve labels.
    """
    h, w = mask_gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    img_r = cv2.warpAffine(np_rgba, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    mask_r = cv2.warpAffine(mask_gray, M, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_r, mask_r, M

def rotate_points(pts: List[Tuple[float, float]], M) -> List[Tuple[float, float]]:
    """Apply 2×3 affine matrix M to a list of (x,y) points."""
    if M is None or not pts:
        return pts
    A = np.hstack([np.array(pts, dtype=float), np.ones((len(pts), 1))])
    XY = (A @ M.T)
    return [(float(x), float(y)) for x, y in XY]
