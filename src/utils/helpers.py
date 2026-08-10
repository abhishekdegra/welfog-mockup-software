"""
Helper utilities for image processing and general operations.
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * clamp(t, 0.0, 1.0)


def ensure_8bit(img: np.ndarray) -> np.ndarray:
    """Ensure image is 8-bit unsigned."""
    if img is None:
        return img

    if img.dtype == np.uint8:
        return img

    if img.dtype in (np.float32, np.float64):
        scale = 255.0 if img.max() <= 1.0 else 1.0
        return np.clip(img * scale, 0, 255).astype(np.uint8)

    if img.dtype == np.uint16:
        return (img / 257).astype(np.uint8)

    return np.clip(img, 0, 255).astype(np.uint8)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert any supported image layout to 3-channel BGR."""
    img = ensure_8bit(img)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


def to_bgra(img: np.ndarray) -> np.ndarray:
    """Convert any supported image layout to 4-channel BGRA."""
    img = ensure_8bit(img)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def get_quadrilateral_points(img: np.ndarray) -> np.ndarray:
    """Corner points of an image as TL, TR, BR, BL."""
    h, w = img.shape[:2]
    return np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four points clockwise as top-left, top-right, bottom-right, bottom-left.

    Uses angular sorting around the centroid, which stays correct for rotated
    and perspective-skewed quads where sum/difference heuristics break down.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

    if len(pts) != 4:
        rect = cv2.minAreaRect(pts)
        pts = cv2.boxPoints(rect).astype(np.float32)

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    clockwise = pts[np.argsort(angles)]

    # Rotate the sequence so it starts at the corner closest to the top-left.
    distances = np.linalg.norm(clockwise - clockwise.min(axis=0), axis=1)
    start = int(np.argmin(distances))
    ordered = np.roll(clockwise, -start, axis=0)

    return ordered.astype(np.float32)


def quad_size(pts: np.ndarray) -> Tuple[float, float]:
    """Average width and height of a quadrilateral ordered TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32)
    top = np.linalg.norm(pts[1] - pts[0])
    bottom = np.linalg.norm(pts[2] - pts[3])
    left = np.linalg.norm(pts[3] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])
    return float((top + bottom) / 2.0), float((left + right) / 2.0)


def rotate_points(pts: np.ndarray, degrees: float,
                  center: Optional[np.ndarray] = None) -> np.ndarray:
    """Rotate points around a center (defaults to their centroid)."""
    pts = np.asarray(pts, dtype=np.float32)
    if abs(degrees) < 1e-6:
        return pts.copy()

    if center is None:
        center = pts.mean(axis=0)

    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    return ((pts - center) @ matrix.T + center).astype(np.float32)


def rounded_rect_mask(size: Tuple[int, int], radius: int,
                      feather: int = 0) -> np.ndarray:
    """
    Build a white rounded-rectangle mask on a black canvas.

    Args:
        size: (height, width) of the mask
        radius: corner radius in pixels
        feather: gaussian feather radius in pixels

    Returns:
        Mask as uint8 (0-255)
    """
    h, w = int(size[0]), int(size[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = int(clamp(radius, 0, min(h, w) // 2))

    if radius <= 0:
        mask[:] = 255
    else:
        cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
        cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
        for cx, cy in ((radius, radius), (w - radius, radius),
                       (radius, h - radius), (w - radius, h - radius)):
            cv2.circle(mask, (cx, cy), radius, 255, -1)

    if feather > 0:
        kernel = int(feather) * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)

    return mask


def luminance(img_bgr: np.ndarray) -> np.ndarray:
    """Perceptual luminance of a float BGR image, in the same 0-1 range."""
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


def screen_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Screen blend two float images in the 0-1 range."""
    return 1.0 - (1.0 - base) * (1.0 - overlay)


def create_highlight_preservation_mask(img: np.ndarray,
                                       threshold: float = 0.7) -> np.ndarray:
    """
    Mask of the bright areas of an image, used to let reflections show through.

    Args:
        img: BGR or grayscale image
        threshold: brightness above which pixels count as highlights (0-1)

    Returns:
        Highlight mask as float32 (0-1)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    if gray.dtype != np.float32:
        gray = gray.astype(np.float32) / 255.0

    threshold = float(clamp(threshold, 0.0, 0.99))
    mask = np.clip((gray - threshold) / (1.0 - threshold), 0, 1)

    return mask.astype(np.float32)


def create_shadow_mask(img: np.ndarray, threshold: float = 0.45) -> np.ndarray:
    """Mask of the dark areas of an image as float32 (0-1)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    if gray.dtype != np.float32:
        gray = gray.astype(np.float32) / 255.0

    threshold = float(clamp(threshold, 0.01, 1.0))
    mask = np.clip((threshold - gray) / threshold, 0, 1)

    return mask.astype(np.float32)


def create_feather_mask(shape: Tuple[int, int], feather_radius: int) -> np.ndarray:
    """
    Mask that fades out towards the borders of the given shape.

    Args:
        shape: (height, width) of the mask
        feather_radius: fade width in pixels

    Returns:
        Mask as float32 (0-1)
    """
    h, w = shape
    mask = np.ones((h, w), dtype=np.uint8) * 255

    if feather_radius <= 0:
        return mask.astype(np.float32) / 255.0

    mask[:1, :] = 0
    mask[-1:, :] = 0
    mask[:, :1] = 0
    mask[:, -1:] = 0

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    alpha = np.clip(distance / float(feather_radius), 0, 1)

    return alpha.astype(np.float32)


def blend_images_alpha(base: np.ndarray, overlay: np.ndarray,
                       alpha: np.ndarray) -> np.ndarray:
    """
    Alpha blend overlay onto base.

    Args:
        base: Base image (BGR)
        overlay: Overlay image (BGR)
        alpha: Alpha mask, either float 0-1 or uint8 0-255

    Returns:
        Blended BGR image as uint8
    """
    base = to_bgr(base)
    overlay = to_bgr(overlay)

    if base.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]),
                             interpolation=cv2.INTER_AREA)

    alpha = alpha.astype(np.float32)
    if alpha.max() > 1.0:
        alpha /= 255.0

    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]

    result = overlay.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha)

    return np.clip(result, 0, 255).astype(np.uint8)


def resize_to_fit(img: np.ndarray, max_width: int, max_height: int,
                  allow_upscale: bool = False) -> np.ndarray:
    """
    Resize an image to fit inside the given box, preserving aspect ratio.

    Args:
        img: Input image
        max_width: Maximum width
        max_height: Maximum height
        allow_upscale: Whether images smaller than the box get enlarged

    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img

    ratio = min(max_width / w, max_height / h)
    if not allow_upscale:
        ratio = min(ratio, 1.0)

    if abs(ratio - 1.0) < 1e-3:
        return img

    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LANCZOS4

    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def find_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Largest contour in a binary mask, or None when the mask is empty."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def get_contour_bounding_quad(contour: np.ndarray) -> np.ndarray:
    """Four corner points approximating a contour."""
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        return order_points(approx.reshape(4, 2).astype(np.float32))

    rect = cv2.minAreaRect(contour)
    return order_points(cv2.boxPoints(rect).astype(np.float32))


def add_grain(img_float: np.ndarray, amount: float,
              mask: Optional[np.ndarray] = None,
              seed: int = 12345) -> np.ndarray:
    """
    Add monochrome film grain to a float BGR image in the 0-1 range.

    Args:
        img_float: Image as float32 (0-1)
        amount: Grain strength (0-1)
        mask: Optional float mask limiting where grain is applied
        seed: RNG seed so the grain stays stable between renders

    Returns:
        Image with grain
    """
    if amount <= 0:
        return img_float

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, amount * 0.09,
                       img_float.shape[:2]).astype(np.float32)

    if mask is not None:
        noise *= mask

    return np.clip(img_float + noise[:, :, np.newaxis], 0.0, 1.0)


def apply_vignette(img_float: np.ndarray, amount: float) -> np.ndarray:
    """Darken the corners of a float BGR image in the 0-1 range."""
    if amount <= 0:
        return img_float

    h, w = img_float.shape[:2]
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
    radius = np.sqrt(xs ** 2 + ys ** 2) / math.sqrt(2.0)

    falloff = 1.0 - amount * np.clip(radius - 0.35, 0, 1) ** 1.5

    return np.clip(img_float * falloff[:, :, np.newaxis], 0.0, 1.0)


def image_stats(img: np.ndarray) -> dict:
    """Basic dimension info for status displays."""
    if img is None:
        return {}

    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim > 2 else 1

    return {'width': w, 'height': h, 'channels': channels}
