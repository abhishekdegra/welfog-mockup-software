"""
Image transformation utilities: cover detection, perspective warping and masks.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..utils.helpers import (
    clamp, get_quadrilateral_points, order_points, quad_size,
    rotate_points, rounded_rect_mask, to_bgr,
)


class PerspectiveTransform:
    """Perspective operations for fitting a design into a phone cover region."""

    DETECT_MAX_SIZE = 900

    @staticmethod
    def detect_cover(phone_image: np.ndarray) -> np.ndarray:
        """
        Detect the back-cover region of a phone photo.

        Tries edge based quad detection, then subject segmentation, and finally
        falls back to a centered phone-shaped rectangle so the caller always
        receives a usable region.

        Args:
            phone_image: Phone image (BGR or BGRA)

        Returns:
            Four corner points ordered TL, TR, BR, BL as float32
        """
        bgr = to_bgr(phone_image)
        h, w = bgr.shape[:2]

        scale = min(1.0, PerspectiveTransform.DETECT_MAX_SIZE / max(h, w))
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr

        quad = PerspectiveTransform._detect_by_edges(small)
        if quad is None:
            quad = PerspectiveTransform._detect_by_segmentation(small)

        if quad is None:
            return PerspectiveTransform.default_cover(phone_image)

        if scale < 1.0:
            quad = quad / scale

        quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)

        return order_points(quad)

    @staticmethod
    def _detect_by_edges(img: np.ndarray) -> Optional[np.ndarray]:
        """Find the most phone-like quadrilateral using edge contours."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 60, 60)

        median = float(np.median(gray))
        lower = int(max(10, 0.66 * median))
        upper = int(min(255, 1.33 * median))

        edges = cv2.Canny(gray, lower, upper)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        return PerspectiveTransform._best_quad(contours, img.shape[:2])

    @staticmethod
    def _detect_by_segmentation(img: np.ndarray) -> Optional[np.ndarray]:
        """Find the phone by separating the subject from a plain background."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # The subject may be darker or lighter than the background; keep the
        # polarity whose foreground does not touch every border.
        candidates = [thresh, cv2.bitwise_not(thresh)]
        best = None
        best_area = 0.0
        total = img.shape[0] * img.shape[1]

        for candidate in candidates:
            cleaned = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE,
                                       np.ones((9, 9), np.uint8), iterations=2)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)

            if area < total * 0.05 or area > total * 0.98:
                continue

            if area > best_area:
                best_area = area
                rect = cv2.minAreaRect(contour)
                best = cv2.boxPoints(rect).astype(np.float32)

        return best

    @staticmethod
    def _best_quad(contours: List[np.ndarray],
                   shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Score candidate contours and return the best phone-cover quad."""
        total = shape[0] * shape[1]
        best = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < total * 0.06 or area > total * 0.97:
                continue

            rect = cv2.minAreaRect(contour)
            (rect_w, rect_h) = rect[1]
            if rect_w < 1 or rect_h < 1:
                continue

            rectangularity = area / (rect_w * rect_h)
            if rectangularity < 0.65:
                continue

            long_side = max(rect_w, rect_h)
            short_side = min(rect_w, rect_h)
            aspect = long_side / short_side

            # Phones are noticeably taller than wide; reward that shape.
            aspect_score = 1.0 - min(1.0, abs(aspect - 2.0) / 2.0)
            area_score = area / total
            score = area_score * 0.55 + rectangularity * 0.25 + aspect_score * 0.20

            if score > best_score:
                best_score = score
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    best = approx.reshape(4, 2).astype(np.float32)
                else:
                    best = cv2.boxPoints(rect).astype(np.float32)

        return best

    @staticmethod
    def default_cover(phone_image: np.ndarray, margin: float = 0.12) -> np.ndarray:
        """
        Centered phone-shaped rectangle, used when detection is not usable.

        Args:
            phone_image: Phone image
            margin: Fraction of the image height kept free above and below

        Returns:
            Four corner points ordered TL, TR, BR, BL as float32
        """
        h, w = phone_image.shape[:2]

        region_h = h * (1.0 - margin * 2.0)
        region_w = min(w * (1.0 - margin * 2.0), region_h * 0.48)

        cx, cy = w / 2.0, h / 2.0
        x1, x2 = cx - region_w / 2.0, cx + region_w / 2.0
        y1, y2 = cy - region_h / 2.0, cy + region_h / 2.0

        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    @staticmethod
    def source_quad(design_shape: Tuple[int, int],
                    cover_points: np.ndarray,
                    fit_mode: str = 'fill',
                    scale: float = 1.0,
                    offset_x: float = 0.0,
                    offset_y: float = 0.0,
                    rotation: float = 0.0) -> np.ndarray:
        """
        Compute which part of the design gets mapped onto the cover.

        Args:
            design_shape: (height, width) of the design image
            cover_points: Cover quad, ordered TL, TR, BR, BL
            fit_mode: 'fill' crops to the cover aspect, 'fit' letterboxes,
                'stretch' uses the whole design
            scale: Zoom factor applied to the design (1.0 = neutral)
            offset_x: Horizontal pan as a fraction of the crop width (-1 to 1)
            offset_y: Vertical pan as a fraction of the crop height (-1 to 1)
            rotation: Rotation of the design in degrees

        Returns:
            Four source points ordered TL, TR, BR, BL as float32
        """
        design_h, design_w = design_shape[:2]
        cover_w, cover_h = quad_size(cover_points)
        cover_aspect = cover_w / max(cover_h, 1e-6)
        design_aspect = design_w / max(design_h, 1e-6)

        if fit_mode == 'stretch':
            crop_w, crop_h = float(design_w), float(design_h)
        elif fit_mode == 'fit':
            # Expand the crop past the image bounds so nothing gets cut off.
            if design_aspect > cover_aspect:
                crop_w = float(design_w)
                crop_h = crop_w / cover_aspect
            else:
                crop_h = float(design_h)
                crop_w = crop_h * cover_aspect
        else:
            # 'fill': largest centered crop matching the cover aspect ratio.
            if design_aspect > cover_aspect:
                crop_h = float(design_h)
                crop_w = crop_h * cover_aspect
            else:
                crop_w = float(design_w)
                crop_h = crop_w / cover_aspect

        scale = max(0.05, float(scale))
        crop_w /= scale
        crop_h /= scale

        cx = design_w / 2.0 + offset_x * crop_w * 0.5
        cy = design_h / 2.0 + offset_y * crop_h * 0.5

        quad = np.array([
            [cx - crop_w / 2.0, cy - crop_h / 2.0],
            [cx + crop_w / 2.0, cy - crop_h / 2.0],
            [cx + crop_w / 2.0, cy + crop_h / 2.0],
            [cx - crop_w / 2.0, cy + crop_h / 2.0],
        ], dtype=np.float32)

        if abs(rotation) > 1e-6:
            quad = rotate_points(quad, -rotation, center=np.array([cx, cy],
                                                                  dtype=np.float32))

        return quad

    @staticmethod
    def warp_design(design_image: np.ndarray,
                    cover_points: np.ndarray,
                    output_shape: Tuple[int, int],
                    source_points: Optional[np.ndarray] = None,
                    mirror: bool = False) -> Optional[np.ndarray]:
        """
        Warp a design onto the cover quad, keeping the alpha channel.

        Args:
            design_image: Design image (any layout)
            cover_points: Destination quad ordered TL, TR, BR, BL
            output_shape: (height, width) of the output canvas
            source_points: Optional source quad; defaults to the whole design
            mirror: Flip the design horizontally

        Returns:
            Warped BGRA image, or None when inputs are invalid
        """
        if design_image is None or design_image.ndim < 2:
            return None

        from ..utils.helpers import to_bgra  # local import avoids a cycle at load

        design = to_bgra(design_image)
        if mirror:
            design = cv2.flip(design, 1)

        src = get_quadrilateral_points(design) if source_points is None \
            else np.asarray(source_points, dtype=np.float32)
        dst = order_points(cover_points)

        matrix = cv2.getPerspectiveTransform(src.astype(np.float32), dst)

        return cv2.warpPerspective(
            design,
            matrix,
            (int(output_shape[1]), int(output_shape[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    @staticmethod
    def create_cover_mask(cover_points: np.ndarray,
                          shape: Tuple[int, int],
                          feather_radius: int = 0,
                          corner_radius_percent: float = 0.0) -> np.ndarray:
        """
        Mask of the cover region with optional rounded corners and feathering.

        Args:
            cover_points: Cover quad ordered TL, TR, BR, BL
            shape: (height, width) of the output mask
            feather_radius: Edge fade width in pixels
            corner_radius_percent: Corner rounding as a percentage of the
                shorter cover side (0-50)

        Returns:
            Mask as float32 (0-1)
        """
        h, w = int(shape[0]), int(shape[1])
        dst = order_points(cover_points)
        cover_w, cover_h = quad_size(dst)

        cover_w = max(2.0, cover_w)
        cover_h = max(2.0, cover_h)

        if corner_radius_percent > 0:
            # Build the rounded rect in cover-local space, then warp it so the
            # rounding follows the same perspective as the design.
            local_w, local_h = int(round(cover_w)), int(round(cover_h))
            radius = int(min(local_w, local_h) * clamp(corner_radius_percent, 0, 50) / 100.0)
            local_mask = rounded_rect_mask((local_h, local_w), radius)

            src = np.array([
                [0, 0], [local_w - 1, 0],
                [local_w - 1, local_h - 1], [0, local_h - 1],
            ], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)

            mask = cv2.warpPerspective(local_mask, matrix, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [dst.reshape(-1, 1, 2).astype(np.int32)], 255)

        if feather_radius > 0:
            kernel = int(feather_radius) * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)

        return mask.astype(np.float32) / 255.0

    @staticmethod
    def scale_points(points: np.ndarray, factor: float) -> np.ndarray:
        """Scale points by a uniform factor."""
        return (np.asarray(points, dtype=np.float32) * float(factor)).astype(np.float32)

    @staticmethod
    def inset_quad(points: np.ndarray, percent: float) -> np.ndarray:
        """
        Shrink or grow a quad towards or away from its centroid.

        Args:
            points: Quad points
            percent: Positive shrinks, negative grows (as a percentage)
        """
        pts = np.asarray(points, dtype=np.float32)
        center = pts.mean(axis=0)
        factor = 1.0 - clamp(percent, -50.0, 50.0) / 100.0

        return ((pts - center) * factor + center).astype(np.float32)
