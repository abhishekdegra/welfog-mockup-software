"""
Image adjustment filters.

Every filter works on float32 BGR images in the 0-1 range, which keeps the
compositing pipeline free of repeated 8-bit rounding.
"""

from typing import Dict, Optional

import cv2
import numpy as np

from ..utils.helpers import clamp, ensure_8bit, luminance, screen_blend


class ImageFilters:
    """Colour and detail adjustments for the design layer."""

    @staticmethod
    def to_float(img: np.ndarray) -> np.ndarray:
        """Convert an 8-bit image to float32 in the 0-1 range."""
        if img.dtype == np.float32 and img.max() <= 1.001:
            return img
        return ensure_8bit(img).astype(np.float32) / 255.0

    @staticmethod
    def to_uint8(img: np.ndarray) -> np.ndarray:
        """Convert a float32 0-1 image back to 8-bit."""
        if img.dtype == np.uint8:
            return img
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def adjust_brightness(img: np.ndarray, value: float) -> np.ndarray:
        """Lift or lower overall brightness. Value range -100 to 100."""
        if value == 0:
            return img
        return np.clip(img + clamp(value, -100, 100) / 200.0, 0.0, 1.0)

    @staticmethod
    def adjust_contrast(img: np.ndarray, value: float) -> np.ndarray:
        """Expand or compress tonal range around mid grey. Range -100 to 100."""
        if value == 0:
            return img

        factor = max(0.1, 1.0 + clamp(value, -100, 100) / 100.0)

        return np.clip((img - 0.5) * factor + 0.5, 0.0, 1.0)

    @staticmethod
    def adjust_saturation(img: np.ndarray, value: float) -> np.ndarray:
        """Scale colour intensity. Range -100 (grey) to 100."""
        if value == 0 or img.ndim != 3:
            return img

        factor = max(0.0, 1.0 + clamp(value, -100, 100) / 100.0)
        lum = luminance(img)[:, :, np.newaxis]

        return np.clip(lum + (img - lum) * factor, 0.0, 1.0)

    @staticmethod
    def adjust_vibrance(img: np.ndarray, value: float) -> np.ndarray:
        """Boost muted colours while protecting already saturated ones."""
        if value == 0 or img.ndim != 3:
            return img

        amount = clamp(value, -100, 100) / 100.0
        lum = luminance(img)[:, :, np.newaxis]
        chroma = np.abs(img - lum).max(axis=2, keepdims=True)
        weight = 1.0 - np.clip(chroma * 2.0, 0.0, 1.0)

        return np.clip(lum + (img - lum) * (1.0 + amount * weight), 0.0, 1.0)

    @staticmethod
    def adjust_hue(img: np.ndarray, value: float) -> np.ndarray:
        """Rotate hues. Range -180 to 180 degrees."""
        if value == 0 or img.ndim != 3:
            return img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + clamp(value, -180, 180), 360.0)

        return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0.0, 1.0)

    @staticmethod
    def adjust_temperature(img: np.ndarray, value: float) -> np.ndarray:
        """Shift white balance warm (positive) or cool (negative)."""
        if value == 0 or img.ndim != 3:
            return img

        amount = clamp(value, -100, 100) / 100.0 * 0.25
        result = img.copy()
        result[:, :, 2] *= 1.0 + amount
        result[:, :, 0] *= 1.0 - amount

        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def adjust_tint(img: np.ndarray, value: float) -> np.ndarray:
        """Shift green (negative) to magenta (positive)."""
        if value == 0 or img.ndim != 3:
            return img

        amount = clamp(value, -100, 100) / 100.0 * 0.2
        result = img.copy()
        result[:, :, 1] *= 1.0 - amount

        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def adjust_gamma(img: np.ndarray, value: float) -> np.ndarray:
        """Apply gamma correction. Value range 0.1 to 3.0."""
        gamma = clamp(value, 0.1, 3.0)

        if abs(gamma - 1.0) < 1e-6:
            return img

        return np.power(np.clip(img, 0.0, 1.0), 1.0 / gamma)

    @staticmethod
    def adjust_exposure(img: np.ndarray, value: float) -> np.ndarray:
        """Multiply light in stops. Range -100 to 100 (about -2 to +2 EV)."""
        if value == 0:
            return img

        return np.clip(img * (2.0 ** (clamp(value, -100, 100) / 50.0)), 0.0, 1.0)

    @staticmethod
    def adjust_highlights(img: np.ndarray, value: float) -> np.ndarray:
        """Recover or bloom the bright end of the range. -100 to 100."""
        if value == 0:
            return img

        amount = clamp(value, -100, 100) / 100.0
        lum = luminance(img)[:, :, np.newaxis]
        mask = np.clip((lum - 0.5) * 2.0, 0.0, 1.0)

        return np.clip(img + img * mask * amount * 0.6, 0.0, 1.0)

    @staticmethod
    def adjust_shadows(img: np.ndarray, value: float) -> np.ndarray:
        """Open up or deepen the dark end of the range. -100 to 100."""
        if value == 0:
            return img

        amount = clamp(value, -100, 100) / 100.0
        lum = luminance(img)[:, :, np.newaxis]
        mask = np.clip((0.5 - lum) * 2.0, 0.0, 1.0)

        return np.clip(img + mask * amount * 0.35, 0.0, 1.0)

    @staticmethod
    def adjust_sharpness(img: np.ndarray, value: float) -> np.ndarray:
        """Unsharp mask for positive values, softening for negative ones."""
        if value == 0:
            return img

        amount = clamp(value, -100, 100) / 100.0

        if amount > 0:
            blurred = cv2.GaussianBlur(img, (0, 0), 1.2)
            return np.clip(img + (img - blurred) * amount * 1.5, 0.0, 1.0)

        return cv2.GaussianBlur(img, (0, 0), abs(amount) * 3.0 + 0.1)

    @staticmethod
    def adjust_blur(img: np.ndarray, value: float) -> np.ndarray:
        """Gaussian blur. Range 0 to 100."""
        value = clamp(value, 0, 100)

        if value <= 0:
            return img

        return cv2.GaussianBlur(img, (0, 0), value / 100.0 * 12.0 + 0.1)

    @staticmethod
    def adjust_clarity(img: np.ndarray, value: float) -> np.ndarray:
        """Local contrast boost that keeps edges from haloing."""
        if value == 0:
            return img

        amount = clamp(value, -100, 100) / 100.0
        blurred = cv2.GaussianBlur(img, (0, 0), 8.0)

        return np.clip(img + (img - blurred) * amount, 0.0, 1.0)

    @staticmethod
    def apply_adjustments(img: np.ndarray, settings: Dict[str, float]) -> np.ndarray:
        """
        Run the full colour pipeline in a sensible order.

        Args:
            img: float32 BGR image (0-1)
            settings: Adjustment values keyed by name

        Returns:
            Adjusted float32 BGR image
        """
        result = img

        result = ImageFilters.adjust_exposure(result, settings.get('exposure', 0))
        result = ImageFilters.adjust_brightness(result, settings.get('brightness', 0))
        result = ImageFilters.adjust_contrast(result, settings.get('contrast', 0))
        result = ImageFilters.adjust_highlights(result, settings.get('highlights', 0))
        result = ImageFilters.adjust_shadows(result, settings.get('shadows', 0))
        result = ImageFilters.adjust_gamma(result, settings.get('gamma', 1.0))
        result = ImageFilters.adjust_temperature(result, settings.get('temperature', 0))
        result = ImageFilters.adjust_tint(result, settings.get('tint', 0))
        result = ImageFilters.adjust_hue(result, settings.get('hue', 0))
        result = ImageFilters.adjust_saturation(result, settings.get('saturation', 0))
        result = ImageFilters.adjust_vibrance(result, settings.get('vibrance', 0))
        result = ImageFilters.adjust_clarity(result, settings.get('clarity', 0))
        result = ImageFilters.adjust_sharpness(result, settings.get('sharpness', 0))
        result = ImageFilters.adjust_blur(result, settings.get('blur', 0))

        return result

    @staticmethod
    def auto_enhance(img: np.ndarray) -> np.ndarray:
        """CLAHE based auto contrast on an 8-bit or float image."""
        as_float = img.dtype != np.uint8
        img8 = ImageFilters.to_uint8(img) if as_float else img

        if img8.ndim == 3:
            lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]),
                                    cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img8)

        return ImageFilters.to_float(enhanced) if as_float else enhanced

    @staticmethod
    def auto_match_tone(design: np.ndarray, reference: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        strength: float = 1.0) -> np.ndarray:
        """
        Nudge the design's brightness and contrast towards a reference region.

        Args:
            design: float32 BGR design layer (0-1)
            reference: float32 BGR reference image (0-1)
            mask: Optional float mask selecting the reference region
            strength: How much of the correction to apply (0-1)

        Returns:
            Tone matched design layer
        """
        if strength <= 0:
            return design

        ref_lum = luminance(reference)
        if mask is not None and mask.max() > 0:
            weights = mask
            ref_mean = float((ref_lum * weights).sum() / weights.sum())
            ref_std = float(np.sqrt(((ref_lum - ref_mean) ** 2 * weights).sum()
                                    / weights.sum()))
        else:
            ref_mean = float(ref_lum.mean())
            ref_std = float(ref_lum.std())

        design_lum = luminance(design)
        design_mean = float(design_lum.mean())
        design_std = float(design_lum.std()) or 1e-6

        gain = clamp(ref_std / design_std, 0.6, 1.6)
        gain = 1.0 + (gain - 1.0) * strength
        offset = (ref_mean - design_mean) * 0.5 * strength

        return np.clip((design - design_mean) * gain + design_mean + offset, 0.0, 1.0)

    @staticmethod
    def apply_material_shading(design: np.ndarray, phone: np.ndarray,
                               texture_strength: float = 0.0,
                               reflection_strength: float = 0.0,
                               shadow_strength: float = 0.0) -> np.ndarray:
        """
        Transfer the phone's shading onto the design so it reads as printed on.

        Args:
            design: float32 BGR design layer (0-1)
            phone: float32 BGR phone image (0-1)
            texture_strength: Amount of local shading detail transferred (0-1)
            reflection_strength: Amount of highlight screening (0-1)
            shadow_strength: Amount of shadow multiplication (0-1)

        Returns:
            Shaded design layer
        """
        result = design

        if texture_strength > 0 or reflection_strength > 0 or shadow_strength > 0:
            phone_lum = luminance(phone)

        if texture_strength > 0:
            base = cv2.GaussianBlur(phone_lum, (0, 0), 12.0)
            shading = phone_lum / np.clip(base, 0.02, None)
            shading = np.clip(shading, 0.4, 1.8)
            factor = 1.0 + (shading - 1.0) * texture_strength

            result = np.clip(result * factor[:, :, np.newaxis], 0.0, 1.0)

        if reflection_strength > 0:
            highlights = np.clip((phone_lum - 0.62) / 0.38, 0.0, 1.0)
            highlights = (highlights ** 1.4) * reflection_strength

            result = screen_blend(result, highlights[:, :, np.newaxis])

        if shadow_strength > 0:
            shadows = np.clip((0.4 - phone_lum) / 0.4, 0.0, 1.0)
            shadows = (shadows ** 1.2) * shadow_strength

            result = np.clip(result * (1.0 - shadows[:, :, np.newaxis] * 0.85),
                             0.0, 1.0)

        return result
