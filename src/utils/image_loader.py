"""
Image loading and saving with validation and error handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from ..config import get_config

logger = logging.getLogger("mockup.images")


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded."""


class ImageSaveError(Exception):
    """Raised when an image cannot be written."""


class ImageLoader:
    """
    Reads and writes images through numpy buffers rather than OpenCV's path
    handling, so non-ASCII Windows paths work.
    """

    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    FILE_FILTER = ("Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;"
                   "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)")

    @staticmethod
    def load_image(file_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from disk.

        Args:
            file_path: Path to the image file

        Returns:
            Image as BGR or BGRA uint8

        Raises:
            ImageLoadError: When the file is missing, unsupported or unreadable
        """
        path = Path(file_path)

        if not path.exists():
            raise ImageLoadError(f"File not found: {path}")

        if not path.is_file():
            raise ImageLoadError(f"Not a file: {path}")

        if path.stat().st_size <= 0:
            raise ImageLoadError(f"File is empty: {path.name}")

        if not ImageLoader.is_supported(path):
            raise ImageLoadError(
                f"Unsupported format '{path.suffix}'. Supported: "
                f"{', '.join(sorted(ImageLoader.SUPPORTED_FORMATS))}"
            )

        image = None

        try:
            buffer = np.fromfile(str(path), dtype=np.uint8)
            if buffer.size:
                image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        except Exception as exc:
            logger.warning("OpenCV decode failed for %s: %s", path.name, exc)
            image = None

        if image is None:
            try:
                with Image.open(path) as pil_image:
                    pil_image.load()
                    if pil_image.mode not in ('RGB', 'RGBA'):
                        pil_image = pil_image.convert(
                            'RGBA' if 'A' in pil_image.mode else 'RGB')
                    array = np.array(pil_image)

                if array.ndim == 3 and array.shape[2] == 4:
                    image = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
                elif array.ndim == 3:
                    image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                else:
                    image = array
            except Exception as exc:
                logger.error("Could not read image %s: %s", path, exc)
                raise ImageLoadError(f"Could not read image: {exc}") from exc

        if image is None or not ImageLoader.validate_image(image):
            raise ImageLoadError(f"Image is empty or corrupted: {path.name}")

        if image.dtype == np.uint16:
            image = (image / 257).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        logger.debug("Loaded %s (%sx%s)", path.name, image.shape[1], image.shape[0])
        return image

    @staticmethod
    def is_supported(file_path: Union[str, Path]) -> bool:
        """Whether the file extension is one we can read."""
        return Path(file_path).suffix.lower() in ImageLoader.SUPPORTED_FORMATS

    @staticmethod
    def validate_image(img: np.ndarray) -> bool:
        """Whether the array holds a usable image."""
        return (img is not None and img.ndim >= 2
                and img.shape[0] > 0 and img.shape[1] > 0)

    @staticmethod
    def get_image_info(file_path: Union[str, Path]) -> dict:
        """Dimensions, channel count and file size, or an empty dict on error."""
        path = Path(file_path)

        try:
            with Image.open(path) as img:
                width, height = img.size
                channels = len(img.getbands())

            return {
                'width': width,
                'height': height,
                'channels': channels,
                'file_size': path.stat().st_size,
                'format': path.suffix.lower(),
            }
        except Exception as exc:
            logger.debug("get_image_info failed for %s: %s", path, exc)
            return {}

    @staticmethod
    def save_image(
        img: np.ndarray,
        file_path: Union[str, Path],
        quality: Optional[int] = None,
    ) -> bool:
        """
        Write an image to disk.

        Returns:
            True on success
        """
        ok, _error = ImageLoader.save_image_ex(img, file_path, quality)
        return ok

    @staticmethod
    def save_image_ex(
        img: np.ndarray,
        file_path: Union[str, Path],
        quality: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Write an image and return `(success, error_message)`.

        Creates parent folders automatically. Never raises for expected I/O
        failures — callers decide whether to escalate.
        """
        if not ImageLoader.validate_image(img):
            return False, "Nothing to save — image is empty"

        path = Path(file_path)
        ext = path.suffix.lower() or '.png'
        cfg = get_config()
        q = int(quality if quality is not None else cfg.export_quality)
        q = max(1, min(100, q))

        if ext in {'.jpg', '.jpeg'}:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            params = [cv2.IMWRITE_JPEG_QUALITY, q]
        elif ext == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(cfg.png_compression)]
        elif ext == '.webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, q]
        else:
            params = []

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            success, buffer = cv2.imencode(ext, img, params)
            if not success or buffer is None:
                return False, f"Encoder rejected format {ext}"

            buffer.tofile(str(path))
            if not path.exists() or path.stat().st_size <= 0:
                return False, f"Write produced an empty file: {path}"

            logger.info("Saved %s (%d bytes)", path, path.stat().st_size)
            return True, ""
        except OSError as exc:
            logger.error("Save failed for %s: %s", path, exc)
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected save failure for %s: %s", path, exc)
            return False, str(exc)

    @staticmethod
    def unique_path(file_path: Union[str, Path]) -> Path:
        """Return a non-colliding path by appending _1, _2, … as needed."""
        path = Path(file_path)
        if not path.exists():
            return path
        stem, suffix, parent = path.stem, path.suffix, path.parent
        index = 1
        while True:
            candidate = parent / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """File size in bytes, or 0 when unavailable."""
        try:
            return Path(file_path).stat().st_size
        except OSError:
            return 0
