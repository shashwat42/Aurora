"""One native-size, orientation-aware input contract for datasets and the UI."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import warnings

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

PREPROCESS_VERSION = "srgb-exif-rgb-v1"
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 24_000_000


class ImageValidationError(ValueError):
    pass


@dataclass
class ValidatedImage:
    image: Image.Image
    source_sha256: str
    source_format: str
    original_size: tuple[int, int]


def validate_image(path: str | Path, *, max_bytes=MAX_BYTES, max_pixels=MAX_PIXELS) -> ValidatedImage:
    try:
        with Path(path).open("rb") as stream:
            content = stream.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ImageValidationError(f"Image exceeds the {max_bytes:,} byte input limit.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                if source.format not in {"JPEG", "PNG"}:
                    raise ImageValidationError("Upload a single-frame JPEG or PNG image; MPO is unsupported.")
                if getattr(source, "n_frames", 1) != 1:
                    raise ImageValidationError("Animated and multi-frame images are unsupported.")
                if source.mode != "RGB":
                    raise ImageValidationError(f"Image color mode {source.mode} is unsupported. Use an RGB JPEG or PNG.")
                if source.width * source.height > max_pixels:
                    raise ImageValidationError(f"Image exceeds the {max_pixels:,} decoded pixel limit.")
                original_size = source.size
                source_format = source.format
                source.load()
                oriented = ImageOps.exif_transpose(source)
                icc = source.info.get("icc_profile")
                if icc:
                    try:
                        oriented = ImageCms.profileToProfile(
                            oriented, ImageCms.ImageCmsProfile(BytesIO(icc)),
                            ImageCms.createProfile("sRGB"), outputMode="RGB",
                        )
                    except Exception as exc:
                        raise ImageValidationError("The embedded ICC color profile could not be converted to sRGB.") from exc
                # Explicitly remove source metadata after orientation/color conversion.
                clean = Image.frombytes("RGB", oriented.size, oriented.tobytes())
        return ValidatedImage(clean, hashlib.sha256(content).hexdigest(), source_format, original_size)
    except ImageValidationError:
        raise
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("The file is unreadable or is not a valid supported image.") from exc
