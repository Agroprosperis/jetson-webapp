import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Regular + bold paths (adjust if you use a different family)
FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
# For HUD panel fonts below
FONT_PATH = FONT_PATH_REG

BASE_FONT_SIZE = 24
FONT_SIZE = int(BASE_FONT_SIZE * 1.2)  # +20%

# Try bold font first, fall back to regular
try:
    OBJECT_BANNER_FONT = ImageFont.truetype(FONT_PATH_BOLD, FONT_SIZE)
except OSError:
    OBJECT_BANNER_FONT = ImageFont.truetype(FONT_PATH_REG, FONT_SIZE)


# Colors for the object-count banner (RGB)
BANNER_BG_RGB = (0, 0, 0)          # black
BANNER_TEXT_RGB = (255, 255, 255)  # white

# Cache for banner so we don't recreate it every frame
_LAST_COUNT_VAL: int | None = None
_LAST_BANNER_BGR: np.ndarray | None = None


def _make_object_banner_bgr(text: str, pad_x: int = 10, pad_y: int = 6) -> np.ndarray:
    """
    Create a small RGB banner with Pillow and return it as a BGR numpy array.
    This is called only when text changes (via the cache in draw_object_count_banner).
    """
    # Measure text on a tiny temp image
    dummy = Image.new("RGB", (1, 1), (0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    text_bbox = draw.textbbox((0, 0), text, font=OBJECT_BANNER_FONT)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    banner_w = text_w + 2 * pad_x
    banner_h = text_h + 2 * pad_y

    # Real banner image
    img = Image.new("RGB", (banner_w, banner_h), BANNER_BG_RGB)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x, pad_y), text, font=OBJECT_BANNER_FONT, fill=BANNER_TEXT_RGB)

    banner_rgb = np.array(img, dtype=np.uint8)        # H x W x 3, RGB
    banner_bgr = banner_rgb[..., ::-1].copy()         # convert to BGR for OpenCV
    return banner_bgr


def draw_object_count_banner(frame_bgr: np.ndarray, count_val: int):
    """
    Fast banner using Pillow only on a tiny overlay, then pasting into the BGR frame.
    No RGBA, no full-frame conversions.
    """
    global _LAST_COUNT_VAL, _LAST_BANNER_BGR

    if count_val is None:
        return frame_bgr

    # Rebuild the small banner only when the count changes
    if _LAST_BANNER_BGR is None or count_val != _LAST_COUNT_VAL:
        text = f"Objects detected: {count_val}"
        _LAST_BANNER_BGR = _make_object_banner_bgr(text)
        _LAST_COUNT_VAL = count_val

    banner = _LAST_BANNER_BGR
    bh, bw = banner.shape[:2]
    h, w = frame_bgr.shape[:2]

    # Top-left corner for the banner
    x0, y0 = 10, 10

    if x0 >= w or y0 >= h:
        return frame_bgr

    # Clip if banner would go out of frame
    bw = min(bw, w - x0)
    bh = min(bh, h - y0)

    frame_bgr[y0:y0 + bh, x0:x0 + bw] = banner[:bh, :bw]
    return frame_bgr
