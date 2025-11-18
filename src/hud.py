import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Regular + bold paths (adjust if you use a different family)
FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

BASE_FONT_SIZE = 24
FONT_SIZE = int(BASE_FONT_SIZE * 1.2)  # +20%

# Try bold font first, fall back to regular
try:
    OBJECT_BANNER_FONT = ImageFont.truetype(FONT_PATH_BOLD, FONT_SIZE)
except OSError:
    OBJECT_BANNER_FONT = ImageFont.truetype(FONT_PATH_REG, FONT_SIZE)

def rgb_to_bgr(rgb):
    r, g, b = rgb
    return (b, g, r)


# Define colors in normal RGB for readability
_BANNER_BG_RGB = (0, 0, 0)          # black
_BANNER_TEXT_RGB = (255, 255, 255)  # white

# Convert once to the BGR triplets that we will feed to Pillow
BANNER_BG_BGR = rgb_to_bgr(_BANNER_BG_RGB)
BANNER_TEXT_BGR = rgb_to_bgr(_BANNER_TEXT_RGB)


def draw_object_count_banner(frame_bgr: np.ndarray, count_val: int):
    """
    Fast banner: no transparency, no BGR<->RGB conversion.
    Uses Pillow for text (better fonts) but keeps frame in BGR layout.
    """
    if count_val is None:
        return frame_bgr

    # Pillow will treat this as RGB, but underlying bytes remain BGR.
    img = Image.fromarray(frame_bgr, mode="RGB")
    draw = ImageDraw.Draw(img)

    text = f"Objects detected: {count_val}"

    # Measure text once per call (font is cached globally)
    text_bbox = draw.textbbox((0, 0), text, font=OBJECT_BANNER_FONT)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    pad_x = 10
    pad_y = 6
    x0 = 10
    y0 = 10

    # Solid background rectangle (no alpha)
    bg_rect = (
        x0,
        y0,
        x0 + text_w + 2 * pad_x,
        y0 + text_h + 2 * pad_y,
    )
    # IMPORTANT: use BGR triplets; cv2 will later read them as BGR
    draw.rectangle(bg_rect, fill=BANNER_BG_BGR)

    # Text on top
    text_x = x0 + pad_x
    text_y = y0 + pad_y
    draw.text((text_x, text_y), text, font=OBJECT_BANNER_FONT, fill=BANNER_TEXT_BGR)

    # No RGB->BGR conversion here; layout is still BGR
    out_bgr = np.array(img)
    return out_bgr


def draw_hud(
    frame_bgr,
    send_ts: str | None = None,
    recv_ts: str | None = None,
    latency_ms: float | None = None,
    fps: float | None = None,
    panel_bg = (5, 10, 25, 180)
):
    """
    Draw a sci-fi style HUD using a TTF font via Pillow.
    frame_bgr: OpenCV BGR frame
    returns:   modified BGR frame
    """
    h, w = frame_bgr.shape[:2]

    # Convert BGR -> RGB -> PIL image
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img, "RGBA")

    # Colors
    cyan = (0, 255, 255, 255)
    accent = (0, 200, 255, 255)
    grid = (40, 80, 120, 140)
    text_dim = (130, 190, 220, 255)
    text_main = (230, 250, 255, 255)

    # Fonts
    try:
        font_label = ImageFont.truetype(FONT_PATH, 13)
        font_value = ImageFont.truetype(FONT_PATH, 18)
    except OSError:
        # Fallback if font missing
        font_label = ImageFont.load_default()
        font_value = ImageFont.load_default()

    panel_h = 80
    draw.rectangle((0, h - panel_h, w, h), fill=panel_bg)

    x_left = 16
    x_mid = w // 3 + 10
    x_right = 2 * w // 3 + 10
    y_base = h - panel_h + 18

    draw.line((w // 3, h - panel_h + 8, w // 3, h - 8), fill=grid, width=1)
    draw.line((2 * w // 3, h - panel_h + 8, 2 * w // 3, h - 8), fill=grid, width=1)

    def put_block(label, value, x, y):
        if value is None:
            return
        draw.text((x, y), label, font=font_label, fill=text_dim)
        draw.text((x, y + 20), value, font=font_value, fill=text_main)

    # Left: SEND
    if send_ts is not None:
        put_block("SEND", send_ts, x_left, y_base)

    # Middle: RECV
    if recv_ts is not None:
        put_block("RECV", recv_ts, x_mid, y_base)

    # Right: LAT / FPS
    if latency_ms is not None:
        put_block("LATENCY", f"{latency_ms:.1f} ms", x_right, y_base)
    if fps is not None:
        put_block("FPS", f"{fps:4.1f}", x_right, y_base - 22)

    # Convert back to BGR
    out_rgb = np.array(img)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr
