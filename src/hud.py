import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def draw_object_count_banner(frame_bgr, count_val: int):
    """
    Draw a semi-transparent banner in the top-left corner with the
    current number of detected objects, similar to the provided
    WorkflowImageData example.
    """
    if count_val is None:
        return frame_bgr

    # BGR -> RGB for Pillow
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base_img = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    text = f"Objects detected: {count_val}"

    try:
        font = ImageFont.truetype(FONT_PATH, 24)
    except OSError:
        font = ImageFont.load_default()

    x = 10
    y = 10
    pad = 6

    bbox = draw_overlay.textbbox((x, y), text, font=font)
    left, top, right, bottom = bbox
    text_w = right - left
    text_h = bottom - top

    bg_box = (x - pad, y - pad, x + text_w + pad, y + text_h + pad)
    draw_overlay.rectangle(bg_box, fill=(0, 0, 0, 160))  # ~60% opacity

    composed = Image.alpha_composite(base_img, overlay)
    draw_text = ImageDraw.Draw(composed)
    draw_text.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    out_rgb = composed.convert("RGB")
    out_np = np.ascontiguousarray(np.array(out_rgb, dtype=np.uint8))
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
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
