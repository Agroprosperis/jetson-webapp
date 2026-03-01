import cv2
import numpy as np


class FramePreprocessor:
    """Downsample an input frame for analysis and enhance local contrast for line detection."""

    def __init__(
        self,
        min_side: int = 640,
        clahe_clip_limit: float = 6.0,
        clahe_tile_grid: tuple[int, int] = (6, 6),
        global_contrast_alpha: float = 1.35,
        global_contrast_beta: float = -10.0,
        oriented_bg_kernel: int = 71,
        oriented_bg_subtract_gain: float = 1.10,
        oriented_detail_gain: float = 2.0,
    ) -> None:
        self._min_side = min_side
        self._global_contrast_alpha = global_contrast_alpha
        self._global_contrast_beta = global_contrast_beta
        self._oriented_bg_kernel = oriented_bg_kernel
        self._oriented_bg_subtract_gain = oriented_bg_subtract_gain
        self._oriented_detail_gain = oriented_detail_gain
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid,
        )

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        downsampled_bgr = self._downsample(frame_bgr)
        return self._enhance_contrast_for_linea(downsampled_bgr)

    def _downsample(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        short_side = min(height, width)
        if short_side <= 0 or short_side <= self._min_side:
            return frame_bgr
        scale = self._min_side / short_side
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _enhance_contrast_for_linea(self, frame_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_chan)

        bg_h = cv2.GaussianBlur(l_enhanced, (self._oriented_bg_kernel, 1), 0)
        bg_v = cv2.GaussianBlur(l_enhanced, (1, self._oriented_bg_kernel), 0)
        bg = cv2.addWeighted(bg_h, 0.5, bg_v, 0.5, 0.0)
        residual = cv2.subtract(l_enhanced, bg)

        l_enhanced = cv2.addWeighted(
            l_enhanced,
            1.0 + self._oriented_bg_subtract_gain,
            bg,
            -self._oriented_bg_subtract_gain,
            0.0,
        )
        l_enhanced = cv2.addWeighted(
            l_enhanced,
            1.0,
            residual,
            self._oriented_detail_gain,
            0.0,
        )

        lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
        enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        enhanced_bgr = cv2.convertScaleAbs(
            enhanced_bgr,
            alpha=self._global_contrast_alpha,
            beta=self._global_contrast_beta,
        )
        return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
