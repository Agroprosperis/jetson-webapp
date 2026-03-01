import math

import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from models.registry import MODULE_BUILD_FUNCS
from util.slconfig import SLConfig
from .line import Line


class LineDetector:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        threshold: float = 0.04,
        min_length_ratio: float = 0.3,
        axis_angle_tol_deg: float = 5.0,
        device: str = "cuda:0",
    ) -> None:
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._model_input_size = 640
        self._threshold = threshold
        self._min_length_ratio = min_length_ratio
        self._axis_angle_tol_deg = axis_angle_tol_deg
        
        requested_device = device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self._device = requested_device
        
        self._transform = transforms.Compose(
            [
                transforms.Resize((self._model_input_size, self._model_input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.538, 0.494, 0.453],
                    std=[0.257, 0.263, 0.273],
                ),
            ]
        )
        
        self._model, self._postprocessor = self._build_model()

    @property
    def device(self) -> str:
        return self._device

    def __call__(self, image_rgb: np.ndarray) -> list[Line]:
        lines_np, scores_np, frame_w, frame_h = self._detect_lines(image_rgb)
        return [
            self._build_line(line_xyxy, score, frame_w, frame_h)
            for line_xyxy, score in (
                (line_xyxy, float(score))
                for line_xyxy, score in zip(lines_np, scores_np)
            )
            if self.is_valid(line_xyxy, score, frame_w, frame_h)
        ]

    def _build_model(self) -> tuple[any, any]:
        cfg = SLConfig.fromfile(self._config_path)
        if "HGNetv2" in cfg.backbone:
            cfg.pretrained = False
        cfg.multiscale = None
        builder_name = getattr(cfg, "modelname")
        assert builder_name in MODULE_BUILD_FUNCS._module_dict
        build_func = MODULE_BUILD_FUNCS.get(builder_name)
        model, postprocessor = build_func(cfg)
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        model.load_state_dict(state)
        model = model.deploy().to(self._device)
        postprocessor = postprocessor.deploy().to(self._device)
        model.eval()
        postprocessor.eval()
        return model, postprocessor

    def _detect_lines(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        frame_pil = Image.fromarray(image_rgb)
        w, h = frame_pil.size
        
        with torch.no_grad():
            outputs = self._model(self._transform(frame_pil).unsqueeze(0).to(self._device))
            lines, scores = self._postprocessor(outputs, torch.tensor([[w, h]], device=self._device))
        
        return lines[0].detach().cpu().numpy(), scores[0].detach().cpu().numpy(), w, h

    def is_valid(self, line_xyxy: np.ndarray, score: float, w: int, h: int) -> bool:
        if score < self._threshold:
            return False

        x1f, y1f, x2f, y2f = [float(value) for value in line_xyxy]
        dx, dy = x2f - x1f, y2f - y1f
        length = math.hypot(dx, dy)
        
        min_length_px = self._min_length_ratio * math.hypot(w, h)
        if length < min_length_px:
            return False
        
        theta = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        horizontal_delta = min(theta, 180.0 - theta)
        vertical_delta = abs(theta - 90.0)
        
        return min(horizontal_delta, vertical_delta) <= self._axis_angle_tol_deg

    def _build_line(self, line_xyxy: np.ndarray, score: float, w: int, h: int) -> Line:
        x1f, y1f, x2f, y2f = [float(value) for value in line_xyxy]
        dx, dy = x2f - x1f, y2f - y1f

        theta_deg = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        horizontal_delta = min(theta_deg, 180.0 - theta_deg)
        vertical_delta = abs(theta_deg - 90.0)
        
        orientation = "horizontal" if horizontal_delta <= vertical_delta else "vertical"
        if orientation == "vertical":
            y_ref = h * 0.5
            if abs(dy) > 1e-6:
                t = (y_ref - y1f) / dy
                axis_pos = x1f + t * dx
            else:
                axis_pos = (x1f + x2f) * 0.5
        else:
            x_ref = w * 0.5
            if abs(dx) > 1e-6:
                t = (x_ref - x1f) / dx
                axis_pos = y1f + t * dy
            else:
                axis_pos = (y1f + y2f) * 0.5
        
        return Line(
            x1=x1f,
            y1=y1f,
            x2=x2f,
            y2=y2f,
            score=score,
            orientation=orientation,
            axis_pos=float(axis_pos),
            theta=math.atan2(dy, dx),
        )
