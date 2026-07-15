import numpy as np
import supervision as sv
import logging
import cv2
import os
from datetime import datetime

from s_value_model import calculate_s_value

BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(
    text_scale = 1.2,
    text_thickness = 2,
    text_padding = 8,
)

SEEN_TRACK_IDS_BY_CLASS: dict[str, set[int]] = {}

LOGGER = logging.getLogger("visualization")


def _format_banner_model_name(args) -> str:
    model_path = getattr(args, "model_path", "") or ""
    if not model_path:
        return "unknown"

    return os.path.basename(model_path)


def _format_banner_mode(args, frame_w: int, frame_h: int) -> str:
    fps = getattr(args, "fps", None)
    if fps is None:
        return f"{frame_w}x{frame_h}"

    return f"{frame_w}x{frame_h} @ {fps}fps"


def _format_optional_dimension(value) -> str:
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return "N/D"
    return str(dimension) if dimension >= 32 else "N/D"

def _format_banner_inference_mode(args) -> str:
    inference_width = _format_optional_dimension(getattr(args, "model_inference_width", None))
    inference_height = _format_optional_dimension(getattr(args, "model_inference_height", None))
    return f"{inference_width}x{inference_height}"


def class_name_for_id(args, cls_id: int) -> str:
    names = getattr(args, "class_names", None)
    if isinstance(names, dict):
        return str(names.get(cls_id, names.get(str(cls_id), str(cls_id))))
    return str(names[cls_id]) if names is not None and 0 <= cls_id < len(names) else str(cls_id)


def _display_class_names(args, class_counts: dict[str, int]) -> list[str]:
    names = getattr(args, "class_names", None)
    if isinstance(names, dict):
        ordered_names = [str(names[key]) for key in sorted(names, key=str)]
    elif names is not None:
        ordered_names = [str(name) for name in names]
    else:
        ordered_names = []

    display_names = [
        name
        for name in ordered_names
        if name and not name.lower().startswith("background")
    ]
    for name in class_counts:
        if name not in display_names:
            display_names.append(name)
    return display_names


def display_class_names(args, class_counts: dict[str, int]) -> list[str]:
    return _display_class_names(args, class_counts)


def _format_class_counts(args, class_counts: dict[str, int]) -> str:
    parts = [
        f"{name}: {int(class_counts.get(name, 0))}"
        for name in _display_class_names(args, class_counts)
    ]
    return "Objects: " + (" | ".join(parts) if parts else "none")


def class_s_values_from_counts(args, class_counts: dict[str, int]) -> dict[str, float]:
    return {
        name: calculate_s_value(int(class_counts.get(name, 0)))
        for name in display_class_names(args, class_counts)
    }


def _format_s_values(args, class_counts: dict[str, int], s_values: dict[str, float] | None = None) -> str:
    values = s_values if s_values is not None else class_s_values_from_counts(args, class_counts)
    parts = []
    for name in display_class_names(args, class_counts):
        parts.append(f"S-{name}: {float(values.get(name, 0.0)):.1f}")
    return " | ".join(parts) if parts else "S: none"


def _update_cumulative_class_counts(
    detections: sv.Detections,
    args,
    allowed_track_ids: set[int] | None = None,
) -> dict[str, int]:
    """
    Cumulative unique objects by class name over the whole run.
    Uses tracker IDs and counts each track once per class name.
    """
    global SEEN_TRACK_IDS_BY_CLASS

    tracker_ids = detections.tracker_id
    if len(detections) == 0 or tracker_ids is None or detections.class_id is None:
        return {class_name: len(track_ids) for class_name, track_ids in SEEN_TRACK_IDS_BY_CLASS.items()}

    for track_id, cls_id in zip(tracker_ids, detections.class_id):
        track_id = int(track_id)
        if track_id < 0:
            continue
        if allowed_track_ids is not None and track_id not in allowed_track_ids:
            continue

        class_name = class_name_for_id(args, int(cls_id))
        SEEN_TRACK_IDS_BY_CLASS.setdefault(class_name, set()).add(track_id)

    return {class_name: len(track_ids) for class_name, track_ids in SEEN_TRACK_IDS_BY_CLASS.items()}


def reset_object_counter() -> None:
    """
    Reset cumulative counter for a new pipeline run.
    """
    global SEEN_TRACK_IDS_BY_CLASS
    SEEN_TRACK_IDS_BY_CLASS = {}


def tracks_to_sv_detections(tracks: np.ndarray) -> sv.Detections:
    """
    tracks: Nx8 array
    [x1, y1, x2, y2, track_id, score, class_id, extra]
    """
    if tracks is None or len(tracks) == 0:
        return sv.Detections.empty()

    xyxy = tracks[:, 0:4]
    tracker_id = tracks[:, 4].astype(int)
    confidence = tracks[:, 5]
    class_id = tracks[:, 6].astype(int)

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )


def draw_combined_banner(scene: np.ndarray, class_counts: dict[str, int], s_values: dict[str, float], analysis_id: str, args) -> np.ndarray:
    """
    Draws the run metadata banner for all rendered outputs.
    """
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frame_h, frame_w = scene.shape[:2]
    mode_text = _format_banner_mode(args, frame_w, frame_h)
    inference_mode_text = _format_banner_inference_mode(args)
    model_text = _format_banner_model_name(args)
    lines = [
        f"{current_time_str} | {_format_class_counts(args, class_counts)} | {_format_s_values(args, class_counts, s_values)}",
        f"Model: {model_text} | Mode: {mode_text} | Inference: {inference_mode_text}",
        f"{analysis_id}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    bg_alpha = 0.55
    panel_x = 20
    panel_y = 20
    pad_x = 12
    pad_y = 12
    line_gap = 8

    text_metrics = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    text_block_width = max(size[0][0] for size in text_metrics)
    text_block_height = sum(size[0][1] + size[1] for size in text_metrics)
    text_block_height += line_gap * (len(text_metrics) - 1)

    overlay = scene.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + text_block_width + (pad_x * 2), panel_y + text_block_height + (pad_y * 2)),
        bg_color,
        -1,
    )
    cv2.addWeighted(overlay, bg_alpha, scene, 1.0 - bg_alpha, 0.0, scene)

    cursor_y = panel_y + pad_y
    for line, ((_, text_h), baseline) in zip(lines, text_metrics):
        baseline_y = cursor_y + text_h
        cv2.putText(scene, line, (panel_x + pad_x, baseline_y), font, scale, text_color, thickness)
        cursor_y += text_h + baseline + line_gap

    return scene


def visualize_frame_with_supervision(
    frame: np.ndarray,
    tracks: np.ndarray,
    args,
    count_track_ids: set[int] | None = None,
) -> np.ndarray:
    vis = frame.copy()

    # convert tracker output to supervision.Detections
    all_detections = tracks_to_sv_detections(tracks)
    detections = all_detections

    vis_conf = getattr(args, "vis_conf", 0.75)
    vis_strategy = getattr(args, "vis_strategy", "confidence")

    if vis_strategy == "confidence":
        if detections.confidence is not None:
            keep = detections.confidence >= vis_conf
            detections = detections[keep]
    else:
        pass

    if len(detections) > 0:
        vis = BOX_ANNOTATOR.annotate(vis, detections)
        if getattr(args, "captions_enabled", False):
            labels = build_labels_from_tracks(detections, args)
            vis = LABEL_ANNOTATOR.annotate(vis, detections, labels=labels)

    class_counts = _update_cumulative_class_counts(all_detections, args, allowed_track_ids=count_track_ids)
    s_values = class_s_values_from_counts(args, class_counts)
    
    p_id = getattr(args, "pipeline_id", "unknown")
    vis = draw_combined_banner(vis, class_counts, s_values, f"{p_id}", args)

    return vis, class_counts



def build_labels_from_tracks(detections: sv.Detections, args) -> list[str]:
    labels = []
    tracker_ids = detections.tracker_id

    for i in range(len(detections)):
        cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
        name = class_name_for_id(args, cls_id)
        if tracker_ids is not None:
            track_id = int(tracker_ids[i])
            labels.append(f"{name} {conf:.2f} id:{track_id}")
        else:
            labels.append(f"{name} {conf:.2f}")

    return labels
