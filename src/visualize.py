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

SEEN_TRACK_IDS: set[int] = set()
CUMULATIVE_OBJECTS: int = 0

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


def _update_cumulative_objects(
    detections: sv.Detections,
    allowed_track_ids: set[int] | None = None,
) -> int:
    """
    Cumulative unique objects over the whole run.
    Uses tracker IDs if available (preferred), otherwise falls back to
    summing detections across frames.
    """
    global SEEN_TRACK_IDS, CUMULATIVE_OBJECTS

    if len(detections) == 0:
        return CUMULATIVE_OBJECTS

    tracker_ids = detections.tracker_id

    # Preferred path: tracker IDs present
    if tracker_ids is not None:
        ids = np.asarray(tracker_ids)
        ids = ids[ids >= 0]
        if allowed_track_ids is not None:
            ids = np.asarray([track_id for track_id in ids.tolist() if int(track_id) in allowed_track_ids])
        if ids.size == 0:
            return CUMULATIVE_OBJECTS

        new_ids = set(ids.tolist()) - SEEN_TRACK_IDS
        if new_ids:
            SEEN_TRACK_IDS.update(new_ids)
            CUMULATIVE_OBJECTS = len(SEEN_TRACK_IDS)
        return CUMULATIVE_OBJECTS

    # Fallback: no tracker IDs – just accumulate per-frame detections
    return CUMULATIVE_OBJECTS


def reset_object_counter() -> None:
    """
    Reset cumulative counter for a new pipeline run.
    """
    global SEEN_TRACK_IDS, CUMULATIVE_OBJECTS
    SEEN_TRACK_IDS = set()
    CUMULATIVE_OBJECTS = 0


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


def draw_combined_banner(scene: np.ndarray, count: int, s_value: float, analysis_id: str, args) -> np.ndarray:
    """
    Draws the run metadata banner for all rendered outputs.
    """
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frame_h, frame_w = scene.shape[:2]
    mode_text = _format_banner_mode(args, frame_w, frame_h)
    model_text = _format_banner_model_name(args)
    lines = [
        f"{current_time_str} | Objects: {count} | S: {s_value:.1f}",
        f"Model: {model_text} | Mode: {mode_text}",
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

    labels = build_labels_from_tracks(detections, args)

    if len(detections) > 0:
        vis = BOX_ANNOTATOR.annotate(vis, detections)
        vis = LABEL_ANNOTATOR.annotate(vis, detections, labels=labels)

    cumulative_count = _update_cumulative_objects(all_detections, allowed_track_ids=count_track_ids)
    s_metric = calculate_s_value(cumulative_count)
    
    p_id = getattr(args, "pipeline_id", "unknown")
    vis = draw_combined_banner(vis, cumulative_count, s_metric, f"{p_id}", args)

    return vis, cumulative_count



def build_labels_from_tracks(detections: sv.Detections, args) -> list[str]:
    names = getattr(args, "class_names", None)
    labels = []
    tracker_ids = detections.tracker_id

    for i in range(len(detections)):
        cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
        name = names[cls_id] if names is not None and cls_id < len(names) else str(cls_id)
        if tracker_ids is not None:
            track_id = int(tracker_ids[i])
            labels.append(f"{name} {conf:.2f} id:{track_id}")
        else:
            labels.append(f"{name} {conf:.2f}")

    return labels
