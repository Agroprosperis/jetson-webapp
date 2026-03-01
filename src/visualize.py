import numpy as np
import supervision as sv
import logging
import cv2
from datetime import datetime

BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(
    text_scale = 1.2,
    text_thickness = 2,
    text_padding = 8,
)

SEEN_TRACK_IDS: set[int] = set()
CUMULATIVE_OBJECTS: int = 0

LOGGER = logging.getLogger("visualization")


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


def draw_combined_banner(scene: np.ndarray, count: int, s_value: float, analysis_id: str) -> np.ndarray:
    """
    Draws the total object count, the calculated S-metric, and the Analysis ID (Pipeline ID).
    """
    # 1. Get current local time in human readable format
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. Define Text (Added Time at the start, rounded S to 1 decimal)
    text = f"{current_time_str} | Objects: {count} | S: {s_value:.1f} | {analysis_id}"
    
    # 3. visual settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    text_color = (255, 255, 255) # White
    bg_color = (0, 0, 0)         # Black
    
    # 4. Calculate size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # 5. Define positions (Top-Left corner with padding)
    x, y = 30, 60
    pad = 10
    
    # 6. Draw Background Rectangle
    cv2.rectangle(
        scene,
        (x - pad, y - text_h - pad),
        (x + text_w + pad, y + pad),
        bg_color,
        -1 # Filled
    )
    
    # 7. Draw Text
    cv2.putText(scene, text, (x, y), font, scale, text_color, thickness)
    
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
    s_metric = round((cumulative_count * 1111. / 100.), 1)
    
    p_id = getattr(args, "pipeline_id", "unknown")
    vis = draw_combined_banner(vis, cumulative_count, s_metric, f"{p_id}")

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
