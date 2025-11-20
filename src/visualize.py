import cv2
import numpy as np
import supervision as sv
import logging

from hud import draw_hud, draw_object_count_banner  # <- UPDATED import

BOX_ANNOTATOR = sv.BoxAnnotator()
MASK_ANNOTATOR = sv.PolygonAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(
    text_scale=1.2,      # ← bigger text
    text_thickness=2,    # ← bolder text
    text_padding=8,      # ← more padding around text
)
SEEN_TRACK_IDS: set[int] = set()
CUMULATIVE_OBJECTS: int = 0

LOGGER = logging.getLogger("visualization")


def _update_cumulative_objects(detections: sv.Detections) -> int:
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


def yolo_to_sv_detections(result, vis_conf: float) -> sv.Detections:
    """
    Convert a single Ultralytics YOLO result (with tracking + optional masks)
    into a supervision.Detections object and apply a visualization confidence
    threshold.
    """
    print(result)
    if result is None:
        return sv.Detections.empty()

    # Let supervision handle proper conversion (boxes, masks, tracker IDs, etc.).

    detections = sv.Detections.from_ultralytics(result)

    if len(detections) == 0:
        return detections

    conf = detections.confidence
    if conf is None:
        return detections

    mask = conf >= vis_conf
    if not np.any(mask):
        return sv.Detections.empty()

    return detections[mask]

def visualize_frame_with_supervision(
    frame: np.ndarray,
    tracks: np.ndarray,
    meta: dict | None,
    args,
) -> np.ndarray:
    vis = frame.copy()

    # convert tracker output to supervision.Detections
    detections = tracks_to_sv_detections(tracks)

    vis_conf = getattr(args, "vis_conf", 0.75)
    if detections.confidence is not None:
        keep = detections.confidence >= vis_conf
        detections = detections[keep]

    labels = build_labels_from_tracks(detections, args)

    if len(detections) > 0:
        vis = BOX_ANNOTATOR.annotate(vis, detections)
        vis = LABEL_ANNOTATOR.annotate(vis, detections, labels=labels)

    cumulative_count = _update_cumulative_objects(detections)
    vis = draw_object_count_banner(vis, cumulative_count)
    return vis


def build_labels_from_tracks(detections: sv.Detections, args) -> list[str]:
    names = getattr(args, "class_names", None)  # or args.names, adapt to your config
    labels = []

    for i in range(len(detections)):
        cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0

        name = names[cls_id] if names is not None and cls_id < len(names) else str(cls_id)
        labels.append(f"{name} {conf:.2f}")

    return labels
