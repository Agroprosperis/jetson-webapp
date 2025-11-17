import cv2
import numpy as np
import supervision as sv
import logging

from hud import draw_hud, draw_object_count_banner  # <- UPDATED import

BOX_ANNOTATOR = sv.BoxAnnotator()
MASK_ANNOTATOR = sv.PolygonAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
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


def yolo_to_sv_detections(result, vis_conf: float) -> sv.Detections:
    """
    Convert a single Ultralytics YOLO result (with tracking + optional masks)
    into a supervision.Detections object and apply a visualization confidence
    threshold.
    """
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
    result,
    meta: dict | None,
    args,
) -> np.ndarray:
    vis = frame.copy()

    vis_conf = getattr(args, "vis_conf", 0.75)
    detections = yolo_to_sv_detections(result, vis_conf)
    labels = build_labels(result, detections)

    if len(detections) > 0:
        # Order: masks → boxes → labels
        vis = MASK_ANNOTATOR.annotate(vis, detections)
        vis = BOX_ANNOTATOR.annotate(vis, detections)
        vis = LABEL_ANNOTATOR.annotate(vis, detections, labels=labels)

    # Unique object counter banner (top-left)
    cumulative_count = _update_cumulative_objects(detections)
    vis = draw_object_count_banner(vis, cumulative_count)
    
    cv2.imwrite('/app/debug_vis.jpeg', vis)

    if debug:=False and meta is not None:
        vis = draw_hud(
            vis,
            send_ts=meta.get("send_ts"),
            recv_ts=None,
            latency_ms=meta.get("latency_dt", 0.0) * 1000.0,
            fps=meta.get("inst_fps", 0.0),
        )
    return vis


def build_labels(result, detections: sv.Detections):
    if result is None or len(detections) == 0:
        return []

    names = result.names
    labels = []
    tracker_ids = getattr(detections, "tracker_id", None)

    for i in range(len(detections)):
        cls_id = int(detections.class_id[i])
        conf = float(detections.confidence[i])
        name = names.get(cls_id, str(cls_id))

        if tracker_ids is not None:
            tid = tracker_ids[i]
        else:
            tid = None

        if tid is not None:
            labels.append(f"#{int(tid)} {name} {conf:.2f}")
        else:
            labels.append(f"{name} {conf:.2f}")

    return labels
