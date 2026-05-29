import argparse
import cv2
import threading
import time
import queue
import logging
import numpy as np
import os
import csv
import json
import torch
import tensorrt as trt
import ultralytics.trackers.bot_sort as bot_sort_mod

from datetime import datetime

from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
from ultralytics.trackers.utils.gmc import GMC

from profiler import Profiler
from stream_readers import StreamReader, FileReader
from visualize import (
    class_s_values_from_counts,
    class_name_for_id,
    visualize_frame_with_supervision,
    reset_object_counter,
)
from grid_runtime import GridProcessor, GridOverlay


LOGGER = logging.getLogger("inference_pipeline")
PROFILER = Profiler()
GRID_SCORE_RESET_THRESHOLD = 0.40
GRID_QUEUE_SIZE = 1
GRID_RESULT_HISTORY_SIZE = 128


class CustomBOTrack(BOTrack):
    """
    A custom BoT-SORT Track that enforces a strict 'N-frame' confirmation rule.
    The track is kept in 'New' state (invisible) until it has been matched
    successfully for `n_init` consecutive frames.
    """
    n_init = 5

    def __init__(self, xywh, score, cls, feat=None, feat_history=50):
        super().__init__(xywh, score, cls, feat, feat_history)
        self.hits = 0

    def activate(self, kalman_filter, frame_id):
        """
        Called when a new object is first detected.
        We override this to force the track into a hidden 'New' state initially.
        """
        super().activate(kalman_filter, frame_id)
        self.hits = 1
        
        # Override standard behavior: Force to 'New' state if strictly filtering
        if self.n_init > 1:
            self.is_activated = False
            self.state = TrackState.New
            LOGGER.debug(f"TRACK-DEBUG: ID {self.track_id} initialized. Hits: {self.hits}/{self.n_init}. Status: HIDDEN")

    def update(self, new_track, frame_id):
        """
        Called when an existing track is matched with a new detection.
        We increment hits and only 'Activate' if hits >= n_init.
        """
        # 1. Run standard update (This sets state=Tracked, is_activated=True in parent)
        super().update(new_track, frame_id)
        
        # 2. Increment internal counter
        self.hits += 1
        
        # 3. OVERRIDE State if we haven't met the threshold yet
        if self.hits < self.n_init:
            self.is_activated = False
            self.state = TrackState.New
            LOGGER.debug(f"TRACK-DEBUG: ID {self.track_id} matched. Hits: {self.hits}/{self.n_init}. Status: HIDDEN (Warming up)")
        else:
            # Ensure it is active (Parent likely did this, but we confirm)
            self.is_activated = True
            self.state = TrackState.Tracked
            LOGGER.debug(f"TRACK-DEBUG: ID {self.track_id} matched. Hits: {self.hits}/{self.n_init}. Status: ACTIVE (Confirmed)")


bot_sort_mod.BOTrack = CustomBOTrack
LOGGER.info(f"System: Injected CustomBOTrack with {CustomBOTrack.n_init}-frame verification.")


class InferenceBackend():
    """Base class for inference backends (UL, RF, etc)."""
    def _detect_objects(self, frame: np.ndarray, args):
        raise NotImplementedError()
    
    def _track(self, results: list, frame: np.ndarray, args):
        raise NotImplementedError()

    def predict(self, frame: np.ndarray, args):
        """
        Run inference and tracking on the frame.
        Returns:
            tuple: (tracks, inference_end_time)
        """
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        results, inference_time = self._detect_objects(frame, args)
        results, tracker_time = self._track(results, frame, args)

        return results, inference_time, tracker_time


class BotSortTrackerBackend(InferenceBackend):
    """
    Handles BoT-SORT tracking.
    """
    def __init__(self):
        LOGGER.info(f"Initializing BotSort Tracker")
        self.tracker = None
        self.frame_cnt = 0
        
        # BoT-SORT Configuration
        self.tracker_cfg = {
            "tracker_type": "botsort",
            "track_high_thresh": 0.6,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "match_thresh": 0.99,
            "track_buffer": 200,
            "fuse_score": False,
            "gmc_method": "sparseOptFlow",
            "with_reid": False,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
            "model": "auto"
        }

    def _init_tracker(self, frame_shape, args):
        """Lazy initializer for tracker to ensure we have frame dimensions."""
        vis_conf = getattr(args, "vis_conf", 0.5)
        self.tracker_cfg['new_track_thresh'] = vis_conf
        self.tracker_cfg['track_high_thresh'] = 0.8 * vis_conf

        tracker = BOTSORT(argparse.Namespace(**self.tracker_cfg), frame_rate=args.fps)
        scale_factor = max(max(frame_shape) // 640, 1)
        tracker.gmc = OptimizedGMC(downscale=scale_factor)
        
        LOGGER.info(f'Tracker initialized. Scaling is {scale_factor}')
        return tracker
    
    def dump_debug_frame(self, frame):
        """Dumps debug visualization of the tracker state and GMC flow."""
        if self.tracker is None:
            return

        debug_dir = "/app/debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        vis_frame = frame.copy()

        # --- 1. Visualize GMC Optical Flow (Background Motion) ---
        # We access the GMC instance attached to the tracker
        if hasattr(self.tracker, 'gmc'):
            gmc = self.tracker.gmc
            # Check if features from the current step are exposed
            if hasattr(gmc, 'last_good_old') and hasattr(gmc, 'last_good_new'):
                p0 = gmc.last_good_old
                p1 = gmc.last_good_new
                scale = getattr(gmc, 'downscale', 1)

                if p0 is not None and p1 is not None:
                    for new, old in zip(p1, p0):
                        # Points are stored in downscaled coordinates, scale up for visualization
                        a, b = new.ravel() * scale
                        c, d = old.ravel() * scale
                        
                        # Draw Flow Vector in Cyan (distinct from Green tracks)
                        # Line thickness increased for visibility
                        cv2.line(vis_frame, (int(a), int(b)), (int(c), int(d)), (255, 255, 0), 3) 
                        cv2.circle(vis_frame, (int(a), int(b)), 4, (255, 255, 0), -1)

        # --- 2. Visualize Tracks ---
        def draw_track(t, color, status_label):
            # Using xyxy attribute directly
            x1, y1, x2, y2 = t.xyxy
            tid = t.track_id
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(vis_frame, f"{tid} {status_label}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Active / Warming Up
        for t in self.tracker.tracked_stracks:
            if t.is_activated:
                 draw_track(t, (0, 255, 0), "ACTIVE") # Green
            else:
                 draw_track(t, (0, 255, 255), "WARMUP") # Yellow

        # Lost Tracks
        for t in self.tracker.lost_stracks:
            draw_track(t, (0, 0, 255), "LOST") # Red

        filename = f"tracker_{self.frame_cnt:06d}.jpg"
        cv2.imwrite(os.path.join(debug_dir, filename), vis_frame)

    def _track(self, results: list, frame: np.ndarray, args):
        if self.tracker is None:
            self.tracker = self._init_tracker(frame.shape, args)
        
        self.frame_cnt += 1
        start_time = time.perf_counter()
        detection_boxes = results[0].boxes
        detection_boxes = detection_boxes.cpu().numpy()
        tracks = self.tracker.update(detection_boxes, frame)
        
        if debug:=False:
            self.dump_debug_frame(frame)

        return tracks, time.perf_counter() - start_time


class OptimizedGMC(GMC):
    """
    Optimized GMC for desktop microscopy workloads.
    - Uses sparseOptFlow with increased search depth (maxLevel=6)
    - Fixes coordinate scaling bug
    - Exposes flow points for external debug visualization
    """
    def __init__(self, downscale=2):
        super().__init__(method="sparseOptFlow", downscale=downscale)
        self.frame_cnt = 0 # Explicit frame counter (independent of parent)
        self.last_H = np.eye(2, 3)
        
        # Storage for debug visualization
        self.last_good_old = None
        self.last_good_new = None

        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.02, 
            minDistance=20,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(31, 31), 
            maxLevel=6,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def apply(self, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        self.frame_cnt += 1
        t_start = time.perf_counter()

        # Reset debug points for this frame
        self.last_good_old = None
        self.last_good_new = None

        if self.downscale > 1:
            h, w = raw_frame.shape[:2]
            frame_resized = cv2.resize(raw_frame, (w // self.downscale, h // self.downscale))
        else:
            frame_resized = raw_frame

        curr_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        if self.prevFrame is None:
            self.prevFrame = curr_frame
            self.prevKeyPoints = cv2.goodFeaturesToTrack(self.prevFrame, mask=None, **self.feature_params)
            return np.eye(2, 3)

        next_pts = None
        status = None
        
        if self.prevKeyPoints is not None and len(self.prevKeyPoints) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, curr_frame, self.prevKeyPoints, None, **self.lk_params)
        
        matched_pts = 0
        H = np.eye(2, 3)
        
        if next_pts is not None:
            good_old = self.prevKeyPoints[status == 1]
            good_new = next_pts[status == 1]
            matched_pts = len(good_old)

            # Store for external debug visualization
            self.last_good_old = good_old
            self.last_good_new = good_new

            if matched_pts > 4:
                H, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3)
                if H is None: 
                    H = np.eye(2, 3)

        # Re-detect features for next frame
        self.prevKeyPoints = cv2.goodFeaturesToTrack(curr_frame, mask=None, **self.feature_params)
        self.prevFrame = curr_frame
        
        t_total = time.perf_counter() - t_start
        PROFILER.record('gmc-total', t_total)

        # Scale Back to Full Resolution
        if H is not None and self.downscale > 1:
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale
            
        self.last_H = H
        return H


class UltralyticsBackend(BotSortTrackerBackend):
    def __init__(self, model_path: str, args):
        super().__init__()
        LOGGER.info(f"Initializing Ultralytics Backend with model: {model_path}")
        self.model_path = model_path
        self.model_task = self._normalize_model_task(getattr(args, "model_task", None))
        self.inference_width = max(int(getattr(args, "model_inference_width", 640) or 640), 32)
        self.inference_height = max(int(getattr(args, "model_inference_height", 640) or 640), 32)
        self.model = self._load_model(model_path, self.model_task)
        
        args.class_names = self.model.names
        LOGGER.info(f"Model initialized with class names: {self.model.names}")
        LOGGER.info(
            "Ultralytics runtime settings: size=%dx%d",
            self.inference_width,
            self.inference_height,
        )

    def _normalize_model_task(self, task):
        if task in ("segment", "detect"):
            return task
        return None

    def _load_model(self, path: str, task=None):
        try:
            LOGGER.info(f"Loading Ultralytics model with task: {task or 'auto'}")
            return YOLO(model=path, task=task)
        except Exception as e:
            LOGGER.error(f"Failed to load UL model {path}: {e}")
            raise e

    def _predict_frame_boxes(self, frame: np.ndarray, model_conf: float, *, offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        results = self.model(
            frame,
            conf=model_conf,
            verbose=False,
            show=False,
            save=False,
            half=True,
        )

        if results is None or len(results) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)
        predictions = np.hstack((xyxy, conf, cls)).astype(np.float32, copy=False)
        if predictions.size == 0:
            return predictions
        predictions[:, [0, 2]] += float(offset_x)
        predictions[:, [1, 3]] += float(offset_y)
        return predictions

    def _detect_objects(self, frame: np.ndarray, args):
        model_conf = getattr(args, "model_conf", 0.1)

        start_time = time.perf_counter()
        predictions = self._predict_frame_boxes(frame, model_conf)
        end_time = time.perf_counter()

        h, w = frame.shape[:2]
        if predictions.size == 0:
            empty_boxes = Boxes(torch.zeros((0, 6)), orig_shape=(h, w))
            return [MockResult(empty_boxes)], end_time - start_time

        boxes_obj = Boxes(torch.from_numpy(predictions), orig_shape=(h, w))
        return [MockResult(boxes_obj)], end_time - start_time


class MockResult:
    def __init__(self, boxes):
        self.boxes = boxes


class RoboflowBackend(BotSortTrackerBackend):
    def __init__(self, model_path: str, args):
        super().__init__()
        LOGGER.info(f"Initializing TensorRT Backend for: {model_path}")
        self.model_path = model_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        try:
            with open(model_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            LOGGER.error(f"Failed to load engine {model_path}: {e}")
            raise e
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = [None] * self.engine.num_io_tensors
        self.input_shape = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = self.engine.get_tensor_shape(name)
            dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            if -1 in shape: shape = (1, 3, 640, 640)
            tensor = torch.zeros(tuple(shape), dtype=dtype, device='cuda').contiguous()
            binding = {'index': i, 'tensor': tensor, 'ptr': tensor.data_ptr(), 'shape': shape}
            self.bindings[i] = binding['ptr']
            if is_input: 
                self.inputs.append(binding)
                self.input_shape = shape
            else: 
                self.outputs.append(binding)
        self.h, self.w = (self.input_shape[2], self.input_shape[3]) if self.input_shape else (640, 640)

    def _trt_to_torch_dtype(self, trt_dtype):
        return {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32}.get(trt_dtype, torch.float32)

    def _preprocess(self, image):
        img = cv2.resize(image, (self.w, self.h))
        tensor = torch.from_numpy(img).cuda().permute(2, 0, 1).float()
        tensor = tensor[[2, 1, 0], :, :] / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0)

    def _postprocess_gpu(self, outputs, orig_w, orig_h, conf_thresh=0.5):
        logits, boxes = None, None
        for out in outputs:
            t = out['tensor']
            if t.shape[-1] == 4: boxes = t
            elif len(t.shape) >= 3 and t.shape[-2] == 4: boxes = t.transpose(1, 2)
            elif len(t.shape) == 2 and t.shape[0] == 4: boxes = t.t().unsqueeze(0)
            else: logits = t
        if logits is None or boxes is None: return None
        num_queries = boxes.shape[1]
        if logits.shape[-1] == num_queries and logits.shape[-2] != num_queries: logits = logits.transpose(1, 2)
        scores = torch.sigmoid(logits.clamp(-100, 100))
        max_scores, class_ids = torch.max(scores[0], dim=-1)
        mask = max_scores > conf_thresh
        if not mask.any(): return None
        filt_scores = max_scores[mask]
        filt_classes = class_ids[mask]
        filt_boxes = boxes[0][mask]
        cx, cy, w, h = filt_boxes.unbind(-1)
        x1 = (cx - 0.5 * w) * orig_w
        y1 = (cy - 0.5 * h) * orig_h
        x2 = (cx + 0.5 * w) * orig_w
        y2 = (cy + 0.5 * h) * orig_h
        xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        return xyxy.cpu().numpy(), filt_scores.cpu().numpy(), filt_classes.cpu().numpy()
    
    def _detect_objects(self, frame: np.ndarray, args):
        start_time = time.perf_counter()
        input_tensor = self._preprocess(frame)
        self.inputs[0]['tensor'].copy_(input_tensor)
        self.context.execute_v2(self.bindings)
        conf_thresh = getattr(args, "model_conf", 0.5)
        orig_h, orig_w = frame.shape[:2]
        try:
            res = self._postprocess_gpu(self.outputs, orig_w, orig_h, conf_thresh)
        except Exception as e:
            LOGGER.error(f"Post-process error: {e}")
            empty_boxes = Boxes(torch.zeros((0, 6)), orig_shape=(orig_h, orig_w))
            return [MockResult(empty_boxes)], time.perf_counter() - start_time
        if res is None:
            empty_boxes = Boxes(torch.zeros((0, 6)), orig_shape=(orig_h, orig_w))
            return [MockResult(empty_boxes)], time.perf_counter() - start_time
        xyxy, confs, classes = res
        confs = confs.reshape(-1, 1)
        classes = classes.reshape(-1, 1)
        preds = np.hstack((xyxy, confs, classes))
        torch_preds = torch.from_numpy(preds)
        boxes_obj = Boxes(torch_preds, orig_shape=(orig_h, orig_w))
        return [MockResult(boxes_obj)], time.perf_counter() - start_time


class ModelManager:
    def __init__(self):
        self.current_backend = None
        self.current_model_path = None

    def _get_backend_type(self, path: str):
        if os.path.isfile(f"{path}.json"):
            from backends.roboflow_package_backend import RoboflowPackageBackend

            return RoboflowPackageBackend
        if "/rf/" in path or "rfdetr" in path.lower():
            return RoboflowBackend
        return UltralyticsBackend

    def predict(self, frame: np.ndarray, args):
        requested_path = getattr(args, "model_path", "/app/model/weights-fp16.engine")
        if self.current_backend is None or self.current_model_path != requested_path:
            backend_cls = self._get_backend_type(requested_path)
            self.current_backend = None 
            torch.cuda.empty_cache()

            self.current_backend = backend_cls(requested_path, args)
            self.current_model_path = requested_path

        return self.current_backend.predict(frame, args)

def build_rtsp_and_hq_gst(host: str, port: int, path: str, width: int, height: int, fps: int, hq_path: str) -> str:
    bitrate_kbit = 8000
    rtsp_url = f"rtsp://{host}:{port}/{path}"

    pipeline = (
        "appsrc is-live=true block=true format=time do-timestamp=true "
        "max-bytes=100000000 ! " 
        "queue max-size-buffers=5 leaky=no ! "
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=I420 ! "
        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbit} "
        "key-int-max=30 bframes=0 sliced-threads=true threads=4 ! "
        "h264parse config-interval=-1 ! "
        "tee name=t "
        "t. ! queue max-size-buffers=10 leaky=no ! "
        "matroskamux ! "
        f"filesink location=\"{hq_path}\" sync=false "
        "t. ! queue max-size-buffers=200 leaky=no ! "
        f"rtspclientsink location={rtsp_url} protocols=tcp "
    )
    LOGGER.info("Pipeline: %s", pipeline)
    return pipeline


def build_hq_file_gst(width: int, height: int, fps: int, hq_path: str) -> str:
    bitrate_kbit = 16000
    pipeline = (
        "appsrc is-live=true block=true format=time do-timestamp=true "
        "max-bytes=100000000 ! "
        "queue max-size-buffers=5 leaky=no ! "
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=I420 ! "
        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbit} "
        "key-int-max=30 bframes=0 sliced-threads=true threads=4 ! "
        "h264parse config-interval=-1 ! "
        "matroskamux ! "
        f"filesink location=\"{hq_path}\" sync=false "
    )
    LOGGER.info("File pipeline: %s", pipeline)
    return pipeline


class GridResultBuffer:
    def __init__(self, max_items: int = GRID_RESULT_HISTORY_SIZE) -> None:
        self._max_items = max(int(max_items), 1)
        self._lock = threading.Lock()
        self._items: list[tuple[int, GridOverlay]] = []

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def update(self, frame_id: int, overlay: GridOverlay | None) -> None:
        if overlay is None:
            return

        with self._lock:
            if self._items and frame_id <= self._items[-1][0]:
                self._items = [
                    (stored_frame_id, stored_overlay)
                    for stored_frame_id, stored_overlay in self._items
                    if stored_frame_id < frame_id
                ]
            self._items.append((int(frame_id), overlay))
            if len(self._items) > self._max_items:
                self._items = self._items[-self._max_items :]

    def get_for_frame(self, frame_id: int) -> GridOverlay | None:
        with self._lock:
            for stored_frame_id, overlay in reversed(self._items):
                if stored_frame_id <= frame_id:
                    return overlay
        return None


def _enqueue_latest(queue_obj: queue.Queue, item) -> bool:
    try:
        queue_obj.put_nowait(item)
        return False
    except queue.Full:
        try:
            queue_obj.get_nowait()
        except queue.Empty:
            return False
        try:
            queue_obj.put_nowait(item)
            return True
        except queue.Full:
            return False


def capture_loop(reader: StreamReader, capture_queue: queue.Queue, stop_event: threading.Event) -> None:
    is_file = isinstance(reader, FileReader)
    target_interval = 1.0 / reader.fps if (reader.fps and reader.fps > 0) else 0.033
    frame_id = 0
    with reader:
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            ret, frame = reader.read()
            read_done = time.perf_counter()

            if not ret or frame is None:
                try:
                    capture_queue.put((None, None, 0.0, 0.0), timeout=0.1)
                except:
                    pass
                break

            frame_id += 1
            while not stop_event.is_set():
                try:
                    capture_queue.put((frame_id, frame, read_done, read_done - loop_start), timeout=0.025)
                    break 
                except queue.Full:
                    continue
            
            if is_file:
                process_dur = time.perf_counter() - loop_start
                sleep_time = target_interval - process_dur
                if sleep_time > 0: 
                    time.sleep(sleep_time)


def inference_loop(
    frame_queue: queue.Queue,
    result_queue: queue.Queue,
    grid_queue: queue.Queue,
    stop_event: threading.Event,
    args,
) -> None:
    model_manager = ModelManager()
    while not stop_event.is_set():
        frame_id, frame, capture_start_end, capture_time = frame_queue.get()
        
        if frame is None:
            result_queue.put((None, None, None))
            _enqueue_latest(grid_queue, (None, None))
            break
        
        start_inference = time.perf_counter()
        result, inference_time, tracker_time = model_manager.predict(frame, args)
        if _grid_count_enabled(args):
            _enqueue_latest(grid_queue, (frame_id, frame))
        result_queue.put((frame_id, frame, result))
        end_inference = time.perf_counter()

        PROFILER.record("capture", capture_time).record("inf", inference_time).record("track", tracker_time)\
            .record("latency", time.perf_counter() - capture_start_end).record("processing", end_inference - start_inference)\
            .record("capture_queue", frame_queue.qsize()).record("output_queue", result_queue.qsize())


def dump_screenshot(result, vis, hq_output_dir, pipeline_id, saved_track_ids, frame_count, fps) -> None:
    if fps and fps > 0:
        frame_ts_ms = int(round((frame_count - 1) * 1000.0 / fps))
    else:
        frame_ts_ms = 0

    for row in result:
        track_id = int(row[4])
        if track_id < 0 or track_id in saved_track_ids:
            continue
        
        cv2.imwrite(
            os.path.join(
                hq_output_dir,
                f"{pipeline_id}-fn-{frame_count:06d}-ms-{frame_ts_ms:06d}-oid-{track_id:06d}.jpg",
            ),
            vis,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )
        saved_track_ids.add(track_id)


def dump_manual_snapshot(frame, run_dir, pipeline_id, frame_count, fps, request_id) -> str:
    if fps and fps > 0:
        frame_ts_ms = int(round((frame_count - 1) * 1000.0 / fps))
    else:
        frame_ts_ms = 0

    filename = (
        f"{pipeline_id}-snapshot-rq-{request_id:06d}-fn-{frame_count:06d}-"
        f"ms-{frame_ts_ms:06d}.jpg"
    )
    saved = cv2.imwrite(
        os.path.join(run_dir, filename),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 100],
    )
    if not saved:
        raise RuntimeError("Failed to write manual snapshot.")
    return filename


def dump_csv_line(csv_writer, frame_count, pipeline_id, class_counts, result, args) -> None:
    if csv_writer is None:
        return

    detections = []
    for row in result:
        x0, y0, x1, y1 = row[:4]
        conf_val = float(row[5])
        cls_id = int(row[6])
        detections.append(
            {
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "class_id": cls_id,
                "class_name": class_name_for_id(args, cls_id),
                "confidence": round(conf_val, 3),
            }
        )

    s_values = class_s_values_from_counts(args, class_counts)
    row = {
        "frame": frame_count,
        "analysis_number": pipeline_id,
        "class_counts": json.dumps(class_counts, ensure_ascii=False, sort_keys=True),
        "s_values": json.dumps(s_values, ensure_ascii=False, sort_keys=True),
        "detections": json.dumps(detections, ensure_ascii=False),
    }
    csv_writer.writerow(row)


def _grid_count_enabled(args) -> bool:
    getter = getattr(args, "grid_count_enabled_getter", None)
    if callable(getter):
        try:
            return bool(getter())
        except Exception:
            LOGGER.exception("Failed to read grid counting toggle")
    return bool(getattr(args, "grid_count_enabled", False))


def _grid_score_threshold(args) -> float:
    getter = getattr(args, "grid_score_threshold_getter", None)
    if callable(getter):
        try:
            return max(float(getter()), 0.0)
        except Exception:
            LOGGER.exception("Failed to read grid score threshold")
    return max(float(getattr(args, "grid_score_threshold", 0.0)), 0.0)


def _grid_debug_enabled(args) -> bool:
    getter = getattr(args, "grid_debug_enabled_getter", None)
    if callable(getter):
        try:
            return bool(getter())
        except Exception:
            LOGGER.exception("Failed to read grid debug toggle")
    return bool(getattr(args, "grid_debug_enabled", False))


def _set_grid_count_enabled(args, enabled: bool, *, auto: bool = False) -> bool:
    setter = getattr(args, "grid_count_enabled_setter", None)
    if callable(setter):
        try:
            return bool(setter(enabled, auto=auto))
        except TypeError:
            return bool(setter(enabled))
        except Exception:
            LOGGER.exception("Failed to update grid counting toggle")
    return bool(enabled)


def _publish_grid_score(args, score: float | None) -> None:
    setter = getattr(args, "grid_score_setter", None)
    if callable(setter):
        try:
            setter(score)
        except Exception:
            LOGGER.exception("Failed to publish grid score")


def _viewport_polygon(grid_overlay) -> np.ndarray | None:
    corners = getattr(grid_overlay, "viewport_corners", None)
    if not corners or len(corners) != 4:
        return None
    return np.asarray(corners, dtype=np.float32)


def _track_ids_inside_viewport(result, grid_overlay) -> set[int]:
    if result is None or len(result) == 0:
        return set()

    viewport = _viewport_polygon(grid_overlay)
    if viewport is None:
        return set()

    viewport_cv = viewport.reshape((-1, 1, 2))
    inside_ids: set[int] = set()
    for row in result:
        track_id = int(row[4])
        if track_id < 0:
            continue

        x0, y0, x1, y1 = [float(value) for value in row[:4]]
        if x1 <= x0 or y1 <= y0:
            continue

        box_poly = np.asarray(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=np.float32,
        ).reshape((-1, 1, 2))

        try:
            overlap_area, _ = cv2.intersectConvexConvex(viewport_cv, box_poly)
            if overlap_area > 0.0:
                inside_ids.add(track_id)
                continue
        except cv2.error:
            pass

        center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
        if cv2.pointPolygonTest(viewport_cv, center, False) >= 0:
            inside_ids.add(track_id)

    return inside_ids


def _filter_tracks_by_ids(result, allowed_track_ids: set[int]):
    if result is None:
        return []
    if len(result) == 0 or not allowed_track_ids:
        return result[:0] if isinstance(result, np.ndarray) else []

    if isinstance(result, np.ndarray):
        mask = np.asarray([int(row[4]) in allowed_track_ids for row in result], dtype=bool)
        return result[mask]

    return [row for row in result if int(row[4]) in allowed_track_ids]


def grid_loop(grid_queue: queue.Queue, grid_results: GridResultBuffer, stop_event: threading.Event, args) -> None:
    grid_processor = GridProcessor(cluster_only_mode=True)
    grid_processing_active = False
    grid_reset_armed = True

    while not stop_event.is_set():
        frame_id, frame = grid_queue.get()
        if frame is None:
            grid_results.clear()
            _publish_grid_score(args, None)
            break

        latest_frame_id, latest_frame = frame_id, frame
        while True:
            try:
                candidate_frame_id, candidate_frame = grid_queue.get_nowait()
            except queue.Empty:
                break
            if candidate_frame is None:
                grid_results.clear()
                _publish_grid_score(args, None)
                return
            latest_frame_id, latest_frame = candidate_frame_id, candidate_frame

        grid_count_enabled = _grid_count_enabled(args)
        grid_score_threshold = _grid_score_threshold(args)
        grid_debug_enabled = _grid_debug_enabled(args)
        if not grid_count_enabled:
            if grid_processing_active:
                grid_results.clear()
                _publish_grid_score(args, None)
            grid_processing_active = False
            grid_reset_armed = True
            continue

        if not grid_processing_active:
            grid_processor = GridProcessor(cluster_only_mode=True)
            grid_results.clear()
            grid_reset_armed = True

        grid_processing_active = True
        grid_overlay = grid_processor.process(latest_frame, include_debug=grid_debug_enabled)
        if grid_overlay is None:
            grid_results.clear()
            _publish_grid_score(args, None)
            grid_processing_active = False
            continue
        grid_score = getattr(grid_overlay, "score", None) if grid_overlay is not None else None
        grid_score_raw = getattr(grid_overlay, "score_raw", None) if grid_overlay is not None else None

        if grid_score is not None:
            _publish_grid_score(args, grid_score)
        if grid_score_raw is not None:
            if grid_score_raw > GRID_SCORE_RESET_THRESHOLD:
                grid_reset_armed = True
            elif grid_reset_armed:
                grid_processor.reset_state()
                grid_results.clear()
                grid_reset_armed = False
        if (
            grid_score is not None
            and grid_score_threshold > 0.0
            and grid_score <= grid_score_threshold
        ):
            _set_grid_count_enabled(args, False, auto=True)
            grid_results.clear()
            _publish_grid_score(args, None)
            grid_processing_active = False
            grid_reset_armed = True
            continue

        grid_results.update(latest_frame_id, grid_overlay)


def output_loop(
    result_queue: queue.Queue,
    grid_results: GridResultBuffer,
    stop_event: threading.Event,
    args,
) -> None:
    output_writer, raw_output_writer, csv_file, csv_writer = None, None, None, None
    frame_count = 0
    pipeline_id = getattr(args, "pipeline_id", "unknown")
    hq_output_dir = getattr(args, "hq_output_dir", "/app")
    run_dir = None
    saved_track_ids = set()
    qualified_track_ids: set[int] = set()
    fps = getattr(args, "fps", 0)
    grid_renderer = GridProcessor()
    try:
        while not stop_event.is_set():
            frame_id, frame, result = result_queue.get()
            if frame is None: 
                break

            grid_count_enabled = _grid_count_enabled(args)
            grid_debug_enabled = _grid_debug_enabled(args)
            grid_overlay = grid_results.get_for_frame(frame_id) if grid_count_enabled else None

            if (
                grid_count_enabled
                and grid_overlay is not None
                and _viewport_polygon(grid_overlay) is not None
            ):
                qualified_track_ids.update(_track_ids_inside_viewport(result, grid_overlay))
                count_track_ids: set[int] | None = qualified_track_ids
            else:
                count_track_ids = None

            start_vis = time.time()
            vis, class_counts = visualize_frame_with_supervision(
                frame,
                result,
                args,
                count_track_ids=count_track_ids,
            )
            if grid_count_enabled and grid_overlay is not None:
                vis = grid_renderer.render(vis, grid_overlay, debug=grid_debug_enabled)
            PROFILER.record('vis', time.time() - start_vis)

            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            vis = np.ascontiguousarray(vis, dtype=np.uint8)
            frame_count += 1

            if run_dir is None:
                run_dir = os.path.join(hq_output_dir, pipeline_id)
                os.makedirs(run_dir, exist_ok=True)

            if output_writer is None:
                h, w = frame.shape[:2]
                hq_path = os.path.join(run_dir, f"{pipeline_id}.mkv")
                out_pipeline = build_rtsp_and_hq_gst(args.stream_host, args.stream_port, args.output_path, w, h, args.fps, hq_path)
                output_writer = cv2.VideoWriter(out_pipeline, cv2.CAP_GSTREAMER, 0, args.fps, (w, h), True)
                raw_hq_path = os.path.join(run_dir, f"{pipeline_id}-raw.mkv")
                raw_out_pipeline = build_hq_file_gst(w, h, args.fps, raw_hq_path)
                raw_output_writer = cv2.VideoWriter(raw_out_pipeline, cv2.CAP_GSTREAMER, 0, args.fps, (w, h), True)
                hq_csv_path = os.path.join(run_dir, f"{pipeline_id}.csv")
                csv_file = open(hq_csv_path, "w", newline="", encoding="utf-8")
                csv_writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "frame",
                        "analysis_number",
                        "class_counts",
                        "s_values",
                        "detections",
                    ],
                )
                csv_writer.writeheader()
            
            safe_result = result if result is not None else []
            if (
                grid_count_enabled
                and grid_overlay is not None
                and _viewport_polygon(grid_overlay) is not None
            ):
                reactive_result = _filter_tracks_by_ids(safe_result, qualified_track_ids)
            else:
                reactive_result = safe_result
            dump_screenshot(
                reactive_result,
                vis,
                run_dir or hq_output_dir,
                pipeline_id,
                saved_track_ids,
                frame_count,
                fps,
            )
            snapshot_controller = getattr(args, "snapshot_controller", None)
            if snapshot_controller is not None:
                for request_id in snapshot_controller.consume_snapshot_requests():
                    try:
                        filename = dump_manual_snapshot(
                            frame,
                            run_dir,
                            pipeline_id,
                            frame_count,
                            fps,
                            request_id,
                        )
                        snapshot_controller.complete_snapshot_request(
                            request_id,
                            {
                                "filename": filename,
                                "path": f"{pipeline_id}/{filename}",
                            },
                        )
                    except Exception as exc:
                        snapshot_controller.fail_snapshot_request(request_id, str(exc))
            dump_csv_line(csv_writer, frame_count, pipeline_id, class_counts, reactive_result, args)
            
            if output_writer is not None:
                output_writer.write(vis)
            if raw_output_writer is not None:
                raw_output_writer.write(frame)
    finally:
        if output_writer is not None: 
            output_writer.release()
        if raw_output_writer is not None:
            raw_output_writer.release()

        if csv_file is not None: 
            csv_file.close()


class StreamPipeline:
    def __init__(self, reader: StreamReader, args):
        reset_object_counter()
        self.reader = reader
        self.args = args
        self.frame_queue = queue.Queue(maxsize=80)
        self.result_queue = queue.Queue(maxsize=120)
        self.grid_queue = queue.Queue(maxsize=GRID_QUEUE_SIZE)
        self.grid_results = GridResultBuffer()
        self.stop_event = threading.Event()
        self._snapshot_condition = threading.Condition()
        self._snapshot_pending_ids: list[int] = []
        self._snapshot_results: dict[int, dict[str, str]] = {}
        self._next_snapshot_request_id = 0
        self.args.snapshot_controller = self
        self.capture_t = threading.Thread(
            target=capture_loop,
            args=(self.reader, self.frame_queue, self.stop_event),
            daemon=True,
        )
        self.inference_t = threading.Thread(
            target=inference_loop,
            args=(self.frame_queue, self.result_queue, self.grid_queue, self.stop_event, self.args),
            daemon=True,
        )
        self.grid_t = threading.Thread(
            target=grid_loop,
            args=(self.grid_queue, self.grid_results, self.stop_event, self.args),
            daemon=True,
        )
        self.output_t = threading.Thread(
            target=output_loop,
            args=(self.result_queue, self.grid_results, self.stop_event, self.args),
            daemon=True,
        )
        self._threads = [self.capture_t, self.inference_t, self.grid_t, self.output_t]

    def consume_snapshot_requests(self) -> list[int]:
        with self._snapshot_condition:
            pending = list(self._snapshot_pending_ids)
            self._snapshot_pending_ids.clear()
            return pending

    def complete_snapshot_request(self, request_id: int, result: dict[str, str]) -> None:
        with self._snapshot_condition:
            self._snapshot_results[request_id] = result
            self._snapshot_condition.notify_all()

    def fail_snapshot_request(self, request_id: int, error_message: str) -> None:
        self.complete_snapshot_request(request_id, {"error": error_message})

    def request_snapshot(self, timeout: float = 5.0) -> dict[str, str]:
        with self._snapshot_condition:
            self._next_snapshot_request_id += 1
            request_id = self._next_snapshot_request_id
            self._snapshot_pending_ids.append(request_id)
            deadline = time.monotonic() + timeout

            while request_id not in self._snapshot_results:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if request_id in self._snapshot_pending_ids:
                        self._snapshot_pending_ids.remove(request_id)
                    raise TimeoutError("Timed out waiting for the next frame.")
                self._snapshot_condition.wait(timeout=remaining)

            result = self._snapshot_results.pop(request_id)

        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def start(self) -> None:
        for t in self._threads: t.start()

    def stop(self) -> None:
        LOGGER.info("Stopping pipeline...")
        self.stop_event.set()
        with self._snapshot_condition:
            pending = list(self._snapshot_pending_ids)
            self._snapshot_pending_ids.clear()
            for request_id in pending:
                self._snapshot_results[request_id] = {"error": "Pipeline stopped before snapshot was captured."}
            if pending:
                self._snapshot_condition.notify_all()

        try:
            LOGGER.info("Clearing input queue")
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            LOGGER.info("Input queue cleared")
        except Exception:
            pass

        try:
            LOGGER.info("Clearing result queue")
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
            LOGGER.info("Result queue cleared")
        except Exception:
            pass

        try:
            LOGGER.info("Clearing grid queue")
            while not self.grid_queue.empty():
                self.grid_queue.get_nowait()
            self.grid_results.clear()
            LOGGER.info("Grid queue cleared")
        except Exception:
            pass

        try:
            LOGGER.info("Inject stop frame to the input queue")
            self.frame_queue.put((None, None, 0.0, 0.0), timeout=0.1)
            LOGGER.info("Injected stop frame to the input queue")
        except: 
            pass
        
        try: 
            LOGGER.info("Inject stop frame to the result queue")
            self.result_queue.put((None, None, None), timeout=0.1)
            LOGGER.info("Injected stop frame to the result queue")
        except: 
            pass

        try:
            LOGGER.info("Inject stop frame to the grid queue")
            _enqueue_latest(self.grid_queue, (None, None))
            LOGGER.info("Injected stop frame to the grid queue")
        except:
            pass

        LOGGER.info("Waiting threads for termination")
        for t in self._threads: 
            t.join(timeout=2.0)
            if t.is_alive():
                LOGGER.warning(f"Thread {t.name} did not exit cleanly (Zombie thread).")

        LOGGER.info("check if cap is closed")
        if hasattr(self.reader, 'cap') and self.reader.cap is not None:
            if self.reader.cap.isOpened():
                LOGGER.warning("Force releasing camera resource in stop()")
                self.reader.cap.release()

    def run(self) -> None:
        try:
            while self.output_t.is_alive(): time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        LOGGER.warning("Triggered for stop event")
        self.stop()
        return False
