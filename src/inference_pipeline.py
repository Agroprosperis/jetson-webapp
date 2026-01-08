import argparse
import cv2
import threading
import time
import queue
import logging
import numpy as np
import os
import csv
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
from visualize import visualize_frame_with_supervision, reset_object_counter


LOGGER = logging.getLogger("inference_pipeline")
PROFILER = Profiler()


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
        tracker.gmc = JetsonOptimizedGMC(downscale=scale_factor)
        
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


class JetsonOptimizedGMC(GMC):
    """
    Optimized GMC for Jetson/Microscopy.
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
        self.model = self._load_model(model_path)
        
        args.class_names = self.model.names
        LOGGER.info(f"Model initialized with class names: {self.model.names}")

    def _load_model(self, path: str):
        try:
            return YOLO(model=path, task=None)
        except Exception as e:
            LOGGER.error(f"Failed to load UL model {path}: {e}")
            raise e

    def _detect_objects(self, frame: np.ndarray, args):
        model_conf = getattr(args, "model_conf", 0.1)

        start_time = time.perf_counter()
        results = self.model(frame, conf=model_conf, verbose=False, show=False, save=False, half=True)
        end_time = time.perf_counter()

        if results is None or len(results) == 0:
            h, w = frame.shape[:2]
            empty_boxes = Boxes(torch.zeros((0, 6)), orig_shape=(h, w))
            return [MockResult(empty_boxes)], end_time - start_time 
        
        return results, end_time - start_time


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


def is_jetson():
    try:
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                return "nvidia" in model or "jetson" in model
    except Exception:
        pass
    return False


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


def capture_loop(reader: StreamReader, capture_queue: queue.Queue, stop_event: threading.Event) -> None:
    is_file = isinstance(reader, FileReader)
    target_interval = 1.0 / reader.fps if (reader.fps and reader.fps > 0) else 0.033
    with reader:
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            ret, frame = reader.read()
            read_done = time.perf_counter()

            if not ret or frame is None:
                try:
                    capture_queue.put((None, 0.0, 0.0), timeout=0.1)
                except:
                    pass
                break

            while not stop_event.is_set():
                try:
                    capture_queue.put((frame, read_done, read_done - loop_start), timeout=0.025)
                    break 
                except queue.Full:
                    continue
            
            if is_file:
                process_dur = time.perf_counter() - loop_start
                sleep_time = target_interval - process_dur
                if sleep_time > 0: 
                    time.sleep(sleep_time)


def inference_loop(frame_queue: queue.Queue, result_queue: queue.Queue, stop_event: threading.Event, args) -> None:
    model_manager = ModelManager()
    while not stop_event.is_set():
        frame, capture_start_end, capture_time = frame_queue.get()
        
        if frame is None:
            result_queue.put((None, None))
            break
        
        start_inference = time.perf_counter()
        result, inference_time, tracker_time = model_manager.predict(frame, args)
        result_queue.put((frame, result))
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


def dump_csv_line(csv_writer, frame_count, pipeline_id, total_unique_objects, result) -> None:
    if csv_writer is None:
        return

    detections_serialized = ""
    det_parts = []
    for row in result:
        x0, y0, x1, y1 = row[:4]
        conf_val = float(row[5])
        cls_id = int(row[6])
        det_parts.append(f"x0={int(x0)}_y0={int(y0)}_x1={int(x1)}_y1={int(y1)}_class={cls_id}_conf={conf_val:.3f}")
    detections_serialized = "|".join(det_parts)

    s_value = round((total_unique_objects * 1111.0) / 100.0, 1)
    csv_writer.writerow([frame_count, pipeline_id, s_value, total_unique_objects, detections_serialized])


def output_loop(result_queue: queue.Queue, stop_event: threading.Event, args) -> None:
    output_writer, csv_file, csv_writer = None, None, None
    frame_count = 0
    pipeline_id = getattr(args, "pipeline_id", "unknown")
    hq_output_dir = getattr(args, "hq_output_dir", "/app")
    saved_track_ids = set()
    fps = getattr(args, "fps", 0)
    try:
        while not stop_event.is_set():
            frame, result = result_queue.get()
            if frame is None: 
                break

            start_vis = time.time()
            vis, total_unique_objects = visualize_frame_with_supervision(frame, result, args)
            PROFILER.record('vis', time.time() - start_vis)

            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            vis = np.ascontiguousarray(vis, dtype=np.uint8)
            frame_count += 1
            
            if output_writer is None:
                h, w = frame.shape[:2]
                os.makedirs(hq_output_dir, exist_ok=True)
                hq_path = os.path.join(hq_output_dir, f"{pipeline_id}.mkv")
                out_pipeline = build_rtsp_and_hq_gst(args.stream_host, args.stream_port, args.output_path, w, h, args.fps, hq_path)
                output_writer = cv2.VideoWriter(out_pipeline, cv2.CAP_GSTREAMER, 0, args.fps, (w, h), True)
                hq_csv_path = os.path.join(hq_output_dir, f"{pipeline_id}.csv")
                csv_file = open(hq_csv_path, "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["frame", "analysis_number", "s_value", "total_unique_objects", "detections"])
            
            safe_result = result if result is not None else []
            dump_screenshot(safe_result, vis, hq_output_dir, pipeline_id, saved_track_ids, frame_count, fps)
            dump_csv_line(csv_writer, frame_count, pipeline_id, total_unique_objects, safe_result)
            
            if output_writer is not None:
                output_writer.write(vis)
    finally:
        if output_writer is not None: 
            output_writer.release()

        if csv_file is not None: 
            csv_file.close()


class StreamPipeline:
    def __init__(self, reader: StreamReader, args):
        reset_object_counter()
        self.reader = reader
        self.args = args
        self.frame_queue = queue.Queue(maxsize=80)
        self.result_queue = queue.Queue(maxsize=120)
        self.stop_event = threading.Event()
        self.capture_t = threading.Thread(target=capture_loop, args=(self.reader, self.frame_queue, self.stop_event), daemon=True)
        self.inference_t = threading.Thread(target=inference_loop, args=(self.frame_queue, self.result_queue, self.stop_event, self.args), daemon=True)
        self.output_t = threading.Thread(target=output_loop, args=(self.result_queue, self.stop_event, self.args), daemon=True)
        self._threads = [self.capture_t, self.inference_t, self.output_t]

    def start(self) -> None:
        for t in self._threads: t.start()

    def stop(self) -> None:
        LOGGER.info("Stopping pipeline...")
        self.stop_event.set()

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
            LOGGER.info("Inject stop frame to the input queue")
            self.frame_queue.put((None, 0.0, 0.0), timeout=0.1)
            LOGGER.info("Injected stop frame to the input queue")
        except: 
            pass
        
        try: 
            LOGGER.info("Inject stop frame to the result queue")
            self.result_queue.put((None, None), timeout=0.1)
            LOGGER.info("Injected stop frame to the result queue")
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
