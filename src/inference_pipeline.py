import argparse
import cv2
import threading
import time
import queue
import logging
import numpy as np
import os
import csv
from datetime import datetime
from abc import ABC, abstractmethod

from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.gmc import GMC

from profiler import Profiler
from stream_readers import StreamReader, FileReader
from visualize import visualize_frame_with_supervision, reset_object_counter

LOGGER = logging.getLogger("inference_pipeline")


class InferenceBackend(ABC):
    """Abstract base class for inference backends (UL, RF, etc)."""
    
    @abstractmethod
    def predict(self, frame: np.ndarray, args):
        """
        Run inference and tracking on the frame.
        Returns:
            tuple: (tracks, inference_end_time)
        """
        pass


class GMCOnYolo(GMC):
    """Helper class for Ultralytics BoT-SORT GMC."""
    def __init__(self, method: str = "sparseOptFlow", downscale: int = 1):
        super().__init__(method=method, downscale=downscale)

    def apply(self, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        return super().apply(raw_frame, detections)


class UltralyticsBackend(InferenceBackend):
    """
    Handles Ultralytics YOLO model loading and BoT-SORT tracking.
    """
    def __init__(self, model_path: str, args):
        LOGGER.info(f"Initializing Ultralytics Backend with model: {model_path}")
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.tracker = None
        
        # BoT-SORT Configuration
        self.tracker_cfg = {
            "tracker_type": "botsort",
            "track_high_thresh": 0.75,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.75,
            "match_thresh": 0.90,
            "track_buffer": 90,
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "with_reid": False,
            "proximity_thresh": 0.1,
            "appearance_thresh": 0.15,
            "model": "auto"
        }

    def _load_model(self, path: str):
        try:
            return YOLO(model=path, task='detect')
        except Exception as e:
            LOGGER.error(f"Failed to load UL model {path}: {e}")
            raise e

    def _init_tracker(self, frame_shape, args):
        """Lazy initializer for tracker to ensure we have frame dimensions."""
        # Dynamic threshold adjustment from args
        vis_conf = getattr(args, "vis_conf", 0.5)
        self.tracker_cfg['new_track_thresh'] = vis_conf
        self.tracker_cfg['track_high_thresh'] = min(vis_conf, 0.4)

        tracker = BOTSORT(argparse.Namespace(**self.tracker_cfg), frame_rate=args.fps)
        
        # Configure GMC
        scale_factor = max(max(frame_shape) // 320, 1)
        tracker.gmc = GMCOnYolo(downscale=scale_factor)
        
        LOGGER.info(f'Tracker initialized. Scaling is {max(max(frame_shape) // 128, 1)}')
        return tracker

    def predict(self, frame: np.ndarray, args):
        # 1. Initialize tracker if first run
        if self.tracker is None:
            self.tracker = self._init_tracker(frame.shape, args)

        # 2. Inference
        model_conf = getattr(args, "model_conf", 0.1)
        img = np.ascontiguousarray(frame, dtype=np.uint8)
        
        # Run YOLO inference
        results = self.model(img, conf=model_conf, verbose=False, classes=[0, 1], show=False, save=False)
        t1 = time.perf_counter() # Inference end time

        if results is None or len(results) == 0:
            return None, t1
        
        # 3. Tracking
        tracks = self.tracker.update(results[0].boxes.cpu().numpy(), img)
        return tracks, t1


class RoboflowMockBackend(InferenceBackend):
    """
    Mock class for Roboflow (RF) integration.
    """
    def __init__(self, model_path: str):
        LOGGER.info(f"Initializing Roboflow Mock Backend for: {model_path}")
        self.model_path = model_path

    def predict(self, frame: np.ndarray, args):
        # Mock Inference delay
        time.sleep(0.02) 
        t1 = time.perf_counter()
        
        # Return empty tracks or dummy tracks
        # Format: [[x1, y1, x2, y2, track_id, conf, class_id], ...]
        # For now, returning None to simulate no detections
        return [], t1


class ModelManager:
    """
    Manages the active inference backend. Detects if model path changes
    and swaps between Ultralytics (UL) and Roboflow (RF) implementations.
    """
    def __init__(self):
        self.current_backend = None
        self.current_model_path = None

    def _get_backend_type(self, path: str):
        # Logic to determine backend based on folder/name
        if "RF" in path or "roboflow" in path.lower():
            return RoboflowMockBackend
        # Default to Ultralytics for "UL" or unknown
        return UltralyticsBackend

    def predict(self, frame: np.ndarray, args):
        requested_path = getattr(args, "model_path", "/app/model/weights-fp16.engine")

        # Check if we need to load/reload the model
        if self.current_backend is None or self.current_model_path != requested_path:
            backend_cls = self._get_backend_type(requested_path)
            self.current_backend = backend_cls(requested_path, args)
            self.current_model_path = requested_path

        return self.current_backend.predict(frame, args)


def is_jetson():
    try:
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                LOGGER.info(f'Device tree reports model is: {model}')
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

def current_ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def capture_loop(reader: StreamReader, capture_queue: queue.Queue, stop_event: threading.Event) -> None:
    is_file = isinstance(reader, FileReader)
    target_interval = 1.0 / reader.fps if (reader.fps and reader.fps > 0) else 0.033
    
    with reader:
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            ret, frame = reader.read()
            read_done = time.perf_counter()

            if not ret or frame is None:
                LOGGER.warning("Capture source ended.")
                capture_queue.put((None, 0.0, 0.0))
                break

            capture_queue.put((frame, read_done, read_done - loop_start))
            
            if is_file:
                process_dur = time.perf_counter() - loop_start
                sleep_time = target_interval - process_dur
                if sleep_time > 0:
                    time.sleep(sleep_time)


def inference_loop(frame_queue: queue.Queue, result_queue: queue.Queue, stop_event: threading.Event, profiler: Profiler, args) -> None:
    model_manager = ModelManager()

    while not stop_event.is_set():
        frame, capture_start_end, capture_time = frame_queue.get()
        if frame is None:
            profiler.clean(["capture", "yolo", "track", "capture_queue"])
            result_queue.put((None, None))
            break

        start_inference = time.perf_counter()
        result, end_inference = model_manager.predict(frame, args)
        end_tracking = time.perf_counter()

        result_queue.put((frame, result))
        result_pushed = time.perf_counter()

        profiler.record("capture", capture_time)
        profiler.record("yolo", end_inference - start_inference)
        profiler.record("track", end_tracking - end_inference)
        profiler.record("latency", time.perf_counter() - capture_start_end)
        profiler.record("result_delayed", result_pushed - end_tracking)
        profiler.record("capture_queue", frame_queue.qsize())
        profiler.record("output_queue", result_queue.qsize())


def output_loop(result_queue: queue.Queue, stop_event: threading.Event, profiler: Profiler, args) -> None:
    input_writer = None
    output_writer = None
    csv_file = None
    csv_writer = None
    frame_count = 0
    
    pipeline_id = getattr(args, "pipeline_id", "unknown")

    try:
        while not stop_event.is_set():
            frame, result = result_queue.get()
            if frame is None:
                break

            start_vis = time.time()
            vis, total_unique_objects = visualize_frame_with_supervision(frame, result, args)
            profiler.record('vis', time.time() - start_vis)

            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            vis = np.ascontiguousarray(vis, dtype=np.uint8)

            frame_count += 1

            if output_writer is None:
                h, w = frame.shape[:2]
                hq_output_dir = getattr(args, "hq_output_dir", "/app")
                os.makedirs(hq_output_dir, exist_ok=True)

                hq_filename = f"output-hq-{pipeline_id}.mkv"
                hq_path = os.path.join(hq_output_dir, hq_filename)
                LOGGER.info("Saving HQ output to %s", hq_path)

                out_pipeline = build_rtsp_and_hq_gst(args.stream_host, args.stream_port, args.output_path, w, h, args.fps, hq_path)
                output_writer = cv2.VideoWriter(out_pipeline, cv2.CAP_GSTREAMER, 0, args.fps, (w, h), True)

                if not output_writer.isOpened():
                    raise RuntimeError("Failed to open combined RTSP+HQ GStreamer pipeline")
                
                hq_csv_filename = f"output-hq-{pipeline_id}.csv"
                hq_csv_path = os.path.join(hq_output_dir, hq_csv_filename)
                csv_file = open(hq_csv_path, "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["frame", "analysis_number", "s_value", "total_unique_objects", "detections"])

            detections_serialized = ""
            if result is not None and len(result) > 0:
                det_parts = []
                for row in result:
                    # Handle possibility of result being just a list vs numpy array if backend differs
                    # Assuming numpy for now as per existing logic
                    x0, y0, x1, y1 = row[:4]
                    conf_val = float(row[5])
                    cls_id = int(row[6])
                    det_parts.append(
                        f"x0={int(x0)}_y0={int(y0)}_x1={int(x1)}_y1={int(y1)}_class={cls_id}_conf={conf_val:.3f}"
                    )
                detections_serialized = "|".join(det_parts)

            if csv_writer is not None:
                csv_start = time.time()
                s_value = round((total_unique_objects * 1111.0) / 100.0, 1)
                csv_writer.writerow([frame_count, pipeline_id, s_value, total_unique_objects, detections_serialized])
                profiler.record('csv_time', time.time() - csv_start)

            if input_writer is not None:
                input_writer.write(frame)

            out_start = time.time()
            output_writer.write(vis)
            profiler.record('dump_out_time', time.time() - out_start)

    finally:
        if input_writer is not None:
            input_writer.release()
        if output_writer is not None:
            output_writer.release()
        if csv_file is not None:
            csv_file.close()


class StreamPipeline:
    def __init__(self, reader: StreamReader, profiler: Profiler, args):
        reset_object_counter()

        self.reader = reader
        self.profiler = profiler
        self.args = args

        self.frame_queue = queue.Queue(maxsize=80)
        self.result_queue = queue.Queue(maxsize=120)
        self.stop_event = threading.Event()

        self.capture_t = threading.Thread(target=capture_loop, args=(self.reader, self.frame_queue, self.stop_event), daemon=True)
        self.inference_t = threading.Thread(target=inference_loop, args=(self.frame_queue, self.result_queue, self.stop_event, self.profiler, self.args), daemon=True)
        self.output_t = threading.Thread(target=output_loop, args=(self.result_queue, self.stop_event, self.profiler, self.args), daemon=True)

        self._threads = [self.capture_t, self.inference_t, self.output_t]

    def start(self) -> None:
        for t in self._threads:
            t.start()

    def _send_sentinels(self) -> None:
        try:
            self.frame_queue.put((None, 0.0, 0.0), timeout=0.1)
        except queue.Full:
            pass
        
        try:
            self.result_queue.put((None, None), timeout=0.1)
        except queue.Full:
            pass

    def stop(self) -> None:
        self.stop_event.set()
        self._send_sentinels()
        for t in self._threads:
            t.join(timeout=2.0)

    def run(self) -> None:
        try:
            while self.output_t.is_alive():
                time.sleep(1.0)
        except KeyboardInterrupt:
            LOGGER.info("Interrupted, stopping...")
        finally:
            self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False