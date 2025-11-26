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

from datetime import datetime

# Third-party imports
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.gmc import GMC

from profiler import Profiler
from stream_readers import StreamReader, FileReader
from visualize import visualize_frame_with_supervision, reset_object_counter


LOGGER = logging.getLogger("inference_pipeline")


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

    def _init_tracker(self, frame_shape, args):
        """Lazy initializer for tracker to ensure we have frame dimensions."""
        vis_conf = getattr(args, "vis_conf", 0.5)
        self.tracker_cfg['new_track_thresh'] = vis_conf
        self.tracker_cfg['track_high_thresh'] = min(vis_conf, 0.4)

        tracker = BOTSORT(argparse.Namespace(**self.tracker_cfg), frame_rate=args.fps)
        scale_factor = max(max(frame_shape) // 320, 1)
        tracker.gmc = GMCOnYolo(downscale=scale_factor)
        
        LOGGER.info(f'Tracker initialized. Scaling is {max(max(frame_shape) // 128, 1)}')
        return tracker
    
    def _track(self, results: list, frame: np.ndarray, args):
        if self.tracker is None:
            self.tracker = self._init_tracker(frame.shape, args)
        
        start_time = time.perf_counter()
        detection_boxes = results[0].boxes
        detection_boxes = detection_boxes.cpu().numpy()
        tracks = self.tracker.update(detection_boxes, frame)

        return tracks, time.perf_counter() - start_time


class GMCOnYolo(GMC):
    """Helper class for Ultralytics BoT-SORT GMC."""
    def __init__(self, method: str = "sparseOptFlow", downscale: int = 1):
        super().__init__(method=method, downscale=downscale)

    def apply(self, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        return super().apply(raw_frame, detections)


class UltralyticsBackend(BotSortTrackerBackend):
    def __init__(self, model_path: str, args):
        super().__init__()

        LOGGER.info(f"Initializing Ultralytics Backend with model: {model_path}")
        self.model_path = model_path
        self.model = self._load_model(model_path)

    def _load_model(self, path: str):
        try:
            return YOLO(model=path, task='detect')
        except Exception as e:
            LOGGER.error(f"Failed to load UL model {path}: {e}")
            raise e

    def _detect_objects(self, frame: np.ndarray, args):
        model_conf = getattr(args, "model_conf", 0.1)
        start_time = time.perf_counter()
        results = self.model(frame, conf=model_conf, verbose=False, classes=[0, 1], show=False, save=False, half=True)
        end_time = time.perf_counter()

        if results is None or len(results) == 0:
            h, w = frame.shape[:2]
            empty_boxes = Boxes(torch.zeros((0, 6)), orig_shape=(h, w))
            return [MockResult(empty_boxes)], end_time - start_time 
        
        return results, end_time - start_time


class MockResult:
    """A simple wrapper to mimic the Ultralytics Results object structure."""
    def __init__(self, boxes):
        self.boxes = boxes


class RoboflowBackend(BotSortTrackerBackend):
    """
    TensorRT Backend for Roboflow/Custom models.
    Integrated from inference_tensorrt_rf.py
    """
    def __init__(self, model_path: str, args):
        super().__init__()

        LOGGER.info(f"Initializing TensorRT Backend for: {model_path}")
        self.model_path = model_path
        
        # Initialize TRT Logger and Runtime
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        try:
            with open(model_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            LOGGER.error(f"Failed to load engine {model_path}: {e}")
            raise e

        self.context = self.engine.create_execution_context()
        
        # Allocation
        self.inputs = []
        self.outputs = []
        self.bindings = [None] * self.engine.num_io_tensors
        self.input_shape = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = self.engine.get_tensor_shape(name)
            dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            
            # Handle dynamic batch size if present, defaulting to 1
            if -1 in shape: 
                shape = (1, 3, 640, 640)
                
            # Allocate GPU memory
            tensor = torch.zeros(tuple(shape), dtype=dtype, device='cuda').contiguous()
            binding = {'index': i, 'tensor': tensor, 'ptr': tensor.data_ptr(), 'shape': shape}
            self.bindings[i] = binding['ptr']
            if is_input: 
                self.inputs.append(binding)
                self.input_shape = shape
            else: 
                self.outputs.append(binding)
            
        self.h, self.w = (self.input_shape[2], self.input_shape[3]) if self.input_shape else (640, 640)
        LOGGER.info(f"TRT Engine loaded. Input Shape: {self.h}x{self.w}")

    def _trt_to_torch_dtype(self, trt_dtype):
        return {
            trt.float32: torch.float32, 
            trt.float16: torch.float16, 
            trt.int32: torch.int32
        }.get(trt_dtype, torch.float32)

    def _preprocess(self, image):
        """Resize and normalize image to tensor."""
        # Resize
        img = cv2.resize(image, (self.w, self.h))
        # Convert to tensor, put on GPU, permute to [C, H, W]
        tensor = torch.from_numpy(img).cuda().permute(2, 0, 1).float()
        # Normalize (0-1) and Standardize (ImageNet stats)
        tensor = tensor[[2, 1, 0], :, :] / 255.0  # BGR to RGB split and normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0)

    def _postprocess_gpu(self, outputs, orig_w, orig_h, conf_thresh=0.5):
        logits, boxes = None, None
        for out in outputs:
            t = out['tensor']
            # Heuristics to identify boxes vs logits based on shape
            if t.shape[-1] == 4: boxes = t
            elif len(t.shape) >= 3 and t.shape[-2] == 4: boxes = t.transpose(1, 2)
            elif len(t.shape) == 2 and t.shape[0] == 4: boxes = t.t().unsqueeze(0)
            else: logits = t

        if logits is None or boxes is None: 
            return None

        # Ensure shapes align
        num_queries = boxes.shape[1]
        if logits.shape[-1] == num_queries and logits.shape[-2] != num_queries:
            logits = logits.transpose(1, 2)
        
        # GPU operations
        scores = torch.sigmoid(logits.clamp(-100, 100))
        max_scores, class_ids = torch.max(scores[0], dim=-1)
        mask = max_scores > conf_thresh
        
        if not mask.any(): 
            return None

        filt_scores = max_scores[mask]
        filt_classes = class_ids[mask]
        filt_boxes = boxes[0][mask]

        # Decode boxes on GPU
        cx, cy, w, h = filt_boxes.unbind(-1)
        x1 = (cx - 0.5 * w) * orig_w
        y1 = (cy - 0.5 * h) * orig_h
        x2 = (cx + 0.5 * w) * orig_w
        y2 = (cy + 0.5 * h) * orig_h
        
        xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        
        # Move to CPU for pipeline consumption
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
            # Return empty Boxes object on error
            empty_boxes = Boxes(torch.zeros((0, 6)), orig_shape=(orig_h, orig_w))
            return [MockResult(empty_boxes)], time.perf_counter() - start_time

        # Handle no detections
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
    """
    Manages the active inference backend. Detects if model path changes
    and swaps between Ultralytics (UL) and Roboflow (RF) implementations.
    """
    def __init__(self):
        self.current_backend = None
        self.current_model_path = None

    def _get_backend_type(self, path: str):
        # Logic to determine backend based on folder/name
        # If path contains 'rf' directory or logic dictates, use TRT Backend
        if "/rf/" in path or "rfdetr" in path.lower():
            return RoboflowBackend
        # Default to Ultralytics
        return UltralyticsBackend

    def predict(self, frame: np.ndarray, args):
        requested_path = getattr(args, "model_path", "/app/model/weights-fp16.engine")

        # Check if we need to load/reload the model
        if self.current_backend is None or self.current_model_path != requested_path:
            backend_cls = self._get_backend_type(requested_path)
            # Clean up old backend if exists (mostly for CUDA memory if needed)
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
        result, inference_time, tracker_time = model_manager.predict(frame, args)
        result_queue.put((frame, result))
        end_inference = time.perf_counter()

        profiler.record("capture", capture_time)
        profiler.record("inf", inference_time)
        profiler.record("track", tracker_time)
        profiler.record("latency", time.perf_counter() - capture_start_end)
        profiler.record("processing", end_inference - start_inference)
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
                    # x1, y1, x2, y2, track_id, conf, class_id
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