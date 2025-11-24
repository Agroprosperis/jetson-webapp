import argparse
import cv2
import threading
import time
import queue
import logging
import numpy as np
import os
import csv
import ultralytics

from datetime import datetime
from profiler import Profiler
from stream_readers import StreamReader, FileReader
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT

from visualize import visualize_frame_with_supervision, reset_object_counter


LOGGER = logging.getLogger("inference_pipeline")
MODEL = None
TRACKER = None


# Your BoT-SORT config
_BOTSORT_CFG = {
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


import platform


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


def build_rtsp_and_hq_gst(
    host: str,
    port: int,
    path: str,
    width: int,
    height: int,
    fps: int,
    hq_path: str,
) -> str:
    # Bitrate Strategy:
    # To ensure the Disk Recording never drops frames, the Pipeline must never block.
    # Since we use TCP for RTSP, a high bitrate will fill the TCP window and BLOCK the pipeline.
    # 8000 kbps (8 Mbps) is a safe sweet spot for 4K H.264 that fits in standard Wi-Fi/Ethernet buffers.
    bitrate_kbit = 8000
    
    rtsp_url = f"rtsp://{host}:{port}/{path}"

    pipeline = (
        "appsrc is-live=true block=true format=time do-timestamp=true "
        "max-bytes=100000000 ! " 
        # RAW QUEUE: Blocking (leaky=no). 
        # We hold frames here if the encoder is busy.
        # Max size 5 buffers (approx 150ms latency).
        "queue max-size-buffers=5 leaky=no ! "
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=I420 ! "
        
        # CPU Encoding
        # 'ultrafast' is mandatory for 4K CPU encoding.
        # 'tune=zerolatency' helps streaming.
        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate_kbit} "
        "key-int-max=30 bframes=0 sliced-threads=true threads=4 ! "
        "h264parse config-interval=-1 ! "
        
        # Split the encoded stream
        "tee name=t "

        # Branch 1: Disk Recording (MKV)
        # Blocking queue ensures we write every encoded frame.
        "t. ! queue max-size-buffers=10 leaky=no ! "
        "matroskamux ! "
        f"filesink location=\"{hq_path}\" sync=false "

        # Branch 2: RTSP Stream
        # CRITICAL FIX: leaky=no (Blocking). 
        # Dropping frames here caused the "blocky/gray" video artifacts.
        # We increase the buffer size (200 buffers) to absorb network jitters without blocking the Disk branch.
        # If this queue fills (Net < 8Mbps consistently), the whole pipeline (including Disk) will slow down.
        "t. ! queue max-size-buffers=200 leaky=no ! "
        f"rtspclientsink location={rtsp_url} protocols=tcp "
    )
    
    LOGGER.info("Pipeline: %s", pipeline)
    return pipeline


def current_ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def capture_loop(reader: StreamReader, capture_queue: queue.Queue, stop_event: threading.Event) -> None:
    # Determine if we need to pace the reading (File Mode)
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
    while not stop_event.is_set():
        frame, capture_start_end, capture_time = frame_queue.get()
        if frame is None:
            profiler.clean(["capture", "yolo", "track", "capture_queue"])
            result_queue.put((None, None))
            break

        start_inference = time.perf_counter()
        result, end_inference = run_inference(frame, args)
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
    
    # Retrieve Pipeline ID (Analysis Number) once
    pipeline_id = getattr(args, "pipeline_id", "unknown")

    try:
        while not stop_event.is_set():
            frame, result = result_queue.get()
            if frame is None:
                break

            start_vis = time.time()
            # Note: result is the 'tracks' numpy array
            vis, total_unique_objects = visualize_frame_with_supervision(frame, result, args)
            profiler.record('vis', time.time() - start_vis)

            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            vis = np.ascontiguousarray(vis, dtype=np.uint8)

            frame_count += 1

            # Lazy-init combined RTSP + HQ writer on first frame
            if output_writer is None:
                h, w = frame.shape[:2]

                hq_output_dir = getattr(args, "hq_output_dir", "/app")
                os.makedirs(hq_output_dir, exist_ok=True)

                hq_filename = f"output-hq-{pipeline_id}.mkv"
                hq_path = os.path.join(hq_output_dir, hq_filename)
                LOGGER.info("Saving HQ output to %s", hq_path)

                LOGGER.info("Configuring 4K CPU Pipeline: %sx%s @ %s fps", w, h, args.fps)
                out_pipeline = build_rtsp_and_hq_gst(args.stream_host, args.stream_port, args.output_path, w, h, args.fps, hq_path)
                output_writer = cv2.VideoWriter(out_pipeline, cv2.CAP_GSTREAMER, 0, args.fps, (w, h), True)

                if not output_writer.isOpened():
                    raise RuntimeError("Failed to open combined RTSP+HQ GStreamer pipeline")
                else:
                    LOGGER.info(f'Open combined output stream {w}x{h}: {out_pipeline}')

                # Init CSV
                hq_csv_filename = f"output-hq-{pipeline_id}.csv"
                hq_csv_path = os.path.join(hq_output_dir, hq_csv_filename)
                csv_file = open(hq_csv_path, "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                # UPDATED HEADER: Added Analysis Number and S Value
                csv_writer.writerow(["frame", "analysis_number", "s_value", "total_unique_objects", "detections"])

            # -----------------------------------------------------
            # FIXED CSV GENERATION FOR NUMPY ARRAY (TRACKS)
            # -----------------------------------------------------
            detections_serialized = ""
            if result is not None and len(result) > 0:
                # result is the 'tracks' numpy array: [x1, y1, x2, y2, track_id, conf, class_id, ...]
                det_parts = []
                for row in result:
                    x0, y0, x1, y1 = row[:4]
                    # row[4] is track_id if needed
                    # Index 5 is confidence
                    # Index 6 is class_id
                    conf_val = float(row[5])
                    cls_id = int(row[6])
                    
                    det_parts.append(
                        f"x0={int(x0)}_y0={int(y0)}_x1={int(x1)}_y1={int(y1)}_class={cls_id}_conf={conf_val:.3f}"
                    )
                detections_serialized = "|".join(det_parts)

            if csv_writer is not None:
                csv_start = time.time()
                
                # Calculate S Value
                # Formula: (count * 1111) / 100
                s_value = round((total_unique_objects * 1111.0) / 100.0, 1)

                # UPDATED ROW: Write pipeline_id and s_value
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


class GMCOnYolo(ultralytics.trackers.utils.gmc.GMC):
    def __init__(self, method: str = "sparseOptFlow", downscale: int = 1):
        super().__init__(method=method, downscale=downscale)

    def apply(self, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        return super().apply(raw_frame, detections)


def run_inference(frame: np.ndarray, args):
    global MODEL, TRACKER

    if MODEL is None:
        MODEL = YOLO(model="/app/model/weights-fp16.engine", task='detect')
    
    if TRACKER is None:
        _BOTSORT_CFG['new_track_thresh'] = args.vis_conf
        _BOTSORT_CFG['track_high_thresh'] = min(args.vis_conf, 0.4)

        TRACKER = BOTSORT(argparse.Namespace(**_BOTSORT_CFG), frame_rate=args.fps)
        LOGGER.info(f'Tracker scaling is {max(max(frame.shape) // 128, 1)}')
        TRACKER.gmc = GMCOnYolo(downscale=(max(max(frame.shape) // 320, 1)))
    
    model_conf = getattr(args, "model_conf", 0.1)
    img = np.ascontiguousarray(frame, dtype=np.uint8)

    results = MODEL(img, conf=model_conf, verbose=False, classes=[0, 1], show=False, save=False)
    t1 = time.perf_counter()

    if results is None or len(results) == 0:
        return
    
    tracks = TRACKER.update(results[0].boxes.cpu().numpy(), img)
    return tracks, t1


class StreamPipeline:
    def __init__(self, reader: StreamReader, profiler: Profiler, args):
        reset_object_counter()

        self.reader = reader
        self.profiler = profiler
        self.args = args

        # Queues & stop flag
        self.frame_queue = queue.Queue(maxsize=80)
        self.result_queue = queue.Queue(maxsize=120)
        self.stop_event = threading.Event()

        # Threads
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
            self.result_queue.put((None, None, None), timeout=0.1)
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