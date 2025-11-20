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
from stream_readers import StreamReader
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.cfg import get_cfg

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


def build_rtsp_out_gst(host: str, port: int, path: str,
                       width: int, height: int, fps: int) -> str:
    """
    Low-latency RTSP publisher to MediaMTX using software encoding for cross-platform compatibility.
    """
    target_height = 720
    target_width = int(width * target_height / height)
    pipeline = (
        "appsrc is-live=true block=true format=time do-timestamp=true ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        # Accept raw BGR from the appsrc at any size, only fix framerate here
        f"video/x-raw,format=BGR,framerate={fps}/1 ! "
        "videoconvert ! "
        "videoscale method=bilinear ! "
        # Force 720p output to encoder
        f"video/x-raw,format=I420,width={target_width},height={target_height},framerate={fps}/1 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast "
        "bitrate=36000 key-int-max=45 bframes=0 "
        "option-string=\"colormatrix=bt709:colorprim=bt709:transfer=bt709:deblock=-1,-1:aq-mode=1:aq-strength=0.8\" ! "
        "h264parse ! "
        f"rtspclientsink protocols=tcp location=rtsp://{host}:{port}/{path}"
    )
    LOGGER.info("OUT RTSP pipeline (%s): %s", path, pipeline)
    return pipeline


def current_ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def capture_loop(reader: StreamReader, frame_queue: queue.Queue, stop_event: threading.Event) -> None:
    """
    Capture frames as fast as *consumer* allows (queue size=1).
    - Puts (frame, capture_t0, capture_dt) into the queue.
    """
    with reader:
        while not stop_event.is_set():
            start_time = time.perf_counter()
            ret, frame = reader.read()
            end_time = time.perf_counter()

            if not ret or frame is None:
                LOGGER.warning("Capture failed, stopping capture thread.")
                try:
                    frame_queue.put((None, 0.0, 0.0), timeout=0.1)
                except queue.Full:
                    pass
                break

            try:
                frame_queue.put((frame, start_time, end_time - start_time), timeout=0.05)
            except queue.Full:
                continue


def inference_loop(
    frame_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    profiler: Profiler,
    args,
) -> None:
    start_time = time.perf_counter()

    while not stop_event.is_set():
        try:
            frame, capture_start_end, capture_time = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Sentinel from capture -> propagate then exit
        if frame is None:
            try:
                result_queue.put((None, None, None), timeout=0.1)
            except queue.Full:
                pass
            break

        start_inference = time.perf_counter()
        result, end_inference = run_inference(frame, args)
        end_tracking = time.perf_counter()

        profiler.record("capture", capture_time)
        profiler.record("yolo", end_inference - start_inference)
        profiler.record("track", end_tracking - end_inference)
        profiler.record("latency", end_inference - capture_start_end)
        profiler.record("interval", end_inference - start_time)

        inst_fps = 1.0 / (end_inference - start_time) if (end_inference - start_time) > 0 else 0.0
        send_ts = current_ts()

        meta = {
            "capture_dt": capture_time,
            "infer_dt": end_inference - start_inference,
            "latency_dt": end_inference - capture_start_end,
            "interval_dt": end_inference - start_time,
            "inst_fps": inst_fps,
            "send_ts": send_ts,
        }

        try:
            result_queue.put((frame, result, meta), timeout=0.1)
        except queue.Full:
            # Drop if output is stuck; preserves low latency
            continue


def output_loop(result_queue: queue.Queue, stop_event: threading.Event, profiler: Profiler, args) -> None:
    input_writer = None
    output_writer = None
    hq_writer = None          # local file writer for HQ video
    csv_file = None           # NEW: CSV file handle
    csv_writer = None         # NEW: CSV writer
    frame_count = 0
    seen_track_ids = set()    # NEW: for total_unique_objects

    try:
        while not stop_event.is_set():
            try:
                frame, result, meta = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:
                break

            # Annotated frame
            start_vis = time.time()
            vis = visualize_frame_with_supervision(frame, result, meta, args)
            vis_time = time.time() - start_vis
            profiler.record('visualization_time', vis_time)

            frame = frame.copy()
            if frame.ndim == 2:  # grayscale RTSP stream, for example
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:  # BGRA
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            # Ensure vis is also proper 3-channel BGR, contiguous
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            elif vis.ndim == 3 and vis.shape[2] == 4:
                vis = cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR)
            vis = np.ascontiguousarray(vis, dtype=np.uint8)

            # Increment frame counter early so CSV starts from frame=1
            frame_count += 1

            # Lazy-init stream writers on first frame
            if output_writer is None:
                h, w = frame.shape[:2]
                print("Open output: ", h, w)

                in_pipeline = build_rtsp_out_gst(
                    host=args.stream_host,
                    port=args.stream_port,
                    path=args.original_path,
                    width=w,
                    height=h,
                    fps=args.fps,
                )
                out_pipeline = build_rtsp_out_gst(
                    host=args.stream_host,
                    port=args.stream_port,
                    path=args.output_path,
                    width=w,
                    height=h,
                    fps=args.fps,
                )

                input_writer = cv2.VideoWriter(
                    in_pipeline, cv2.CAP_GSTREAMER, 0, args.fps, (w, h), True
                )
                output_writer = cv2.VideoWriter(
                    out_pipeline, cv2.CAP_GSTREAMER, 0, args.fps, (w, h), True
                )

                if not input_writer.isOpened() or not output_writer.isOpened():
                    raise RuntimeError("Failed to open GStreamer RTSP output pipeline(s)")
                
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                print(args)
                
                pipeline_id = getattr(args, "pipeline_id", "unknown")
                hq_output_dir = getattr(args, "hq_output_dir", "/app")

                os.makedirs(hq_output_dir, exist_ok=True)
                hq_filename = f"output-hq-{pipeline_id}.mp4"
                hq_path = os.path.join(hq_output_dir, hq_filename)

                LOGGER.info("Saving HQ output to %s", hq_path)

                hq_writer = cv2.VideoWriter(hq_path, fourcc, args.fps, (w, h), True)
                if not hq_writer.isOpened():
                    raise RuntimeError("Failed to open HQ file writer")

                hq_csv_filename = f"output-hq-{pipeline_id}.csv"
                hq_csv_path = os.path.join(hq_output_dir, hq_csv_filename)
                LOGGER.info("Saving per-frame stats CSV to %s", hq_csv_path)

                csv_file = open(hq_csv_path, "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["frame", "total_unique_objects", "detections"])

            total_unique_objects = 0
            detections_serialized = ""

            if result is not None:
                boxes = getattr(result, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy
                    cls = getattr(boxes, "cls", None)
                    conf = getattr(boxes, "conf", None)
                    track_ids = getattr(boxes, "id", None)

                    # Move to CPU / numpy where needed
                    xyxy_np = xyxy.cpu().numpy()
                    cls_np = cls.cpu().numpy() if cls is not None else None
                    conf_np = conf.cpu().numpy() if conf is not None else None
                    track_ids_np = track_ids.cpu().numpy() if track_ids is not None else None

                    # Update unique objects counter using track IDs (BoT-SORT)
                    if track_ids_np is not None:
                        for tid in track_ids_np:
                            seen_track_ids.add(int(tid))
                    total_unique_objects = len(seen_track_ids)

                    # Serialize detections as single string:
                    # x0=..._y0=..._x1=..._y1=..._class=..._conf=...|...
                    det_parts = []
                    num_dets = xyxy_np.shape[0]
                    for i in range(num_dets):
                        x0, y0, x1, y1 = xyxy_np[i]
                        cls_id = int(cls_np[i]) if cls_np is not None else -1
                        conf_val = float(conf_np[i]) if conf_np is not None else 0.0
                        det_parts.append(
                            f"x0={int(x0)}_y0={int(y0)}_x1={int(x1)}_y1={int(y1)}_class={cls_id}_conf={conf_val:.3f}"
                        )
                    detections_serialized = "|".join(det_parts)

            # If tracking is unavailable but we've already seen IDs earlier, keep count
            if total_unique_objects == 0 and seen_track_ids:
                total_unique_objects = len(seen_track_ids)

            if csv_writer is not None:
                csv_start = time.time()
                csv_writer.writerow([frame_count, total_unique_objects, detections_serialized])
                profiler.record('csv_time', time.time() - csv_start)

            if input_writer is not None:
                in_start = time.time()
                input_writer.write(frame)
                profiler.record('dump_input_time', time.time() - in_start)

            if output_writer is not None:
                out_hq_start = time.time()
                if hq_writer is not None:
                    hq_writer.write(vis)
                profiler.record('dump_out_hq_time', time.time() - out_hq_start)

                out_start = time.time()
                output_writer.write(vis)
                profiler.record('dump_out_time', time.time() - out_start)

            if meta is not None and frame_count % args.print_every == 0:
                inst_fps = meta.get("inst_fps", 0.0)
                avg_fps = profiler.avg_fps()
                LOGGER.info(
                    "[%6d] FPS inst/avg: %5.1f / %5.1f | capture: %6.2f ms yolo: %6.2f ms, track: %6.2f ms | latency: %6.2f ms | vis: %6.2f ms | out: %6.2fms, %6.2fms, %6.2fms, %6.2fms",
                    frame_count,
                    inst_fps,
                    avg_fps,
                    profiler.avg_ms("capture"),
                    profiler.avg_ms("yolo"),
                    profiler.avg_ms("track"),
                    profiler.avg_ms("latency"),
                    profiler.avg_ms("visualization_time"),
                    profiler.avg_ms("csv_time"),
                    profiler.avg_ms("dump_input_time"),
                    profiler.avg_ms("dump_out_hq_time"),
                    profiler.avg_ms("dump_out_time"),
                )
    finally:
        if input_writer is not None: input_writer.release()
        if output_writer is not None: output_writer.release()
        if hq_writer is not None: hq_writer.release()
        if csv_file is not None: csv_file.close()


class GMCOnYolo(ultralytics.trackers.utils.gmc.GMC):
    def __init__(self, method: str = "sparseOptFlow", downscale: int = 1):
        super().__init__(method=method, downscale=downscale)

    def apply(self, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        if raw_frame.ndim == 3 and raw_frame.shape[2] == 3:
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = raw_frame

        return super().apply(frame, detections)


def run_inference(frame: np.ndarray, args):
    global MODEL, TRACKER

    if MODEL is None:
        MODEL = YOLO(model="/app/model/weights-fp16.engine", task='detect')
    
    if TRACKER is None:
        TRACKER = BOTSORT(argparse.Namespace(**_BOTSORT_CFG), frame_rate=args.fps)
        TRACKER.gmc = GMCOnYolo(downscale=2)
    
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
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
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
        """Block while output thread is alive."""
        try:
            while self.output_t.is_alive():
                time.sleep(0.5)
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
