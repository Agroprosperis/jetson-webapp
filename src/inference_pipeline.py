import cv2
import threading
import time
import queue
import logging
import numpy as np
import os

from datetime import datetime
from profiler import Profiler
from stream_readers import StreamReader
from ultralytics import YOLO
from visualize import visualize_frame_with_supervision, reset_object_counter


LOGGER = logging.getLogger("inference_pipeline")
MODEL = YOLO(model="/app/model/weights-fp16.engine", task='segment')


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
    "reid_weights": "/work/upwork/igor/osnet_x0_25_msmt17.onnx",
    "reid_device": "cuda",
    "reid_half": False,
    "proximity_thresh": 0.1,
    "appearance_thresh": 0.15,
}

_BOTSORT_YAML_PATH = "botsort_custom.yaml"

def ensure_botsort_yaml(path: str = _BOTSORT_YAML_PATH) -> str:
    """
    Make sure a BoT-SORT YAML config file exists on disk for Ultralytics YOLO.
    Ultralytics' `model.track(..., tracker=...)` API expects a path, not a dict.
    """
    if os.path.exists(path):
        return path

    lines = []
    for key, value in _BOTSORT_CFG.items():
        lines.append(f"{key}: {value}")
    text = "\n".join(lines) + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    LOGGER.info("Written custom BoT-SORT config to %s", path)
    return path


def build_udp_out_gst(host: str, port: int, width: int, height: int, fps: int) -> str:
    """
    Output pipeline: raw BGR frames -> H.264 in MPEG-TS -> UDP.
    View in VLC with: udp://@:<port>
    """
    pipeline = (
        "appsrc is-live=true block=false format=time do-timestamp=true ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "videoconvert ! "
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 key-int-max=10 bframes=0 ! "
        "mpegtsmux ! "
        f"udpsink host={host} port={port} sync=false async=false"
    )
    LOGGER.info("OUT pipeline: %s", pipeline)
    return pipeline


def build_rtsp_out_gst(host: str, port: int, path: str,
                       width: int, height: int, fps: int) -> str:
    """
    Low-latency RTSP publisher to MediaMTX using software encoding for cross-platform compatibility.
    """
    pipeline = (
        "appsrc is-live=true block=true format=time do-timestamp=true ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast "
        "bitrate=8000000 key-int-max=15 bframes=0 "
        "option-string=\"colormatrix=bt709:colorprim=bt709:transfer=bt709\" ! "
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
        result = run_inference(frame, args)
        end_inference = time.perf_counter()

        profiler.record("capture", capture_time)
        profiler.record("infer", end_inference - start_inference)
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
    frame_count = 0

    try:
        while not stop_event.is_set():
            try:
                frame, result, meta = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:
                break

            # Annotated frame
            vis = visualize_frame_with_supervision(frame, result, meta, args)
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

            # Lazy-init stream writers on first frame
            if output_writer is None:
                h, w = frame.shape[:2]
                print('Open output: ', h, w)

                in_pipeline = build_rtsp_out_gst(host=args.stream_host, port=args.stream_port, path=args.original_path, width=w, height=h, fps=args.fps)
                out_pipeline = build_rtsp_out_gst(host=args.stream_host, port=args.stream_port, path=args.output_path, width=w, height=h, fps=args.fps)

                input_writer = cv2.VideoWriter(in_pipeline, cv2.CAP_GSTREAMER, 0, args.fps, (w, h), True)
                output_writer = cv2.VideoWriter(out_pipeline, cv2.CAP_GSTREAMER, 0, args.fps, (w, h), True)

                if not input_writer.isOpened() or not output_writer.isOpened():
                    raise RuntimeError("Failed to open GStreamer RTSP output pipeline(s)")

            if input_writer is not None:
                #cv2.imwrite('/app/output-original.jpeg', frame)
                input_writer.write(frame)

            if output_writer is not None:
                print('Dumping output: ', h, w)
                cv2.imwrite('/app/output-original.jpeg', vis)
                output_writer.write(vis)

            frame_count += 1
            if meta is not None and frame_count % args.print_every == 0:
                inst_fps = meta.get("inst_fps", 0.0)
                avg_fps = profiler.avg_fps()
                LOGGER.info(
                    "[%6d] FPS inst/avg: %5.1f / %5.1f | capture: %6.2f ms infer: %6.2f ms | latency: %6.2f ms",
                    frame_count, inst_fps, avg_fps, profiler.avg_ms("capture"), profiler.avg_ms("infer"), profiler.avg_ms("latency"),
                )
    finally:
        if input_writer is not None:
            input_writer.release()
        if output_writer is not None:
            output_writer.release()


def run_inference(frame: np.ndarray, args):
    tracker_cfg_path = ensure_botsort_yaml()
    model_conf = getattr(args, "model_conf", 0.1)
    results = MODEL.track(
        source=frame,
        persist=True,
        stream=False,
        verbose=False,
        tracker=tracker_cfg_path,
        conf=model_conf,  # <- 0.1 by default from args
        imgsz=640,
        half=True
    )
    return results[0] if results else None


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
