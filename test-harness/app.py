import argparse
import logging
import threading
import queue
import time

from inference_pipeline import capture_loop, inference_loop, output_loop
from profiler import Profiler
from stream_readers import V4L2StreamReader, RTSPStreamReader


LOGGER = logging.getLogger("stream_benchmark")


def main():
    args = argparse.ArgumentParser(description="Low-latency /dev/video0 & RTSP benchmark with profiling")
    args.add_argument("--mode", choices=["v4l2-gs", "rtsp"], required=True, help="v4l2-gs via GStreamer; rtsp URL.")
    args.add_argument("--device", default="/dev/video0", help="V4L2 device (default: /dev/video0).")
    args.add_argument("--rtsp-url", default="rtsp://localhost:8554/test", help="RTSP URL to read from.")
    args.add_argument("--width", type=int, default=1280)
    args.add_argument("--height", type=int, default=720)
    args.add_argument("--fps", type=int, default=30)
    args.add_argument("--print-every", type=int, default=60, help="Print stats every N frames.")
    args.add_argument("--stream-host", default="127.0.0.1", help="Host for UDP stream (default: 127.0.0.1).")
    args.add_argument("--stream-port", default=8554)
    args.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    args.add_argument("--original-path", default="pub-original")
    args.add_argument("--output-path", default="pub-output")
    args = args.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Suppress Ultralytics / YOLO chatter
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    # Build reader
    if args.mode == "v4l2-gs":
        reader = V4L2StreamReader(device=args.device, width=args.width, height=args.height, fps=args.fps)
    else:
        reader = RTSPStreamReader(rtsp_url=args.rtsp_url, fps=args.fps)

    profiler = Profiler(window=200)

    # Queues
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    # Threads
    capture_t = threading.Thread(target=capture_loop, args=(reader, frame_queue, stop_event), daemon=True)
    inference_t = threading.Thread(target=inference_loop, args=(frame_queue, result_queue, stop_event, profiler), daemon=True)
    output_t = threading.Thread(target=output_loop, args=(result_queue, stop_event, profiler, args), daemon=True)

    for thread in [capture_t, inference_t, output_t]: thread.start()

    try:
        while output_t.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted, stopping...")
    finally:
        stop_event.set()
        try:
            frame_queue.put((None, 0.0, 0.0), timeout=0.1)
        except queue.Full:
            pass
        try:
            result_queue.put((None, None, None), timeout=0.1)
        except queue.Full:
            pass

        capture_t.join(timeout=2.0)
        inference_t.join(timeout=2.0)
        output_t.join(timeout=2.0)


if __name__ == "__main__":
    main()
