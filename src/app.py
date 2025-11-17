import argparse
import logging
import os

import cv2
import requests
from flask import Flask, Response, jsonify, request, send_file

from inference_pipeline import StreamPipeline
from profiler import Profiler
from stream_readers import StreamReader, V4L2StreamReader, RTSPStreamReader


LOGGER = logging.getLogger("stream_benchmark")

app = Flask(__name__)

# Global runtime state
pipeline = None  # type: StreamPipeline | None
profiler = None  # type: Profiler | None
current_config = {}
last_error = None  # type: str | None


class FileStreamReader(StreamReader):
    """Simple file-based StreamReader using OpenCV."""
    def __init__(self, file_path, fps):
        super().__init__(width=None, height=None, fps=fps)
        self.file_path = file_path

    def open(self):  # type: ignore[override]
        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap or not self.cap.isOpened():
            LOGGER.error("Failed to open file capture: %s", self.file_path)
            self.cap = None
            return False

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1e-3:
            self.fps = int(fps)

        LOGGER.info(
            "Opened file %s (%sx%s @ %s fps)",
            self.file_path,
            self.width or "?", self.height or "?", self.fps or "?",
        )
        return True


def is_pipeline_running():
    """Return True if there is an active StreamPipeline with any live thread."""
    global pipeline
    if pipeline is None:
        return False

    threads = getattr(pipeline, "_threads", None)
    if not threads:
        return False

    return any(t is not None and t.is_alive() for t in threads)


def determine_mode(video, rtsp_transport):
    """
    Decide how to interpret the `video` argument.

    Returns: (mode, device, rtsp_url, file_path)
    """
    file_path = None
    device = None
    rtsp_url = None
    mode = None

    if isinstance(video, str) and video.startswith("/dev/"):
        mode = "v4l2-gs"
        device = video
    elif isinstance(video, str) and video.startswith("rtsp://"):
        mode = "rtsp"
        # For FFmpeg we do *not* append rtsp_transport as a query parameter –
        # ffmpeg will negotiate transports on its own and gracefully fall back
        # to TCP when UDP is rejected with 461 Unsupported Transport.
        rtsp_url = video
    else:
        mode = "file"
        file_path = video
        if not os.path.isfile(file_path):
            raise ValueError("File not found: %s" % file_path)

    return mode, device, rtsp_url, file_path


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/config")
def api_config():
    # 8889 is MediaMTX WebRTC (WHEP) port in your config
    return jsonify({"stream_port": 8889})


@app.route("/api/status")
def api_status():
    state = "running" if is_pipeline_running() else "idle"
    video = current_config.get("video", "")
    rtsp_transport = current_config.get("rtsp_transport", "")
    video_reference = video
    if rtsp_transport and isinstance(video, str) and video.startswith("rtsp://"):
        # purely cosmetic – to show what transport was requested
        video_reference = "%s?rtsp_transport=%s" % (video, rtsp_transport)

    return jsonify(
        {
            "state": state,
            "pipeline_id": os.getpid() if pipeline and is_pipeline_running() else "-",
            "last_error": last_error,
            "config": {"video_reference": video_reference},
        }
    )


@app.route("/api/start", methods=["POST"])
def api_start():
    global pipeline, profiler, last_error, current_config

    # Don't allow concurrent runs
    if is_pipeline_running():
        return jsonify({"error": "already running"}), 400

    # Clean up stale pipeline object, if any
    if pipeline is not None and not is_pipeline_running():
        try:
            pipeline.__exit__(None, None, None)
        except Exception:
            LOGGER.exception("Error while cleaning up stale pipeline before start")
        finally:
            pipeline = None

    last_error = None
    tmp_pipeline = None

    try:
        data = request.json or {}
        video = data.get("video")
        if not video:
            raise ValueError("No video provided")

        rtsp_transport = data.get("rtsp_transport", "")

        mode, device, rtsp_url, file_path = determine_mode(video, rtsp_transport)

        args_dict = {
            "mode": mode,
            "device": device,
            "rtsp_url": rtsp_url,
            "width": 1280,
            "height": 720,
            "fps": 30,
            "print_every": 60,
            "stream_host": "127.0.0.1",
            # RTSP ingest port of MediaMTX
            "stream_port": 8554,
            "log_level": "INFO",
            "original_path": "pub-original",
            "output_path": "pub-output",

            # NEW: model & visualization thresholds
            "model_conf": 0.10,   # YOLO detection / tracking threshold
            "vis_conf": 0.75,     # visualization-only threshold
        }

        args = argparse.Namespace(**args_dict)

        # Choose reader implementation
        if mode == "file":
            reader = FileStreamReader(file_path, args.fps)
        elif mode == "v4l2-gs":
            reader = V4L2StreamReader(args.device, args.width, args.height, args.fps)
        else:  # "rtsp"
            # Use FFmpeg-based reader instead of the GStreamer one from stream_readers
            reader = RTSPStreamReader(args.rtsp_url, args.fps, rtsp_transport)

        # For file & RTSP we can probe actual dimensions before starting
        if isinstance(reader, (FileStreamReader, RTSPStreamReader)):
            with reader:
                cap = reader.cap
                if cap is None or not cap.isOpened():
                    raise RuntimeError("Failed to open stream reader for properties")

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
                fps_val = cap.get(cv2.CAP_PROP_FPS) or args.fps

                args.width = width
                args.height = height
                if fps_val and fps_val > 1e-3:
                    args.fps = int(fps_val)

                LOGGER.info(
                    "Input stream properties: %sx%s @ %s fps",
                    args.width,
                    args.height,
                    args.fps,
                )

        profiler = Profiler(window=200)

        # Use StreamPipeline as a long-lived context manager. We call __enter__
        # manually here so that the pipeline outlives the HTTP request.
        tmp_pipeline = StreamPipeline(reader, profiler, args)
        tmp_pipeline.__enter__()
        pipeline = tmp_pipeline

        current_config = {"video": video, "rtsp_transport": rtsp_transport}
        LOGGER.info("Started new pipeline with mode=%s, video=%s", mode, video)
        return jsonify({"success": True})
    except Exception as e:
        last_error = str(e)
        LOGGER.exception("Failed to start pipeline: %s", e)
        if tmp_pipeline is not None:
            try:
                tmp_pipeline.__exit__(type(e), e, e.__traceback__)
            except Exception:
                LOGGER.exception("Error during pipeline cleanup after failed start")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global pipeline, last_error, current_config

    # Nothing to stop
    if pipeline is None and not is_pipeline_running():
        current_config = {}
        return jsonify({"success": True})

    try:
        if pipeline is not None:
            # Gracefully stop capture / inference / output threads and
            # close all RTSP / OpenCV resources via the context manager.
            pipeline.__exit__(None, None, None)

        pipeline = None
        current_config = {}
        return jsonify({"success": True})
    except Exception as e:
        last_error = str(e)
        LOGGER.exception("Failed to stop pipeline: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no filename"}), 400

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    # Frontend expects a "video" URL and optional rtsp_transport
    return jsonify({"video": path, "rtsp_transport": ""})


@app.route("/<path:path>/whep", methods=["GET", "POST", "OPTIONS"])
def proxy_whep(path):
    """
    Very small HTTP proxy that forwards WHEP requests from the UI to MediaMTX.

    - The UI will hit:   http://<this-app>:8000/<path>/whep
    - We forward to:     http://127.0.0.1:8889/<path>/whep
    """
    target_url = "http://127.0.0.1:8889/%s/whep" % path
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    if request.method == "OPTIONS":
        resp = requests.options(target_url, headers=headers)
    elif request.method == "POST":
        resp = requests.post(target_url, data=request.data, headers=headers)
    else:
        resp = requests.get(target_url, headers=headers)

    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    response_headers = [
        (name, value)
        for (name, value) in resp.raw.headers.items()
        if name.lower() not in excluded_headers
    ]
    return Response(resp.content, resp.status_code, response_headers)


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8000, threaded=True)
