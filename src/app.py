import argparse
import json
import logging
import os
import uuid

import cv2
import requests
from flask import Flask, Response, jsonify, request, send_file

from inference_pipeline import StreamPipeline
from profiler import Profiler
from stream_readers import V4L2StreamReader, FileStreamReader


LOGGER = logging.getLogger("app")
CONFIG_FILEPATH = "/app/config.json"

app = Flask(__name__)

# Global runtime state
pipeline = None  # type: StreamPipeline | None
profiler = None  # type: Profiler | None
current_config = {}
last_error = None  # type: str | None
pipeline_id = None  # type: str | None


def is_pipeline_running():
    """Return True if there is an active StreamPipeline with any live thread."""
    global pipeline
    if pipeline is None:
        return False

    threads = getattr(pipeline, "_threads", None)
    if not threads:
        return False

    return any(t is not None and t.is_alive() for t in threads)


def determine_mode(video):
    """
    Decide how to interpret the `video` argument.

    Returns: (mode, device, file_path)
    """
    file_path = None
    device = None
    mode = None

    if isinstance(video, str) and video.startswith("/dev/"):
        mode = "v4l2-gs"
        device = video
    else:
        mode = "file"
        file_path = video
        if not os.path.isfile(file_path):
            raise ValueError("File not found: %s" % file_path)

    return mode, device, file_path


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/config")
def api_config():
    return jsonify({"stream_port": 8889})


@app.route("/api/status")
def api_status():
    global pipeline_id
    
    state = "running" if is_pipeline_running() else "idle"
    video = current_config.get("video", "")
    video_reference = video
    pid_value = pipeline_id if pipeline and is_pipeline_running() else "-"

    return jsonify({
        "state": state,
        "pipeline_id": pid_value,
        "last_error": last_error,
        "config": {"video_reference": video_reference},
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    global pipeline, profiler, last_error, current_config, pipeline_id

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

        mode, device, file_path = determine_mode(video)
        pipeline_id = uuid.uuid4().hex

        args_dict = dict()
        if os.path.exists(CONFIG_FILEPATH):
            with open(CONFIG_FILEPATH, 'r') as config_input:
                args_dict = json.load(config_input)

        args_dict = dict(args_dict, 
            mode=mode,
            device=device,
            width=1280,
            height=720,
            fps=30,
            print_every=60,
            stream_host="127.0.0.1",
            stream_port=8554,
            log_level="INFO",
            original_path=None,
            output_path="pub-output",

            model_conf=0.10,   # YOLO detection / tracking threshold
            vis_conf=0.75,     # visualization-only threshold
            pipeline_id=pipeline_id,          # <- pass unique ID
            hq_output_dir="/app/output_hq",   # optional: base dir for HQ files
            dump_original=False,
            dump_processed=True,
            output_stream='WebRTC', # options: WebRTC or MJPEG
            class_names = ['CouldBeTilletia', 'Tilletia']
        )

        args = argparse.Namespace(**args_dict)

        # Choose reader implementation
        if mode == "file":
            reader = FileStreamReader(file_path, args.fps)
        elif mode == "v4l2-gs":
            reader = V4L2StreamReader(args.device, args.width, args.height, args.fps)
        else:
            raise NotImplementedError()

        # For file & RTSP we can probe actual dimensions before starting
        if isinstance(reader, (FileStreamReader)):
            with reader:
                cap = reader.cap
                if cap is None or not cap.isOpened():
                    raise RuntimeError("Failed to open stream reader for properties")

                args.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
                args.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
                fps_val = cap.get(cv2.CAP_PROP_FPS) or args.fps

                if fps_val and fps_val > 1e-3:
                    args.fps = int(fps_val)

                LOGGER.info("Input stream properties: %sx%s @ %s fps", args.width, args.height, args.fps,)

        profiler = Profiler(window=200)

        # Use StreamPipeline as a long-lived context manager. We call __enter__
        # manually here so that the pipeline outlives the HTTP request.
        print(args)
        tmp_pipeline = StreamPipeline(reader, profiler, args)
        tmp_pipeline.__enter__()
        pipeline = tmp_pipeline

        current_config = {"video": video}
        LOGGER.info("Started new pipeline with mode=%s, video=%s", mode, video)

        return jsonify({"success": True, "pipeline_id": pipeline_id})

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
    global pipeline, last_error, current_config, pipeline_id

    # Nothing to stop
    if pipeline is None and not is_pipeline_running():
        current_config = {}
        return jsonify({"success": True})

    try:
        if pipeline is not None:
            # Gracefully stop capture / inference / output threads and
            # close all RTSP / OpenCV resources via the context manager.
            pipeline.__exit__(None, None, None)

        pipeline, pipeline_id = None, None
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
    return jsonify({"video": path})


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
    response_headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
    return Response(resp.content, resp.status_code, response_headers)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8000, threaded=True)