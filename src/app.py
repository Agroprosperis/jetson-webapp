import argparse
import json
import logging
import os
import uuid
import glob
from datetime import datetime

import cv2
import requests
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from inference_pipeline import StreamPipeline
from profiler import Profiler
from stream_readers import V4L2StreamReader, FileReader
from camera_manager import CameraManager

LOGGER = logging.getLogger("app")
CONFIG_FILEPATH = "/app/config.json"
HQ_OUTPUT_DIR = "/app/output_hq"
MODEL_DIR = "/app/model"

app = Flask(__name__)

# Global runtime state
pipeline = None  # type: StreamPipeline | None
profiler = None  # type: Profiler | None
current_config = {}
last_error = None  # type: str | None
pipeline_id = None  # type: str | None


def is_pipeline_running():
    return any(pipeline_thread_states())


def pipeline_thread_states():
    """Return True if there is an active StreamPipeline with any live thread."""
    global pipeline
    if pipeline is None:
        return [False, False, False]

    threads = getattr(pipeline, "_threads", None)
    if not threads:
        return [False, False, False]

    return [t.is_alive() for t in threads] 


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/results")
def results_page():
    return send_file("results.html")


@app.route("/api/config")
def api_config():
    return jsonify({"stream_port": 8889})


@app.route("/api/models")
def api_models():
    """List available *.engine models in /model/ul and /model/rf."""
    try:
        models = []
        # Search in UL and RF folders for engine files
        search_paths = [
            os.path.join(MODEL_DIR, "ul", "*fp16.engine"),
            os.path.join(MODEL_DIR, "rf", "*.engine") 
        ]
        
        for p in search_paths:
            files = glob.glob(p)
            for f in files:
                # Get relative path or clear name
                # We categorize them by parent folder name (ul or rf)
                parent = os.path.basename(os.path.dirname(f))
                name = os.path.basename(f)
                
                models.append({
                    "path": f,
                    "name": name,
                    "type": parent, # 'ul' or 'rf'
                    "display": f"[{parent.upper()}] {name}"
                })
        
        return jsonify({"models": models})
    except Exception as e:
        LOGGER.error(f"Failed to list models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/cameras")
def api_cameras():
    """List attached cameras and their modes."""
    try:
        cams = CameraManager.get_available_cameras()
        return jsonify({"cameras": cams})
    except Exception as e:
        LOGGER.error(f"Failed to list cameras: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def api_status():
    global pipeline_id
    names = ["capture", "inference", "output"]

    states = pipeline_thread_states()
    live_threads = [names[idx] for idx, alive in enumerate(states) if alive]
    state = "running" if any(states) else "idle"
    
    # Return the full current config so UI can show resolution/fps
    video_desc = current_config.get("video_description", current_config.get("video", ""))
    pid_value = pipeline_id if pipeline and is_pipeline_running() else "-"
    
    # Add current model to status if running
    current_model = current_config.get("model_path", "")

    return jsonify({
        "state": state,
        "pipeline_id": pid_value,
        "last_error": last_error,
        "config": {
            "video_reference": video_desc,
            "model": os.path.basename(current_model) if current_model else ""
        },
        "threads": live_threads,
    })


@app.route("/api/results")
def api_list_results():
    """List grouped MKV and CSV files from the output directory."""
    try:
        if not os.path.exists(HQ_OUTPUT_DIR):
            return jsonify({"results": []})

        # Find all MKV files
        mkv_files = glob.glob(os.path.join(HQ_OUTPUT_DIR, "*.mkv"))
        results_map = {}

        for mkv_path in mkv_files:
            filename = os.path.basename(mkv_path)
            # Expected format: output-hq-{pipeline_id}.mkv
            if filename.startswith("output-hq-") and filename.endswith(".mkv"):
                pid = filename[10:-4]  # Extract ID
                
                # Check for corresponding CSV
                csv_filename = f"output-hq-{pid}.csv"
                csv_path = os.path.join(HQ_OUTPUT_DIR, csv_filename)
                
                # Get file stats (size and time)
                stat = os.stat(mkv_path)
                creation_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                size_mb = round(stat.st_size / (1024 * 1024), 2)

                results_map[pid] = {
                    "id": pid,
                    "timestamp": creation_time,
                    "video": filename,
                    "video_size": f"{size_mb} MB",
                    "csv": csv_filename if os.path.exists(csv_path) else None
                }

        # Convert to list and sort by timestamp descending
        results_list = list(results_map.values())
        results_list.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({"results": results_list})

    except Exception as e:
        LOGGER.error(f"Failed to list results: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<pid>", methods=["DELETE"])
def api_delete_result(pid):
    """Delete the MKV and CSV associated with a specific Analysis ID (pid)."""
    try:
        if not pid or ".." in pid or "/" in pid:
            return jsonify({"error": "Invalid ID"}), 400

        mkv_filename = f"output-hq-{pid}.mkv"
        csv_filename = f"output-hq-{pid}.csv"
        
        mkv_path = os.path.join(HQ_OUTPUT_DIR, mkv_filename)
        csv_path = os.path.join(HQ_OUTPUT_DIR, csv_filename)
        
        deleted_files = []

        if os.path.exists(mkv_path):
            os.remove(mkv_path)
            deleted_files.append(mkv_filename)
        
        if os.path.exists(csv_path):
            os.remove(csv_path)
            deleted_files.append(csv_filename)
            
        LOGGER.info(f"Deleted results for ID {pid}: {deleted_files}")
        return jsonify({"success": True, "deleted": deleted_files})

    except Exception as e:
        LOGGER.error(f"Failed to delete results for {pid}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/download/<path:filename>")
def download_file(filename):
    """Download a file from the HQ output directory."""
    try:
        return send_from_directory(HQ_OUTPUT_DIR, filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/start", methods=["POST"])
def api_start():
    global pipeline, profiler, last_error, current_config, pipeline_id

    # Don't allow concurrent runs
    if is_pipeline_running():
        return jsonify({"error": "already running"}), 400

    # Clean up stale pipeline object
    if pipeline is not None and not is_pipeline_running():
        try:
            pipeline.__exit__(None, None, None)
        except Exception:
            LOGGER.exception("Error cleaning up stale pipeline")
        finally:
            pipeline = None

    last_error = None
    tmp_pipeline = None

    try:
        data = request.json or {}
        
        # determine source type
        source_type = data.get("source_type", "file") # 'camera' or 'file'
        
        # Default Args
        args_dict = dict()
        if os.path.exists(CONFIG_FILEPATH):
            with open(CONFIG_FILEPATH, 'r') as config_input:
                args_dict = json.load(config_input)

        # ---------------------------------------------------------------------
        # Analysis Number Logic
        # ---------------------------------------------------------------------
        analysis_num = data.get("analysis_number", "").strip()
        if analysis_num:
            pipeline_id = analysis_num
        else:
            start_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pipeline_id = f"{start_time_str}-{uuid.uuid4().hex}"
        
        # ---------------------------------------------------------------------
        # Model Selection Logic
        # ---------------------------------------------------------------------
        # Check if user provided a specific model path
        model_path = data.get("model_path", None)
        
        # If no model provided, fallback to default (hardcoded) or check default path
        if not model_path:
            # Fallback default
            model_path = "/app/model/weights-fp16.engine"
            
        # Verify existence
        if not os.path.exists(model_path):
             LOGGER.warning(f"Selected model not found: {model_path}. attempting to find any engine...")
             # Fallback logic could go here
        
        # NEW: Extract confidence from request (default to 0.75 if missing)
        requested_conf = float(data.get("vis_conf", 0.75))
        
        # Common defaults
        args_dict.update(dict(
            print_every=60,
            stream_host="127.0.0.1",
            stream_port=8554,
            log_level="INFO",
            output_path="pub-output",
            model_conf=0.10,
            vis_conf=requested_conf,
            pipeline_id=pipeline_id,
            hq_output_dir=HQ_OUTPUT_DIR,
            output_stream='WebRTC',
            class_names=['Background', 'CouldBeTilletia', 'Tilletia'],
            model_path=model_path # PASS MODEL PATH TO ARGS
        ))

        if source_type == "camera":
            device = data.get("device")
            width = int(data.get("width", 1280))
            height = int(data.get("height", 720))
            fps = int(data.get("fps", 30))
            pixel_format = data.get("format", "MJPG")
            
            if not device:
                raise ValueError("No device selected")

            # Update args
            args_dict["mode"] = "v4l2-gs"
            args_dict["device"] = device
            args_dict["width"] = width
            args_dict["height"] = height
            args_dict["fps"] = fps
            args_dict["pixel_format"] = pixel_format
            
            video_desc = f"{device} ({width}x{height} @ {fps}fps {pixel_format})"
            
            # Initialize Reader with explicit caps
            args = argparse.Namespace(**args_dict)
            reader = V4L2StreamReader(args.device, args.width, args.height, args.fps, pixel_format=pixel_format)

        else:
            # File Mode
            video = data.get("video")
            if not video:
                raise ValueError("No file provided")
                
            if not os.path.isfile(video):
                raise ValueError("File not found: %s" % video)
                
            args_dict["mode"] = "file"
            video_desc = os.path.basename(video)
            
            # Defaults for file, will be overwritten by probe
            args_dict["fps"] = 30 
            
            args = argparse.Namespace(**args_dict)
            reader = FileReader(video, args.fps)
            
            # Probe file properties
            with reader:
                cap = reader.cap
                if cap is None or not cap.isOpened():
                    raise RuntimeError("Failed to open file")
                
                args.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                args.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                f_fps = cap.get(cv2.CAP_PROP_FPS)
                if f_fps > 0:
                    args.fps = int(f_fps)
                LOGGER.info(f"File probed: {args.width}x{args.height} @ {args.fps}")

        profiler = Profiler(window=200)

        LOGGER.info(f"Starting pipeline: {args}")
        tmp_pipeline = StreamPipeline(reader, profiler, args)
        tmp_pipeline.__enter__()
        pipeline = tmp_pipeline

        current_config = {
            "video_description": video_desc,
            "model_path": model_path
        }

        return jsonify({"success": True, "pipeline_id": pipeline_id})

    except Exception as e:
        last_error = str(e)
        LOGGER.exception("Failed to start pipeline: %s", e)
        if tmp_pipeline is not None:
            try:
                tmp_pipeline.__exit__(type(e), e, e.__traceback__)
            except Exception:
                pass
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global pipeline, last_error, current_config, pipeline_id

    if pipeline is None and not is_pipeline_running():
        current_config = {}
        return jsonify({"success": True})

    try:
        if pipeline is not None:
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

    return jsonify({"video": path})


@app.route("/<path:path>/whep", methods=["GET", "POST", "OPTIONS"])
def proxy_whep(path):
    # Proxy WHEP requests to the local mediamtx server
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