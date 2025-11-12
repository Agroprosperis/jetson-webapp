#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

import cgi
import cv2 as cv
import numpy as np
import requests

from inference.core.utils.image_utils import load_image
from inference_sdk import InferenceHTTPClient

LOGGER = logging.getLogger("rf-app")

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(threadName)s | rf-app | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Optional[str] = None, max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=[])
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logging.getLogger().addHandler(console_handler)
    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logging.getLogger().addHandler(file_handler)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


class SharedImage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.image: Optional[np.ndarray] = None

    def set_image(self, image: Optional[np.ndarray]) -> None:
        with self.lock:
            self.image = None if image is None else image

    def get_image(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.image is None else self.image.copy()

    def clear_image(self) -> None:
        self.set_image(None)


def prepare_rtsp_url(url: str, transport: Optional[str]) -> str:
    if not url.lower().startswith("rtsp://") or not transport:
        return url
    if transport not in ("tcp", "udp"):
        return url
    parsed_url = urlparse(url)
    query_params = dict(parse_qsl(parsed_url.query, keep_blank_values=True))
    query_params.setdefault("rtsp_transport", transport)
    return urlunparse(parsed_url._replace(query=urlencode(query_params)))


def extract_pipeline_id(response: Any) -> Optional[str]:
    def dfs_search(obj: Any) -> Optional[str]:
        if isinstance(obj, dict):
            value = obj.get("pipeline_id") or obj.get("id")
            if isinstance(value, str) and value:
                return value
            for val in obj.values():
                result = dfs_search(val)
                if result:
                    return result
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                result = dfs_search(item)
                if result:
                    return result
        return None

    return dfs_search(response)


def list_pipeline_ids(client: InferenceHTTPClient) -> List[str]:
    try:
        response = client.list_inference_pipelines()
        pipelines = response.get("pipelines", [])
        LOGGER.debug("[list_ids] Found pipelines: %s", pipelines)
        return pipelines
    except Exception as error:
        LOGGER.debug("list_inference_pipelines failed: %s", error)
        return []


def await_new_pipeline(client: InferenceHTTPClient, existing_ids: set, timeout_seconds: float = 10.0, check_interval: float = 0.25) -> Optional[str]:
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            current_ids = set(list_pipeline_ids(client))
            new_ids = [pipeline_id for pipeline_id in current_ids if pipeline_id and pipeline_id not in existing_ids]
            if new_ids:
                LOGGER.debug("[await] Found new pipeline: %s", new_ids[0])
                return new_ids[0]
        except Exception as error:
            LOGGER.debug("[mgr] await_new_pipeline exception: %s", error)
        time.sleep(check_interval)
    LOGGER.debug("[await] No new pipeline found within timeout")
    return None


def process_pipeline_result(result: Dict[str, Any], csv_writer: csv.writer, timestamp: float, excluded_fields: List[str]) -> None:
    log_result = {key: value for key, value in result.items() if key not in excluded_fields}
    LOGGER.debug(f"[poll] succeeded: {log_result}")

    predictions = result.get("predictions", {}).get('predictions', [])
    total_unique_objects_count = result.get("total_unique_objects_count", 0)
    if predictions:
        for prediction in predictions:
            if not isinstance(prediction, dict):
                LOGGER.warning("[poll] prediction not dict: %r", prediction)
                continue
            class_name = prediction.get('class')
            class_id = prediction.get('class_id')
            confidence = prediction.get('confidence')
            bbox = prediction.get('bbox', {}) or {}
            x_coord = bbox.get('x')
            y_coord = bbox.get('y')
            width = bbox.get('width')
            height = bbox.get('height')
            csv_writer.writerow([timestamp, class_name, class_id, confidence, x_coord, y_coord, width, height, total_unique_objects_count])


def update_shared_image(result: Dict[str, Any], shared_image: SharedImage) -> None:
    rendered_output = result.get('rendered_output_hq')
    if rendered_output is None:
        LOGGER.warning("[poll] no image")
        return

    image, _ = load_image(rendered_output)
    if image is None:
        LOGGER.warning("[poll] load_image failed for value=%r", rendered_output)
        return

    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    shared_image.set_image(image)


def poll_worker(client: InferenceHTTPClient, pipeline_id: str, shared_image: SharedImage, stop_event: threading.Event, excluded_fields: Optional[List[str]] = None) -> None:
    LOGGER.info("[poll] start pipeline_id=%s excluded=%s", pipeline_id, excluded_fields)
    csv_path = f"{pipeline_id}.csv"
    header = ['timestamp', 'class', 'class_id', 'confidence', 'x', 'y', 'width', 'height', 'total_unique_objects_count']
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(header)

    while not stop_event.is_set():
        try:
            response = client.consume_inference_pipeline_result(pipeline_id=pipeline_id, excluded_fields=excluded_fields or [])
            outputs = response.get("outputs", []) or []
            if len(outputs) == 0:
                time.sleep(0.05)
                continue

            single_result = outputs[0]
            timestamp = time.time()
            with open(csv_path, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                process_pipeline_result(single_result, writer, timestamp, ['rendered_output_hq', 'draw_custom_label', 'detected_frames'])

            update_shared_image(single_result, shared_image)
        except Exception as error:
            LOGGER.warning("[poll] exception: %s", error)
            time.sleep(0.01)
    shared_image.clear_image()
    LOGGER.info("[poll] stop")


class PipelineManager:
    # States: idle, starting, running, stopping
    def __init__(self, client: InferenceHTTPClient, workflow_shared_image: SharedImage, base_config: Dict[str, Any]) -> None:
        self.client = client
        self.workflow_shared_image = workflow_shared_image
        self.base_config = dict(base_config)
        self.lock = threading.Lock()
        self.state = "idle"
        self.pipeline_id: Optional[str] = None
        self.config: Dict[str, Any] = {}
        self.poll_thread: Optional[threading.Thread] = None
        self.poll_stop_event: Optional[threading.Event] = None
        self.last_error: Optional[str] = None
        self.cancel_event = threading.Event()

    def set_state(self, new_state: str) -> None:
        if new_state != self.state:
            LOGGER.info("[mgr] state %s -> %s", self.state, new_state)
            self.state = new_state

    def configure_mediamtx_for_external_rtsp(self, video_reference: str) -> str:
        parsed_url = urlparse(video_reference)
        if parsed_url.hostname not in ("127.0.0.1", "localhost", "::1"):
            pathname = parsed_url.path.strip("/").split("/")[-1] or "proxy"
            mediamtx_api_base = self.base_config["mediamtx_api"].rstrip("/")
            remove_api = f"{mediamtx_api_base}/v3/config/paths/remove/{pathname}"
            add_api = f"{mediamtx_api_base}/v3/config/paths/add/{pathname}"
            patch_api = f"{mediamtx_api_base}/v3/config/paths/patch/{pathname}"
            mtx_config = {"source": video_reference}

            try:
                response = requests.delete(remove_api, timeout=5)
                LOGGER.info("[start] Attempted to remove existing path '%s': status %s", pathname, response.status_code)
            except Exception as error:
                LOGGER.debug("[start] Ignore remove error for path '%s': %s", pathname, error)

            try:
                LOGGER.info("[start] Attempting to add mediamtx path '%s'", mtx_config)
                response = requests.post(add_api, json=mtx_config, timeout=5)
                if response.status_code not in (200, 201):
                    if response.status_code == 400 and "already exists" in response.text.lower():
                        LOGGER.info("[start] Add failed due to existing, attempting patch")
                        response = requests.patch(patch_api, json=mtx_config, timeout=5)
                        if response.status_code not in (200, 201):
                            raise Exception(f"MediaMTX PATCH returned {response.status_code}: {response.text}")
                        LOGGER.info("[start] Patched existing RTSP path '%s' to MediaMTX: status %s", pathname, response.status_code)
                    else:
                        raise Exception(f"MediaMTX ADD returned {response.status_code}: {response.text}")
                else:
                    LOGGER.info("[start] Added external RTSP path '%s' to MediaMTX: status %s", pathname, response.status_code)
            except Exception as error:
                LOGGER.warning("[start] Failed to add/patch path to MediaMTX: %s", error)

            local_video_ref = f"rtsp://127.0.0.1:8554/{pathname}"
            return local_video_ref
        return video_reference

    def start_internal(self, config: Dict[str, Any], cancel_event: threading.Event) -> None:
        with self.lock:
            if self.pipeline_id:
                self.stop_locked(graceful=True)
            self.set_state("starting")
            self.last_error = None
        LOGGER.info("[start] Pipeline start initiated with internal config: %s", json.dumps(config, indent=2))

        existing_ids = set(list_pipeline_ids(self.client))
        LOGGER.info("[start] Pipelines before start: %s", existing_ids)
        video_reference = config["video_reference"]
        LOGGER.info("[start] Using video_reference: %s", video_reference)

        video_reference = self.configure_mediamtx_for_external_rtsp(video_reference)

        results_buffer_size = int(config.get("results_buffer_size", 1))
        batch_collection_timeout = float(config.get("batch_collection_timeout", 0.03))
        api_params = {
            "video_reference": [video_reference],
            "workspace_name": self.base_config["workspace_id"],
            "workflow_id": self.base_config["workflow_id"],
            "results_buffer_size": max(1, results_buffer_size),
            "batch_collection_timeout": batch_collection_timeout,
        }

        parsed_url = urlparse(video_reference)
        if parsed_url.scheme.lower() == "rtsp" and parsed_url.hostname in ("127.0.0.1", "localhost", "::1"):
            pathname = parsed_url.path.strip("/").split("/")[-1] or "proxy"
            config["pathname"] = pathname
            LOGGER.info("[start] Parsed pathname from video_reference: %s", pathname)

        LOGGER.info("[start] Exact API params for start_inference_pipeline_with_workflow: %s", json.dumps(api_params, indent=2))
        LOGGER.info("[start] Initiating pipeline start call...")

        try:
            LOGGER.debug("[start] Executing start_inference_pipeline_with_workflow...")
            response = self.client.start_inference_pipeline_with_workflow(**api_params)
            LOGGER.info("[start] Pipeline start call completed successfully, response: %s", response)
        except Exception as error:
            self.last_error = f"start failed: {error}"
            LOGGER.error("[start] %s\n%s", self.last_error, traceback.format_exc())
            self.set_state("idle")
            return

        pipeline_id = extract_pipeline_id(response) or await_new_pipeline(self.client, existing_ids, 10.0, 0.25)
        if not pipeline_id:
            self.last_error = "no pipeline_id from server"
            LOGGER.error("[start] %s", self.last_error)
            self.set_state("idle")
            return

        LOGGER.info("[start] Pipeline ID obtained: %s", pipeline_id)

        if cancel_event.is_set():
            try:
                self.client.terminate_inference_pipeline(pipeline_id=pipeline_id)
            except Exception as error:
                LOGGER.debug("[mgr] terminate after-cancel failed: %s", error)
            self.set_state("idle")
            return

        stop_event = threading.Event()
        poll_thread = threading.Thread(
            target=poll_worker,
            name="Poller",
            daemon=True,
            kwargs=dict(client=self.client, pipeline_id=pipeline_id, shared_image=self.workflow_shared_image, stop_event=stop_event, excluded_fields=config.get("excluded_fields"))
        )
        poll_thread.start()

        with self.lock:
            self.pipeline_id = pipeline_id
            self.config = dict(config)
            self.poll_thread = poll_thread
            self.poll_stop_event = stop_event

        self.set_state("running")

    def stop_locked(self, graceful: bool) -> None:
        self.set_state("stopping")
        try:
            if self.poll_stop_event:
                self.poll_stop_event.set()
        except Exception:
            pass
        if graceful and self.poll_thread:
            try:
                self.poll_thread.join(timeout=2.0)
            except Exception:
                pass
        self.poll_thread = None
        self.poll_stop_event = None
        self.workflow_shared_image.clear_image()
        if self.pipeline_id:
            try:
                active_ids = list_pipeline_ids(self.client)
                if self.pipeline_id in active_ids:
                    self.client.terminate_inference_pipeline(pipeline_id=self.pipeline_id)
                    LOGGER.info("[mgr] terminated pipeline id=%s", self.pipeline_id)
                else:
                    LOGGER.info("[mgr] pipeline id=%s already terminated", self.pipeline_id)
            except Exception as error:
                LOGGER.warning("[mgr] terminate failed: %s", error)
        if (pathname := self.config.get("pathname")):
            mediamtx_api_base = self.base_config["mediamtx_api"].rstrip("/")
            remove_api = f"{mediamtx_api_base}/v3/config/paths/remove/{pathname}"
            try:
                response = requests.delete(remove_api, timeout=5)
                LOGGER.info("[stop] Removed RTSP path '%s' from MediaMTX: status %s", pathname, response.status_code)
            except Exception as error:
                LOGGER.warning("[stop] Failed to remove path from MediaMTX: %s", error)
        self.pipeline_id = None
        self.set_state("idle")

    def start_async(self, config: Dict[str, Any]) -> None:
        with self.lock:
            if self.state in ("starting", "stopping"):
                self.cancel_event.set()
            self.cancel_event.clear()
            threading.Thread(target=self.start_internal, name="StartAsync", daemon=True, args=(dict(config), self.cancel_event)).start()

    def stop_async(self, graceful: bool = True) -> None:
        with self.lock:
            self.cancel_event.set()
            threading.Thread(target=self.stop_locked, name="StopAsync", daemon=True, args=(graceful,)).start()

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "running": self.state == "running",
                "pipeline_id": self.pipeline_id,
                "config": dict(self.config) if self.pipeline_id else None,
                "workspace_id": self.base_config["workspace_id"],
                "workflow_id": self.base_config["workflow_id"],
                "last_error": self.last_error,
            }


class StreamServer(HTTPServer):
    def __init__(self, address: Tuple[str, int], handler_class: type, max_streams: int = 4) -> None:
        super().__init__(address, handler_class)
        self.max_streams = max_streams
        self.current_streams = 0
        self.lock = threading.Lock()

    def can_open_stream(self) -> bool:
        with self.lock:
            if self.current_streams >= self.max_streams:
                return False
            self.current_streams += 1
            return True

    def stream_closed(self) -> None:
        with self.lock:
            self.current_streams = max(0, self.current_streams - 1)


class StreamHandler(BaseHTTPRequestHandler):
    workflow_shared_image: SharedImage = None
    server_instance: StreamServer = None

    def log_message(self, format_str: str, *args: Any) -> None:
        LOGGER.debug("stream: " + format_str, *args)

    def parse_query_params(self) -> Tuple[int, int, int, int]:
        from urllib.parse import parse_qs, urlparse
        query_string = parse_qs(urlparse(self.path).query)
        def get_int_param(key: str, default: int) -> int:
            try:
                return int(query_string.get(key, [str(default)])[0])
            except Exception:
                return default
        width = get_int_param("w", 0)
        height = get_int_param("h", 0)
        quality = max(10, min(95, get_int_param("q", 85)))
        fps = max(1, min(60, get_int_param("fps", 20)))
        return width, height, quality, fps

    def write_headers(self) -> None:
        self.wfile.write(b"HTTP/1.1 200 OK\r\n")
        self.wfile.write(b"Cache-Control: no-cache, no-store, must-revalidate\r\n")
        self.wfile.write(b"Pragma: no-cache\r\n")
        self.wfile.write(b"Expires: 0\r\n")
        self.wfile.write(b"Connection: close\r\n")
        self.wfile.write(b"Access-Control-Allow-Origin: *\r\n")
        self.wfile.write(b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n")
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def encode_jpeg(self, image: np.ndarray, quality: int) -> Optional[bytes]:
        success, encoded = cv.imencode(".jpg", image, [int(cv.IMWRITE_JPEG_QUALITY), int(quality), int(cv.IMWRITE_JPEG_OPTIMIZE), 1])
        return encoded.tobytes() if success else None

    def scale_image(self, image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        if target_width <= 0 or target_height <= 0:
            return image
        height, width = image.shape[:2]
        scale = min(target_width / max(1, width), target_height / max(1, height))
        new_width, new_height = max(1, int(width * scale)), max(1, int(height * scale))
        if (new_width, new_height) == (width, height):
            return image
        interpolation = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
        return cv.resize(image, (new_width, new_height), interpolation=interpolation)

    def write_jpeg_bytes(self, jpeg_data: bytes) -> None:
        self.wfile.write(b"--frame\r\n")
        self.wfile.write(b"Content-Type: image/jpeg\r\n")
        self.wfile.write(b"Content-Length: " + str(len(jpeg_data)).encode("ascii") + b"\r\n\r\n")
        self.wfile.write(jpeg_data + b"\r\n")
        self.wfile.flush()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if not self.server_instance.can_open_stream():
            self.send_response(503)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"stream capacity reached")
            return
        try:
            if path.startswith("/workflow.mjpg"):
                target_width, target_height, quality, fps = self.parse_query_params()
                frame_period = 1.0 / float(max(1, fps))
                self.write_headers()
                placeholder = np.zeros((240, 320, 3), np.uint8)
                cv.putText(placeholder, "Loading workflow...", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                try:
                    while True:
                        image = self.workflow_shared_image.get_image()
                        if image is None:
                            display_image = placeholder.copy()
                        else:
                            display_image = image
                        if target_width > 0 and target_height > 0:
                            display_image = self.scale_image(display_image, target_width, target_height)
                        jpeg_data = self.encode_jpeg(display_image, quality)
                        if jpeg_data is None:
                            LOGGER.warning("[stream] JPEG encode failed")
                            time.sleep(0.01)
                            continue
                        self.write_jpeg_bytes(jpeg_data)
                        time.sleep(frame_period)
                except (BrokenPipeError, ConnectionResetError):
                    LOGGER.info("close workflow.mjpg")
                except Exception as error:
                    LOGGER.warning("workflow.mjpg error: %s", error)
            else:
                self.send_response(404)
                self.end_headers()
        finally:
            self.server_instance.stream_closed()


class ControlServer(HTTPServer):
    def __init__(self, address: Tuple[str, int], handler_class: type, max_workers: int = 3) -> None:
        super().__init__(address, handler_class)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="httpC")
        self.shutting_down = False
        LOGGER.info("HTTP control workers=%d", max_workers)

    def process_request(self, request: Any, client_address: Tuple[str, int]) -> None:
        if self.shutting_down:
            self.shutdown_request(request)
            return
        self.executor.submit(self.handle_request, request, client_address)

    def handle_request(self, request: Any, client_address: Tuple[str, int]) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def server_close(self) -> None:
        self.shutting_down = True
        try:
            super().server_close()
        finally:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass


class ControlHandler(BaseHTTPRequestHandler):
    pipeline_manager: PipelineManager = None
    stream_port: int = None
    mediamtx_http: str = None
    mediamtx_whep_path: str = None
    mediamtx_rtsp_host: str = None
    upload_rtsp_path: str = "uploaded"
    ffmpeg_process: Optional[subprocess.Popen] = None

    def log_message(self, format_str: str, *args: Any) -> None:
        LOGGER.debug("http: " + format_str, *args)

    def clean_path(self) -> str:
        return self.path.split("?", 1)[0]

    def send_json_response(self, data: Any, status_code: int = 200) -> None:
        json_data = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(json_data)))
        self.end_headers()
        self.wfile.write(json_data)

    def read_request_body(self) -> bytes:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            return self.rfile.read(content_length) if content_length > 0 else b""
        except Exception:
            return b""

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        finally:
            try:
                LOGGER.debug("HTTP %s %s", getattr(self, "command", "?"), getattr(self, "path", "?"))
            except Exception:
                pass

    def do_OPTIONS(self) -> None:
        path = self.clean_path()
        if path.startswith("/whep/"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.end_headers()
            return
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        path = self.clean_path()
        if path.startswith("/whep/"):
            self.send_response(405)
            self.send_header("Allow", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return

        if path in ("/", "/index.html"):
            html_content = (
                b"<!doctype html><html><head><meta charset='utf-8'/>"
                b"<title>RF App (WebRTC Original + MJPEG Workflow)</title>"
                b"<meta name='viewport' content='width=device-width,initial-scale=1'/>"
                b"<style>"
                b"body{margin:0;background:#111;color:#eee;font-family:system-ui,Segoe UI,Roboto,Arial}"
                b"header{padding:12px 14px;background:#1b1b1b;display:flex;gap:12px;align-items:center;flex-wrap:wrap}"
                b"input,select,button{padding:8px 10px;border-radius:8px;border:1px solid #333;background:#222;color:#eee}"
                b"button{cursor:pointer;transition:transform .06s ease,opacity .2s}"
                b"button:active{transform:scale(0.98)}"
                b"button[disabled]{opacity:.5;cursor:not-allowed}"
                b".status{font-size:12px;opacity:.9;margin-left:auto;padding:4px 8px;border-radius:8px;background:#222}"
                b".grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;position:fixed;inset:56px 0 0 0}"
                b".pane{display:flex;flex-direction:column;min-width:0;min-height:0}"
                b".pane h3{margin:6px 12px 4px 12px;font-weight:600;font-size:14px;opacity:.9}"
                b".pane video,.pane img{flex:1;min-height:0;width:100%;height:100%;object-fit:contain;background:#000}"
                b"@media(max-width:1000px){.grid{grid-template-columns:1fr}}"
                b"</style></head><body>"
                b"<header>"
                b"<form id='cfg' onsubmit='return false;'>"
                b"<input id='video' type='text' placeholder='rtsp://127.0.0.1:8554/<path> (or file, or 0)' size='48'/>"
                b"<select id='rtsp_transport'><option value=''>auto</option>"
                b"<option value='tcp'>rtsp_transport=tcp</option><option value='udp'>rtsp_transport=udp</option></select>"
                b"<button id='run'>Run</button>"
                b"<button id='stop' type='button'>Stop</button>"
                b"<button id='uploadBtn'>Upload Video</button>"
                b"<input id='upload' type='file' accept='video/*' style='display:none'/>"
                b"</form>"
                b"<div class='status'><span id='status'>status: idle</span></div>"
                b"</header>"
                b"<div class='grid'>"
                b"<div class='pane'><h3>Original (WebRTC via MediaMTX)</h3><video id='orig' playsinline autoplay muted controls></video></div>"
                b"<div class='pane'><h3>Workflow (MJPEG)</h3><img id='wf'/></div>"
                b"</div>"
                b"<script>"
                b"let pc=null; let cfg=null; let state='idle';"
                b"let isUploading=false; let uploadProgress=0;"
                b"const statusEl=document.getElementById('status');"
                b"const btnRun=document.getElementById('run'); const btnStop=document.getElementById('stop');"
                b"const uploadBtn=document.getElementById('uploadBtn'); const uploadInput=document.getElementById('upload');"
                b"const vOrig=document.getElementById('orig'); const imgWf=document.getElementById('wf');"
                b"function applyState(j){"
                b"  if(isUploading){"
                b"    btnRun.disabled=true; btnStop.disabled=true; uploadBtn.disabled=true;"
                b"    statusEl.textContent=`status: uploading ${uploadProgress}%`;"
                b"    return;"
                b"  }"
                b"  state=(j&&j.state)||'idle';"
                b"  btnRun.disabled=state==='starting'||state==='running';"
                b"  btnRun.textContent=(state==='running')?'Running':((state==='starting')?'Starting...':'Run');"
                b"  btnStop.disabled=state==='idle'||state==='stopping';"
                b"  uploadBtn.disabled=false;"
                b"  let msg='status: '+state+' pid:' + ((j&&j.pipeline_id)||'-');"
                b"  if(j&&j.last_error&&state!=='running') msg+=' fail: '+j.last_error;"
                b"  statusEl.textContent=msg;"
                b"}"
                b"async function fetchCfg(){ const r=await fetch('/api/config'); cfg=await r.json(); }"
                b"async function refresh(){"
                b"  try{ const r=await fetch('/api/status'); const j=await r.json(); applyState(j); return j; }catch(e){ statusEl.textContent='status: error'; return null; }"
                b"}"
                b"async function connectWHEP(rtspUrl){"
                b"  if(!cfg) await fetchCfg();"
                b"  let path = cfg.mediamtx_whep_path;  /* default */"
                b"  try{"
                b"    const raw=(rtspUrl||'').trim();"
                b"    const m = raw.match(/^rtsp:\\/\\/[^/]+\\/(.+?)(?:\\?|$)/i);"
                b"    if(m && m[1]){ path = m[1].split('/').filter(Boolean).pop(); }"
                b"  }catch(e){ }"
                b"  if(!path){ path = cfg.mediamtx_whep_path || 'original'; }"
                b"  const whep = '/whep/' + encodeURIComponent(path);"
                b"  try{"
                b"    if(pc) { try{pc.close();}catch(e){} pc=null; }"
                b"    vOrig.srcObject=null;"
                b"    pc=new RTCPeerConnection({iceServers:[{urls:['stun:stun.l.google.com:19302']}]});"
                b"    pc.addEventListener('track',(ev)=>{ vOrig.srcObject=ev.streams[0]; });"
                b"    const offer=await pc.createOffer({offerToReceiveVideo:true, offerToReceiveAudio:false});"
                b"    await pc.setLocalDescription(offer);"
                b"    const rr=await fetch(whep,{method:'POST',headers:{'Content-Type':'application/sdp'},body:offer.sdp});"
                b"    if(!rr.ok){ throw new Error('WHEP HTTP '+rr.status); }"
                b"    const answer=await rr.text();"
                b"    await pc.setRemoteDescription({type:'answer', sdp:answer});"
                b"  }catch(e){ throw new Error('webrtc failed: '+(e&&e.message?e.message:e)); }"
                b"}"
                b"function disconnectWHEP(){ try{ if(pc){ pc.close(); } }catch(e){} pc=null; vOrig.srcObject=null; }"
                b"btnRun.onclick=async()=>{"
                b"  const video=document.getElementById('video').value.trim();"
                b"  if(!video){alert('Enter RTSP URL');return;}"
                b"  const rtsp_transport=document.getElementById('rtsp_transport').value;"
                b"  try{"
                b"    statusEl.textContent='status: starting workflow pipeline...';"
                b"    const startRes=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video,rtsp_transport})});"
                b"    if(!startRes.ok){ throw new Error('start failed: HTTP '+startRes.status); }"
                b"    await refresh();"
                b"    statusEl.textContent='status: starting original preview...';"
                b"    await connectWHEP(video);"
                b"  }catch(e){ statusEl.textContent='status: failed -> '+e; disconnectWHEP(); }"
                b"};"
                b"btnStop.onclick=async()=>{"
                b"  try{"
                b"    statusEl.textContent='status: stopping...';"
                b"    disconnectWHEP();"
                b"    const stopRes=await fetch('/api/stop',{method:'POST'});"
                b"    if(!stopRes.ok){ throw new Error('stop failed: HTTP '+stopRes.status); }"
                b"    await refresh();"
                b"  }catch(e){ statusEl.textContent='status: stop error -> '+e; }"
                b"};"
                b"uploadBtn.onclick=()=>{ uploadInput.click(); };"
                b"uploadInput.onchange=function(){"
                b"  if(!uploadInput.files.length) return;"
                b"  if(isUploading) return;"
                b"  const file=uploadInput.files[0];"
                b"  const fd=new FormData(); fd.append('file',file);"
                b"  const xhr=new XMLHttpRequest();"
                b"  xhr.open('POST','/api/upload');"
                b"  xhr.upload.onprogress=function(e){"
                b"    if(e.lengthComputable){"
                b"      uploadProgress=Math.round((e.loaded/e.total)*100);"
                b"      applyState();"
                b"    }"
                b"  };"
                b"  xhr.onload=function(){"
                b"    isUploading=false;"
                b"    if(xhr.status!==200){"
                b"      statusEl.textContent='upload error: '+xhr.status;"
                b"      refresh();"
                b"      return;"
                b"    }"
                b"    try{"
                b"      const j=JSON.parse(xhr.responseText);"
                b"      document.getElementById('video').value=j.video;"
                b"      document.getElementById('rtsp_transport').value=j.rtsp_transport||'';"
                b"      uploadInput.value='';"
                b"      btnRun.click();"
                b"    }catch(e){"
                b"      statusEl.textContent='upload error: '+e;"
                b"      refresh();"
                b"    }"
                b"  };"
                b"  xhr.onerror=function(){"
                b"    isUploading=false;"
                b"    statusEl.textContent='upload error';"
                b"    refresh();"
                b"  };"
                b"  isUploading=true; uploadProgress=0; applyState();"
                b"  xhr.send(fd);"
                b"};"
                b"window.addEventListener('load', async()=>{"
                b"  await fetchCfg();"
                b"  const base='http://'+location.hostname+':'+cfg.stream_port; imgWf.src=base+'/workflow.mjpg';"
                b"  let j = await refresh(); setInterval(refresh,1500);"
                b"  if(state==='running' && j && j.config && 'video_reference' in j.config){"
                b"    const video = j.config.video_reference || '';"
                b"    document.getElementById('video').value = video;"
                b"    let transport = '';"
                b"    if(video){"
                b"      try{"
                b"        const u = new URL(video);"
                b"        transport = u.searchParams.get('rtsp_transport') || '';"
                b"      }catch(e){}"
                b"    }"
                b"    document.getElementById('rtsp_transport').value = transport;"
                b"    await connectWHEP(video);"
                b"  }"
                b"});"
                b"window.addEventListener('beforeunload',()=>{ disconnectWHEP(); });"
                b"</script>"
                b"</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_content)))
            self.end_headers()
            self.wfile.write(html_content)
            return

        if path.startswith("/api/status"):
            self.send_json_response(self.pipeline_manager.status(), 200)
            return

        if path.startswith("/api/config"):
            self.send_json_response({
                "stream_port": self.stream_port,
                "mediamtx_http": self.mediamtx_http,
                "mediamtx_whep_path": self.mediamtx_whep_path
            }, 200)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        path = self.clean_path()

        if path == "/whep" or path == "/whep/":
            message = b"missing WHEP path (expected /whep/<path>)"
            self.send_response(400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            self.wfile.write(message)
            return

        if path.startswith("/whep/"):
            whep_path = unquote(path[len("/whep/"):].lstrip("/"))
            sdp_data = self.read_request_body()
            target_url = self.mediamtx_http.rstrip("/") + "/" + whep_path + "/whep"
            try:
                LOGGER.debug("WHEP proxy -> %s", target_url)
                response = requests.post(target_url, data=sdp_data, headers={"Content-Type": "application/sdp"}, timeout=10)
                body = response.text.encode("utf-8")
                self.send_response(response.status_code)
                self.send_header("Content-Type", "application/sdp")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as error:
                message = f"whep proxy error: {error}".encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
            return

        if path.startswith("/api/upload"):
            try:
                environ = {
                    'REQUEST_METHOD': self.command,
                    'CONTENT_TYPE': self.headers['Content-Type'],
                }
                form_data = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)
                if 'file' not in form_data:
                    self.send_json_response({"ok": False, "error": "no file"}, 400)
                    return
                file_item = form_data['file']
                if not file_item.filename:
                    self.send_json_response({"ok": False, "error": "no filename"}, 400)
                    return
                file_extension = os.path.splitext(file_item.filename)[1] or ".mp4"
                uploaded_file_path = f"/tmp/rf_uploaded_{int(time.time())}{file_extension}"
                with open(uploaded_file_path, 'wb') as output_file:
                    output_file.write(file_item.file.read())
                if self.ffmpeg_process:
                    try:
                        self.ffmpeg_process.terminate()
                        self.ffmpeg_process.wait(timeout=5)
                    except Exception as error:
                        LOGGER.warning("[upload] failed to terminate previous ffmpeg: %s", error)
                    self.ffmpeg_process = None
                rtsp_url = f"rtsp://{self.mediamtx_rtsp_host}/{self.upload_rtsp_path}"
                ffmpeg_command = [
                    "ffmpeg",
                    "-stream_loop", "-1",
                    "-re",
                    "-i", uploaded_file_path,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-tune", "zerolatency",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-f", "rtsp",
                    "-rtsp_transport", "tcp",
                    rtsp_url
                ]
                LOGGER.info("[upload] starting ffmpeg: %s", " ".join(ffmpeg_command))
                process = subprocess.Popen(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.ffmpeg_process = process
                if not self.is_stream_available(rtsp_url):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except Exception:
                        pass
                    try:
                        os.remove(uploaded_file_path)
                    except Exception:
                        pass
                    self.send_json_response({"ok": False, "error": "failed to start stream"}, 500)
                    return
                time.sleep(2)  # additional delay for WebRTC to be ready
                self.send_json_response({"ok": True, "video": rtsp_url, "rtsp_transport": "tcp"}, 200)
            except Exception as error:
                LOGGER.warning("[upload] error: %s", error)
                self.send_json_response({"ok": False, "error": str(error)}, 500)
            return

        if path.startswith("/api/start"):
            raw_body = self.read_request_body()
            LOGGER.debug("[api/start] Raw body: %s", raw_body)
            try:
                body = json.loads(raw_body.decode("utf-8") or "{}")
            except Exception:
                body = {}
            LOGGER.debug("[api/start] Parsed body: %s", body)
            video = str(body.get("video", "")).strip()
            rtsp_transport = body.get("rtsp_transport", None)
            LOGGER.debug("[api/start] Extracted video: %s", video)
            LOGGER.debug("[api/start] rtsp_transport: %s", rtsp_transport)
            if not video:
                self.send_json_response({"ok": False, "error": "video is required"}, 400)
                return
            try:
                video_ref = int(video)
            except Exception:
                video_ref = video
                if isinstance(video_ref, str) and video_ref.lower().startswith("rtsp://"):
                    video_ref = prepare_rtsp_url(video_ref, rtsp_transport)
            LOGGER.debug("[api/start] video_ref after prepare: %s", video_ref)
            status = self.pipeline_manager.status()
            results_buffer_size = (status.get("config") or {}).get("results_buffer_size") or 1
            batch_collection_timeout = (status.get("config") or {}).get("batch_collection_timeout") or 0.03
            excluded_fields = (status.get("config") or {}).get("excluded_fields") or []
            config = dict(
                video_reference=video_ref,
                results_buffer_size=results_buffer_size,
                batch_collection_timeout=batch_collection_timeout,
                excluded_fields=excluded_fields,
                original_video=video,
                rtsp_transport=rtsp_transport
            )
            self.pipeline_manager.start_async(config)
            self.send_json_response({"ok": True, "accepted": True}, 202)
            return

        if path.startswith("/api/stop"):
            self.pipeline_manager.stop_async(graceful=True)
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=5)
                except Exception as error:
                    LOGGER.warning("[stop] failed to terminate ffmpeg: %s", error)
                self.ffmpeg_process = None
            self.send_json_response({"ok": True, "accepted": True}, 202)
            return

        self.send_response(404)
        self.end_headers()

    def is_stream_available(self, rtsp_url: str, timeout: int = 10) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                subprocess.check_call(["ffprobe", "-v", "error", "-show_format", "-i", rtsp_url], timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                LOGGER.info("[upload] stream available at %s", rtsp_url)
                return True
            except subprocess.TimeoutExpired:
                pass
            except subprocess.CalledProcessError:
                time.sleep(0.5)
            except FileNotFoundError:
                LOGGER.error("ffprobe not found")
                return False
        LOGGER.warning("[upload] stream not available within timeout at %s", rtsp_url)
        return False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF video app: WebRTC Original via existing MediaMTX + MJPEG Workflow")
    parser.add_argument("--inference-server-url", required=True, type=str)
    parser.add_argument("--api-key", required=True, type=str)
    parser.add_argument("--workspace-id", required=True, type=str)
    parser.add_argument("--workflow-id", required=True, type=str)
    parser.add_argument("--rtsp-transport", choices=["tcp", "udp"], default=None)
    parser.add_argument("--results-buffer-size", default=4, type=int)
    parser.add_argument("--batch-collection-timeout", default=0.05, type=float)
    parser.add_argument("--http-port", default=8081, type=int)
    parser.add_argument("--http-workers", default=3, type=int)
    parser.add_argument("--stream-port", default=8082, type=int)
    parser.add_argument("--stream-max", default=4, type=int)
    parser.add_argument("--mediamtx-http", default="http://127.0.0.1:8889", type=str, help="MediaMTX HTTP base for WHEP (existing)")
    parser.add_argument("--mediamtx-api", default="http://127.0.0.1:9997", type=str, help="MediaMTX API base for config (existing)")
    parser.add_argument("--mediamtx-rtsp", default="127.0.0.1:8554", type=str, help="MediaMTX RTSP host:port for publishing (existing)")
    parser.add_argument("--mediamtx-whep-path", default="original", type=str, help="MediaMTX path already being published (matches your rtsp://...:8554/<path>)")
    parser.add_argument("--log-level", default="INFO", type=str)
    parser.add_argument("--log-file", default=None, type=str)
    parser.add_argument("--log-max-bytes", default=5 * 1024 * 1024, type=int)
    parser.add_argument("--log-backup-count", default=3, type=int)
    parser.add_argument("--exclude", default="", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    setup_logging(args.log_level, args.log_file, args.log_max_bytes, args.log_backup_count)

    LOGGER.info(
        "boot ctrl=%d stream=%d inference=%s ws=%s wf=%s mediamtx_http=%s mediamtx_api=%s whep_path=%s mediamtx_rtsp=%s",
        args.http_port, args.stream_port, args.inference_server_url,
        args.workspace_id, args.workflow_id, args.mediamtx_http, args.mediamtx_api, args.mediamtx_whep_path, args.mediamtx_rtsp
    )

    inference_client = InferenceHTTPClient(api_url=args.inference_server_url, api_key=args.api_key)
    LOGGER.info("Inference client initialized with url: %s", args.inference_server_url)
    workflow_shared_image = SharedImage()
    base_config = {
        "workspace_id": args.workspace_id,
        "workflow_id": args.workflow_id,
        "mediamtx_http": args.mediamtx_http,
        "mediamtx_api": args.mediamtx_api
    }
    pipeline_manager = PipelineManager(client=inference_client, workflow_shared_image=workflow_shared_image, base_config=base_config)

    existing_pipelines = list_pipeline_ids(inference_client)
    if existing_pipelines:
        pipeline_id = existing_pipelines[0]
        LOGGER.info("Attaching to existing pipeline: %s", pipeline_id)
        excluded_fields = [field.strip() for field in args.exclude.split(",") if field.strip()]
        video_reference = ""
        status_url = f"{args.inference_server_url}/inference_pipelines/{pipeline_id}/status"
        try:
            response = requests.get(status_url, params={"api_key": args.api_key}, timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                sources_metadata = status_data.get("report", {}).get("sources_metadata", [])
                if len(sources_metadata) > 0:
                    source_ref = sources_metadata[0].get("source_reference", "")
                    video_reference = source_ref
                    LOGGER.info("Fetched video_reference from pipeline status: %s", source_ref)
        except Exception as error:
            LOGGER.warning("Failed to fetch pipeline status for video_reference: %s", error)
        parsed_url = urlparse(video_reference)
        pathname = parsed_url.path.strip("/").split("/")[-1] or "proxy" if parsed_url.scheme.lower() == "rtsp" else None
        query_params = dict(parse_qsl(parsed_url.query))
        rtsp_transport = query_params.get("rtsp_transport")
        config = dict(
            video_reference=video_reference,
            results_buffer_size=max(1, int(args.results_buffer_size)),
            batch_collection_timeout=float(args.batch_collection_timeout),
            excluded_fields=excluded_fields,
            original_video=video_reference,
            rtsp_transport=rtsp_transport
        )
        if pathname:
            config["pathname"] = pathname
        stop_event = threading.Event()
        poll_thread = threading.Thread(
            target=poll_worker,
            name="Poller",
            daemon=True,
            kwargs=dict(client=inference_client, pipeline_id=pipeline_id, shared_image=workflow_shared_image, stop_event=stop_event, excluded_fields=config.get("excluded_fields"))
        )
        poll_thread.start()
        with pipeline_manager.lock:
            pipeline_manager.pipeline_id = pipeline_id
            pipeline_manager.config = dict(config)
            pipeline_manager.poll_thread = poll_thread
            pipeline_manager.poll_stop_event = stop_event
        pipeline_manager.set_state("running")

    stream_server = StreamServer(("0.0.0.0", int(args.stream_port)), StreamHandler, max_streams=int(args.stream_max))
    StreamHandler.workflow_shared_image = workflow_shared_image
    StreamHandler.server_instance = stream_server

    control_server = ControlServer(("0.0.0.0", int(args.http_port)), ControlHandler, max_workers=int(args.http_workers))
    ControlHandler.pipeline_manager = pipeline_manager
    ControlHandler.stream_port = int(args.stream_port)
    ControlHandler.mediamtx_http = args.mediamtx_http
    ControlHandler.mediamtx_whep_path = args.mediamtx_whep_path
    ControlHandler.mediamtx_rtsp_host = args.mediamtx_rtsp

    def serve_http() -> None:
        LOGGER.info("UI/API: http://0.0.0.0:%d/  |  API: /api/config /api/status /api/start /api/stop /api/upload | WHEP proxy: /whep/<path>", args.http_port)
        control_server.serve_forever()

    def serve_stream() -> None:
        LOGGER.info("Workflow MJPEG: http://0.0.0.0:%d/workflow.mjpg", args.stream_port)
        stream_server.serve_forever()

    http_thread = threading.Thread(target=serve_http, name="HTTP-Ctrl", daemon=True)
    stream_thread = threading.Thread(target=serve_stream, name="HTTP-Stream", daemon=True)
    http_thread.start()
    stream_thread.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested")
    finally:
        for server in [control_server, stream_server]:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        try:
            pipeline_manager.stop_async(graceful=True)
        except Exception:
            pass
        if ControlHandler.ffmpeg_process:
            try:
                ControlHandler.ffmpeg_process.terminate()
            except Exception:
                pass
        LOGGER.info("server stopped")


if __name__ == "__main__":
    main()