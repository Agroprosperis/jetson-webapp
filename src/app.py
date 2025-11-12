#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
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
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

import cgi
import cv2 as cv
import numpy as np
import requests
import shutil
import json

from mediamtx import ControlStream

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


class PipelineManager:
    # States: idle, starting, running, stopping
    def __init__(self, workflow_shared_image: SharedImage, base_config: Dict[str, Any]) -> None:
        self.workflow_shared_image = workflow_shared_image
        self.base_config = dict(base_config)
        self.lock = threading.Lock()
        self.state = "idle"
        self.config: Dict[str, Any] = {}
        self.capture_thread: Optional[threading.Thread] = None
        self.capture_stop_event: Optional[threading.Event] = None
        self.last_error: Optional[str] = None
        self.cancel_event = threading.Event()
        self.mediamtx_controller = ControlStream(api_base_url="http://127.0.0.1:9997", rtsp_output_host="127.0.0.1")

    def set_state(self, new_state: str) -> None:
        if new_state != self.state:
            LOGGER.info("[mgr] state %s -> %s", self.state, new_state)
            self.state = new_state

    def start_internal(self, config: Dict[str, Any], cancel_event: threading.Event) -> None:
        with self.lock:
            if self.state in ("running", "starting"):
                self.stop_locked(graceful=True)
            self.set_state("starting")
            self.last_error = None
        LOGGER.info("[start] Stream start initiated with config: %s", json.dumps(config, indent=2))

        video_reference = config["video_reference"]
        LOGGER.info("[start] Using video_reference: %s", video_reference)

        stream_name = "stream"
        rtsp_for_workflow = f"rtsp://127.0.0.1:8554/{stream_name}"

        try:
            if isinstance(video_reference, int):
                device = f"/dev/video{video_reference}"
                self.mediamtx_controller.start_video(device, name=stream_name)
            elif os.path.isfile(video_reference):
                abspath = os.path.abspath(video_reference)
                media_abspath = os.path.abspath(self.base_config["media_folder"])
                if not abspath.startswith(media_abspath):
                    filename = os.path.basename(video_reference)
                    dest = os.path.join(self.base_config["media_folder"], filename)
                    if os.path.exists(dest):
                        base, ext = os.path.splitext(filename)
                        filename = f"{base}_{int(time.time())}{ext}"
                        dest = os.path.join(self.base_config["media_folder"], filename)
                    shutil.copy(video_reference, dest)
                    file_path_in_media = f"/media/{filename}"
                else:
                    rel = os.path.relpath(abspath, media_abspath)
                    file_path_in_media = "/media/" + rel.replace(os.sep, "/")
                self.mediamtx_controller.start_file(file_path_in_media, name=stream_name)
            else:
                video_reference = prepare_rtsp_url(video_reference, config.get("rtsp_transport"))
                self.mediamtx_controller.start_rtsp_copy(video_reference, name=stream_name)
        except Exception as error:
            self.last_error = f"mediamtx start failed: {error}"
            LOGGER.error("[start] %s\n%s", self.last_error, traceback.format_exc())
            self.set_state("idle")
            return

        time.sleep(1.0)
        start_time = time.time()
        while time.time() - start_time < 10.0:
            try:
                subprocess.check_call(["ffprobe", "-v", "error", "-show_format", "-i", rtsp_for_workflow], timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                LOGGER.info("[start] stream available at %s", rtsp_for_workflow)
                break
            except Exception:
                time.sleep(0.5)
        else:
            self.last_error = "stream not available within timeout"
            LOGGER.error("[start] %s", self.last_error)
            self.mediamtx_controller.stop_all()
            self.set_state("idle")
            return

        if cancel_event.is_set():
            self.mediamtx_controller.stop_all()
            self.set_state("idle")
            return

        stop_event = threading.Event()
        capture_thread = threading.Thread(
            target=capture_worker,
            name="Capturer",
            daemon=True,
            kwargs=dict(rtsp_url=rtsp_for_workflow, shared_image=self.workflow_shared_image, stop_event=stop_event)
        )
        capture_thread.start()

        with self.lock:
            self.config = dict(config)
            self.capture_thread = capture_thread
            self.capture_stop_event = stop_event

        self.set_state("running")

    def stop_locked(self, graceful: bool) -> None:
        self.set_state("stopping")
        try:
            if self.capture_stop_event:
                self.capture_stop_event.set()
        except Exception:
            pass
        if graceful and self.capture_thread:
            try:
                self.capture_thread.join(timeout=2.0)
            except Exception:
                pass
        self.capture_thread = None
        self.capture_stop_event = None
        self.workflow_shared_image.clear_image()
        self.mediamtx_controller.stop_all()
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
                "config": dict(self.config) if self.state == "running" else None,
                "last_error": self.last_error,
            }


def capture_worker(rtsp_url: str, shared_image: SharedImage, stop_event: threading.Event) -> None:
    LOGGER.info("[capture] start rtsp_url=%s", rtsp_url)
    cap = cv.VideoCapture(rtsp_url)
    if not cap.isOpened():
        LOGGER.warning("[capture] failed to open %s", rtsp_url)
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            shared_image.set_image(frame)
        else:
            time.sleep(0.01)
    cap.release()
    shared_image.clear_image()
    LOGGER.info("[capture] stop")


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
                cv.putText(placeholder, "Loading stream...", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
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
    mediamtx_http = "http://127.0.0.1:8889"

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
            with open("index.html", "rb") as f:
                html_content = f.read()
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
                "whep_path": "stream"
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
                uploaded_file_path = os.path.join(self.pipeline_manager.base_config["media_folder"], f"uploaded_{int(time.time())}{file_extension}")
                with open(uploaded_file_path, 'wb') as output_file:
                    output_file.write(file_item.file.read())
                self.send_json_response({"ok": True, "video": uploaded_file_path, "rtsp_transport": ""}, 200)
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
            config = dict(
                video_reference=video_ref,
                original_video=video,
                rtsp_transport=rtsp_transport
            )
            self.pipeline_manager.start_async(config)
            self.send_json_response({"ok": True, "accepted": True}, 202)
            return

        if path.startswith("/api/stop"):
            self.pipeline_manager.stop_async(graceful=True)
            self.send_json_response({"ok": True, "accepted": True}, 202)
            return

        self.send_response(404)
        self.end_headers()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF video app: WebRTC Original via existing MediaMTX + MJPEG Stream")
    parser.add_argument("--http-port", default=8081, type=int)
    parser.add_argument("--http-workers", default=3, type=int)
    parser.add_argument("--stream-port", default=8082, type=int)
    parser.add_argument("--stream-max", default=4, type=int)
    parser.add_argument("--media-folder", default="/media", type=str, help="Folder for uploaded and local files (mounted to MediaMTX /media)")
    parser.add_argument("--log-level", default="INFO", type=str)
    parser.add_argument("--log-file", default=None, type=str)
    parser.add_argument("--log-max-bytes", default=5 * 1024 * 1024, type=int)
    parser.add_argument("--log-backup-count", default=3, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    setup_logging(args.log_level, args.log_file, args.log_max_bytes, args.log_backup_count)

    LOGGER.info(
        "boot ctrl=%d stream=%d",
        args.http_port, args.stream_port
    )

    workflow_shared_image = SharedImage()
    base_config = {
        "media_folder": args.media_folder
    }
    pipeline_manager = PipelineManager(workflow_shared_image=workflow_shared_image, base_config=base_config)

    stream_server = StreamServer(("0.0.0.0", int(args.stream_port)), StreamHandler, max_streams=int(args.stream_max))
    StreamHandler.workflow_shared_image = workflow_shared_image
    StreamHandler.server_instance = stream_server

    control_server = ControlServer(("0.0.0.0", int(args.http_port)), ControlHandler, max_workers=int(args.http_workers))
    ControlHandler.pipeline_manager = pipeline_manager
    ControlHandler.stream_port = int(args.stream_port)

    def serve_http() -> None:
        LOGGER.info("UI/API: http://0.0.0.0:%d/  |  API: /api/config /api/status /api/start /api/stop /api/upload | WHEP proxy: /whep/<path>", args.http_port)
        control_server.serve_forever()

    def serve_stream() -> None:
        LOGGER.info("Stream MJPEG: http://0.0.0.0:%d/workflow.mjpg", args.stream_port)
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
        LOGGER.info("server stopped")


if __name__ == "__main__":
    main()