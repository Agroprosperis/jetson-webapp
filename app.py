#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF video app — uses existing MediaMTX for Original via WebRTC (WHEP) + MJPEG for Workflow.
States: idle, starting, running, stopping. Control/UI and stream servers are split to avoid deadlocks.

Run:
  python app-v4.py --workflow-id "$WORKFLOW_ID" --workspace-id "$WORKSPACE_ID" \
    --inference-server-url http://127.0.0.1:9001 --api-key "$ROBOFLOW_API_KEY" \
    --http-port 8081 --image-field rendered_output_hq
"""
import os
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k,"1")

import argparse, json, logging, sys, time, threading, traceback
from logging.handlers import RotatingFileHandler
from http.server import BaseHTTPRequestHandler, HTTPServer
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse, unquote  # WHEP FIX

import cv2 as cv
import numpy as np
import requests  # WHEP proxy
import csv
import cgi
import subprocess

_TJ = None
try:
    from turbojpeg import TurboJPEG  # type: ignore
    _TJ = TurboJPEG()
except Exception:
    _TJ = None

from inference_sdk import InferenceHTTPClient  # type: ignore
from inference.core.utils.image_utils import load_image  # type: ignore

LOGGER = logging.getLogger("rf-app")


def setup_logging(level="INFO", log_file=None, max_bytes=5*1024*1024, backup_count=3):
    fmt="%(asctime)s | %(levelname)s | %(threadName)s | rf-app | %(message)s"
    datefmt="%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=[])
    ch=logging.StreamHandler(sys.stderr); ch.setFormatter(logging.Formatter(fmt,datefmt))
    logging.getLogger().addHandler(ch)
    if log_file:
        fh=RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        fh.setFormatter(logging.Formatter(fmt,datefmt)); logging.getLogger().addHandler(fh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

try: cv.setNumThreads(1)
except Exception: pass
try:
    if hasattr(cv,"ocl"): cv.ocl.setUseOpenCL(False)
except Exception: pass


class SharedFrame:
    def __init__(self): self._lock=threading.Lock(); self._img: Optional[np.ndarray]=None
    def set(self, img: Optional[np.ndarray]):
        with self._lock: self._img = None if img is None else img
    def get(self) -> Optional[np.ndarray]:
        with self._lock: return None if self._img is None else self._img.copy()
    def clear(self): self.set(None)


def _prepare_rtsp_url(url: str, transport: Optional[str]) -> str:
    if not url.lower().startswith("rtsp://") or not transport: return url
    if transport not in ("tcp","udp"): return url
    p = urlparse(url); q = dict(parse_qsl(p.query, keep_blank_values=True))
    q.setdefault("rtsp_transport", transport)
    return urlunparse(p._replace(query=urlencode(q)))


def _adjust_for_docker(vref: Any) -> Any:
    if not isinstance(vref, str) or not vref.lower().startswith("rtsp://"):
        return vref
    LOGGER.debug("[adjust] Input vref: %s", vref)
    p = urlparse(vref)
    LOGGER.debug("[adjust] Parsed: scheme=%s, netloc=%s, hostname=%s, port=%s, path=%s, query=%s",
                 p.scheme, p.netloc, p.hostname, p.port, p.path, p.query)
    LOGGER.debug("[adjust] On Linux, no adjustment for --network host")
    return vref


def _extract_pipeline_id(resp: Any) -> Optional[str]:
    def dfs(o: Any) -> Optional[str]:
        if isinstance(o, dict):
            v = o.get("pipeline_id") or o.get("id")
            if isinstance(v,str) and v: return v
            for vv in o.values():
                r = dfs(vv); 
                if r: return r
        elif isinstance(o,(list,tuple)):
            for it in o:
                r=dfs(it)
                if r: return r
        return None
    return dfs(resp)


def _list_ids(client: InferenceHTTPClient) -> List[str]:
    try:
        LOGGER.debug("[list_ids] Requesting existing pipelines")
        resp = client.list_inference_pipelines()
        LOGGER.debug(f"[list_ids] existing pipelines raw={resp}")
        if isinstance(resp, dict):
            lst = resp.get("pipelines", [])
        else:
            lst = resp or []
        out = []
        for p in lst:
            if isinstance(p, dict):
                v = p.get("pipeline_id") or p.get("id")
                if isinstance(v, str) and v:
                    out.append(v)
            elif isinstance(p, str) and p:
                out.append(p)
        LOGGER.debug("[list_ids] Found pipelines: %s", out)
        return out
    except Exception as e:
        LOGGER.debug("list_inference_pipelines failed: %s", e)
        return []


def _await_new_pipeline(client: InferenceHTTPClient, before: set, timeout_s=10.0, every=0.25) -> Optional[str]:
    t0=time.time(); last=None
    while time.time()-t0<timeout_s:
        try:
            after=set(_list_ids(client))
            new=[x for x in after if x and x not in before]
            if new:
                LOGGER.debug("[await] Found new pipeline: %s", new[0])
                return new[0]
            last=after
        except Exception as e:
            LOGGER.debug("[mgr] await_new_pipeline exception: %s", e)
        time.sleep(every)
    LOGGER.debug("[await] No new pipeline found within timeout")
    return None


def poll_worker(client: InferenceHTTPClient, pipeline_id: str, image_field: str,
                shared: SharedFrame, stop_ev: threading.Event, excluded_fields: Optional[List[str]]=None) -> None:
    LOGGER.info("[poll] start pipeline_id=%s image_field=%s excluded=%s", pipeline_id, image_field, excluded_fields)
    cands=[image_field,"preview","rendered_output_hq","rendered_output","image"]
    csv_path = f"{pipeline_id}.csv"
    header = ['timestamp', 'class', 'class_id', 'confidence', 'x', 'y', 'width', 'height', 'total_unique_objects_count']
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
    while not stop_ev.is_set():
        try:
            res = client.consume_inference_pipeline_result(pipeline_id=pipeline_id, excluded_fields=excluded_fields or [])
            if not isinstance(res, dict):
                LOGGER.warning("[poll] consumed non-dict res: type=%s value=%r", type(res), res)
                time.sleep(0.05)
                continue
            outs = res.get("outputs") or []
            if not isinstance(outs, list):
                LOGGER.warning("[poll] outputs not list: %r", outs)
                time.sleep(0.05)
                continue
            if not outs:
                LOGGER.info("[poll] no data")
                time.sleep(0.01)
                continue
            sres = outs[0]
            if not isinstance(sres, dict):
                LOGGER.warning("[poll] sres not dict: %r", sres)
                time.sleep(0.05)
                continue
            if not sres:
                LOGGER.info("[poll] no data")
                time.sleep(0.01)
                continue
            LOGGER.debug("[poll] sres keys: %s", list(sres.keys()))
            # Dump to CSV excluding b64 images
            predictions = sres.get("predictions", [])
            if isinstance(predictions, dict):
                inner_predictions = predictions.get('predictions', [])
                if isinstance(inner_predictions, list):
                    predictions = inner_predictions
                else:
                    LOGGER.warning("[poll] inner predictions not list: %r", inner_predictions)
                    time.sleep(0.05)
                    continue
            if not isinstance(predictions, list):
                LOGGER.warning("[poll] predictions not list: %r", predictions)
                time.sleep(0.05)
                continue
            total_unique_objects_count = sres.get("total_unique_objects_count", 0)
            if predictions:
                timestamp = time.time()
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for pred in predictions:
                        if not isinstance(pred, dict):
                            LOGGER.warning("[poll] pred not dict: %r", pred)
                            continue
                        cls = pred.get('class')
                        cls_id = pred.get('class_id')
                        conf = pred.get('confidence')
                        bbox = pred.get('bbox', {}) or {}
                        x = bbox.get('x')
                        y = bbox.get('y')
                        w = bbox.get('width')
                        h = bbox.get('height')
                        writer.writerow([timestamp, cls, cls_id, conf, x, y, w, h, total_unique_objects_count])
            val=None
            for k in cands:
                if k in sres: val=sres[k]; break
            if val is None:
                LOGGER.warning("[poll] no image field in sres (tried %s) keys: %s", cands, list(sres.keys()))
                time.sleep(0.01)
                continue
            LOGGER.debug("[poll] Found val for key %s type %s", k, type(val))
            if isinstance(val, dict):
                LOGGER.debug("[poll] val dict keys: %s", list(val.keys()))
                val = val.get('value') or val.get('image') or val.get('base64') or val.get('b64') or val
            if not isinstance(val, str):
                LOGGER.warning("[poll] val not str after extraction: type=%s val=%r", type(val), val)
                time.sleep(0.01)
                continue
            img,_ = load_image(val)
            if img is None:
                LOGGER.warning("[poll] load_image failed for val=%r", val)
                time.sleep(0.01)
                continue
            LOGGER.debug("[poll] Loaded img shape %s", img.shape)
            if img.ndim==2: img=cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            elif img.shape[2]==4: img=cv.cvtColor(img, cv.COLOR_BGRA2BGR)
            shared.set(img)
            LOGGER.info("[poll] success - sres keys: %s", list(sres.keys()))
        except Exception as e:
            LOGGER.warning("[poll] exception: %s", e)
            time.sleep(0.05)
    shared.clear()
    LOGGER.info("[poll] stop")


class PipelineManager:
    # states: idle, starting, running, stopping
    def __init__(self, client: InferenceHTTPClient, wf_shared: SharedFrame, base_cfg: Dict[str,Any]):
        self._client=client; self._wf_shared=wf_shared; self._base=dict(base_cfg)
        self._lock=threading.Lock(); self._state="idle"
        self._pipeline_id: Optional[str]=None; self._cfg: Dict[str,Any]={}
        self._poll_th: Optional[threading.Thread]=None; self._poll_stop: Optional[threading.Event]=None
        self._last_error: Optional[str]=None
        self._cancel_ev=threading.Event()

    def _set_state(self,s:str):
        if s!=self._state:
            LOGGER.info("[mgr] state %s -> %s", self._state, s)
            self._state=s

    def _start_internal(self, cfg: Dict[str,Any], cancel_ev: threading.Event):
        if self._pipeline_id:
            self._stop_locked(grace=True)
        self._set_state("starting"); self._last_error=None
        LOGGER.info("[start] Pipeline start initiated with internal config: %s", json.dumps(cfg, indent=2))

        before=set(_list_ids(self._client))
        LOGGER.info("[start] Pipelines before start: %s", before)
        video_reference=cfg["video_reference"]
        LOGGER.info("[start] Using video_reference: %s", video_reference)

        # Detect and configure external RTSP in MediaMTX if necessary
        is_external = False
        pathname = None
        if isinstance(video_reference, str) and video_reference.lower().startswith("rtsp://"):
            p = urlparse(video_reference)
            if p.hostname not in ("127.0.0.1", "localhost", "::1"):
                is_external = True
                pathname = p.path.strip("/").split("/")[-1] or "proxy"
                mediamtx_api_base = self._base["mediamtx_api"].rstrip("/")
                mediamtx_remove_api = mediamtx_api_base + f"/v3/config/paths/remove/{pathname}"
                mediamtx_add_api = mediamtx_api_base + f"/v3/config/paths/add/{pathname}"
                mediamtx_patch_api = mediamtx_api_base + f"/v3/config/paths/patch/{pathname}"
                mtx_config = {"source": video_reference}
                try:
                    # Remove first to handle conflicts (ignore errors)
                    resp = requests.delete(mediamtx_remove_api, timeout=5)
                    LOGGER.info("[start] Attempted to remove existing path '%s': status %s", pathname, resp.status_code)
                except Exception as e:
                    LOGGER.debug("[start] Ignore remove error for path '%s': %s", pathname, e)
                try:
                    LOGGER.info("[start] Attempting to add mediamtx path '%s'", mtx_config)
                    resp = requests.post(mediamtx_add_api, json=mtx_config, timeout=5)
                    if resp.status_code not in (200, 201):
                        if resp.status_code == 400 and "already exists" in resp.text.lower():
                            LOGGER.info("[start] Add failed due to existing, attempting patch")
                            resp = requests.patch(mediamtx_patch_api, json=mtx_config, timeout=5)
                            if resp.status_code not in (200, 201):
                                raise Exception(f"MediaMTX PATCH returned {resp.status_code}: {resp.text}")
                            LOGGER.info("[start] Patched existing RTSP path '%s' to MediaMTX: status %s", pathname, resp.status_code)
                        else:
                            raise Exception(f"MediaMTX ADD returned {resp.status_code}: {resp.text}")
                    else:
                        LOGGER.info("[start] Added external RTSP path '%s' to MediaMTX: status %s", pathname, resp.status_code)
                except Exception as e:
                    LOGGER.warning("[start] Failed to add/patch path to MediaMTX: %s", e)
                    # Continue anyway, as workflow may still work (preview might fail)

        rbs=int(cfg.get("results_buffer_size",1))
        bct=float(cfg.get("batch_collection_timeout",0.03))
        api_params = {
            "video_reference": [video_reference],
            "workspace_name": self._base["workspace_id"],
            "workflow_id": self._base["workflow_id"],
            "results_buffer_size": max(1, rbs),
            "batch_collection_timeout": bct,
        }
        LOGGER.info("[start] Exact API params for start_inference_pipeline_with_workflow: %s", json.dumps(api_params, indent=2))
        LOGGER.info("[start] Initiating pipeline start call...")

        try:
            LOGGER.debug("[start] Executing start_inference_pipeline_with_workflow...")
            resp=self._client.start_inference_pipeline_with_workflow(**api_params)
            LOGGER.info("[start] Pipeline start call completed successfully, response: %s", resp)
        except Exception as e:
            self._last_error=f"start failed: {e}"
            LOGGER.error("[start] %s\n%s", self._last_error, traceback.format_exc())
            self._set_state("idle"); return

        pid=_extract_pipeline_id(resp) or _await_new_pipeline(self._client, before, 10.0, 0.25)
        if not pid:
            self._last_error="no pipeline_id from server"
            LOGGER.error("[start] %s", self._last_error)
            self._set_state("idle"); return

        LOGGER.info("[start] Pipeline ID obtained: %s", pid)

        if cancel_ev.is_set():
            try: self._client.terminate_inference_pipeline(pipeline_id=pid)
            except Exception as e: LOGGER.debug("[mgr] terminate after-cancel failed: %s", e)
            self._set_state("idle"); return

        stop_ev=threading.Event()
        th=threading.Thread(target=poll_worker, name="Poller", daemon=True,
                            kwargs=dict(client=self._client, pipeline_id=pid, image_field=cfg["image_field"],
                                        shared=self._wf_shared, stop_ev=stop_ev, excluded_fields=cfg.get("excluded_fields")))
        th.start()

        with self._lock:
            self._pipeline_id=pid; self._cfg=dict(cfg); self._poll_th=th; self._poll_stop=stop_ev
            self._cfg["is_external"] = is_external
            self._cfg["pathname"] = pathname

        self._set_state("running")

    def _stop_locked(self, grace: bool):
        pid=self._pipeline_id
        self._set_state("stopping")
        try:
            if self._poll_stop: self._poll_stop.set()
        except Exception: pass
        if grace and self._poll_th:
            try: self._poll_th.join(timeout=2.0)
            except Exception: pass
        self._poll_th=None; self._poll_stop=None; self._wf_shared.clear()
        if pid:
            try:
                active_ids = _list_ids(self._client)
                if pid in active_ids:
                    self._client.terminate_inference_pipeline(pipeline_id=pid)
                    LOGGER.info("[mgr] terminated pipeline id=%s", pid)
                else:
                    LOGGER.info("[mgr] pipeline id=%s already terminated", pid)
            except Exception as e:
                LOGGER.warning("[mgr] terminate failed: %s", e)
        # Clean up external path in MediaMTX if applicable
        if self._cfg.get("is_external", False) and (pathname := self._cfg.get("pathname")):
            mediamtx_api_base = self._base["mediamtx_api"].rstrip("/")
            mediamtx_remove_api = mediamtx_api_base + f"/v3/config/paths/remove/{pathname}"
            try:
                resp = requests.delete(mediamtx_remove_api, timeout=5)
                LOGGER.info("[stop] Removed external RTSP path '%s' from MediaMTX: status %s", pathname, resp.status_code)
            except Exception as e:
                LOGGER.warning("[stop] Failed to remove path from MediaMTX: %s", e)
        self._pipeline_id=None
        self._set_state("idle")

    def start_async(self, cfg: Dict[str,Any]) -> None:
        with self._lock:
            if self._state in ("starting","stopping"):
                self._cancel_ev.set()
            self._cancel_ev.clear()
            threading.Thread(target=self._start_internal, name="StartAsync", daemon=True,
                             args=(dict(cfg), self._cancel_ev)).start()

    def stop_async(self, grace: bool=True) -> None:
        with self._lock:
            self._cancel_ev.set()
            threading.Thread(target=self._stop_locked, name="StopAsync", daemon=True,
                             args=(grace,)).start()

    def status(self) -> Dict[str,Any]:
        with self._lock:
            return {
                "state": self._state,
                "running": self._state=="running",
                "pipeline_id": self._pipeline_id,
                "config": dict(self._cfg) if self._pipeline_id else None,
                "workspace_id": self._base["workspace_id"],
                "workflow_id": self._base["workflow_id"],
                "last_error": self._last_error,
            }


class StreamServer(HTTPServer):
    def __init__(self, addr, HandlerClass, max_streams: int=4):
        super().__init__(addr, HandlerClass)
        self._max=max_streams; self._cur=0; self._lock=threading.Lock()
    def may_open(self)->bool:
        with self._lock:
            if self._cur>=self._max: return False
            self._cur+=1; return True
    def closed(self)->None:
        with self._lock: self._cur=max(0,self._cur-1)

class StreamHandler(BaseHTTPRequestHandler):
    wf_shared: SharedFrame=None
    server_ref: StreamServer=None

    def log_message(self, fmt,*args): LOGGER.debug("stream: "+fmt, *args)

    def _parse_qs(self)->Tuple[int,int,int,int]:
        from urllib.parse import urlparse, parse_qs
        qs=parse_qs(urlparse(self.path).query)
        def _i(k,d): 
            try: return int(qs.get(k,[str(d)])[0])
            except Exception: return d
        return _i("w",0), _i("h",0), max(10,min(95,_i("q",85))), max(1,min(60,_i("fps",20)))

    def _write_headers(self):
        self.wfile.write(b"HTTP/1.1 200 OK\r\n")
        self.wfile.write(b"Cache-Control: no-cache, no-store, must-revalidate\r\n")
        self.wfile.write(b"Pragma: no-cache\r\n")
        self.wfile.write(b"Expires: 0\r\n")
        self.wfile.write(b"Connection: close\r\n")
        self.wfile.write(b"Access-Control-Allow-Origin: *\r\n")
        self.wfile.write(b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n")
        self.wfile.write(b"\r\n"); self.wfile.flush()

    def _encode_jpeg(self, img: np.ndarray, q: int)->Optional[bytes]:
        if _TJ is not None:
            try: return _TJ.encode(img, quality=q)
            except Exception as e: LOGGER.debug("TurboJPEG encode failed: %s", e)
        ok,enc=cv.imencode(".jpg", img, [int(cv.IMWRITE_JPEG_QUALITY),int(q),int(cv.IMWRITE_JPEG_OPTIMIZE),1])
        return enc.tobytes() if ok else None

    def _scale(self, img: np.ndarray, W: int, H: int)->np.ndarray:
        if W<=0 or H<=0: return img
        h,w=img.shape[:2]; s=min(W/max(1,w), H/max(1,h))
        nw,nh=max(1,int(w*s)), max(1,int(h*s))
        if (nw,nh)==(w,h): return img
        return cv.resize(img,(nw,nh), interpolation=cv.INTER_AREA if s<1.0 else cv.INTER_LINEAR)

    def _write_jpeg_bytes(self, jpeg: bytes):
        self.wfile.write(b"--frame\r\n")
        self.wfile.write(b"Content-Type: image/jpeg\r\n")
        self.wfile.write(b"Content-Length: "+str(len(jpeg)).encode("ascii")+b"\r\n\r\n")
        self.wfile.write(jpeg+b"\r\n"); self.wfile.flush()

    def do_GET(self):
        path=self.path.split("?")[0]
        if not self.server_ref.may_open():
            self.send_response(503); self.send_header("Content-Type","text/plain; charset=utf-8")
            self.end_headers(); self.wfile.write(b"stream capacity reached"); return
        try:
            if path.startswith("/workflow.mjpg"):
                W,H,Q,FPS=self._parse_qs(); per=1.0/float(max(1,FPS))
                self._write_headers()
                placeholder = np.zeros((240, 320, 3), np.uint8)
                cv.putText(placeholder, "Loading workflow...", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                try:
                    while True:
                        img=self.wf_shared.get()
                        if img is None:
                            disp_img = placeholder.copy()
                        else:
                            disp_img = img
                        if W>0 and H>0: disp_img=self._scale(disp_img,W,H)
                        jpeg=self._encode_jpeg(disp_img,Q)
                        if jpeg is None:
                            LOGGER.warning("[stream] JPEG encode failed")
                            time.sleep(0.01); continue
                        self._write_jpeg_bytes(jpeg)
                        time.sleep(per)
                except (BrokenPipeError, ConnectionResetError):
                    LOGGER.info("close workflow.mjpg")
                except Exception as e:
                    LOGGER.warning("workflow.mjpg error: %s", e)
            else:
                self.send_response(404); self.end_headers()
        finally:
            self.server_ref.closed()


class ControlServer(HTTPServer):
    def __init__(self, addr, HandlerClass, max_workers: int=3):
        super().__init__(addr, HandlerClass)
        self._exec=ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="httpC")
        self._shutting=False
        LOGGER.info("HTTP control workers=%d", max_workers)
    def process_request(self, request, client_address):
        if self._shutting: self.shutdown_request(request); return
        self._exec.submit(self._handle, request, client_address)
    def _handle(self, request, client_address):
        try: self.finish_request(request, client_address)
        except Exception: self.handle_error(request, client_address)
        finally: self.shutdown_request(request)
    def server_close(self):
        self._shutting=True
        try: super().server_close()
        finally:
            try: self._exec.shutdown(wait=False, cancel_futures=True)
            except Exception: pass


class ControlHandler(BaseHTTPRequestHandler):
    manager: PipelineManager=None
    stream_port: int=None
    mediamtx_http: str=None
    mediamtx_whep_path: str=None
    mediamtx_rtsp_host: str=None
    upload_rtsp_path: str = "uploaded"
    ffmpeg_proc: Optional[subprocess.Popen] = None

    def log_message(self, fmt,*args): LOGGER.debug("http: "+fmt, *args)

    # WHEP FIX: normalize the path, no querystring
    def _clean_path(self) -> str:
        return self.path.split("?", 1)[0]

    def _send_json(self, obj: Any, code: int=200):
        data=json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_bytes(self) -> bytes:
        try:
            L=int(self.headers.get("Content-Length","0"))
            return self.rfile.read(L) if L>0 else b""
        except Exception:
            return b""

    # WHEP FIX: log every request line
    def handle_one_request(self):
        try:
            super().handle_one_request()
        finally:
            try:
                LOGGER.debug("HTTP %s %s", getattr(self, "command", "?"), getattr(self, "path", "?"))
            except Exception:
                pass

    # Handle CORS preflight for the local WHEP proxy
    def do_OPTIONS(self):
        p=self._clean_path()
        if p.startswith("/whep/"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.end_headers()
            return
        self.send_response(204); self.end_headers()

    def do_GET(self):
        p=self._clean_path()
        if p.startswith("/whep/"):
            # WHEP FIX: explicit method guard helps debugging (curl will see 405 instead of 404)
            self.send_response(405)
            self.send_header("Allow", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return

        if p in ("/","/index.html"):
            page = (
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
                b"  const base='http://'+location.hostname+':'+cfg.stream_port; imgWf.src=base+'/workflow.mjpg?fps=20';"
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
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        if p.startswith("/api/status"):
            self._send_json(self.manager.status(), 200); return

        if p.startswith("/api/config"):
            self._send_json({
                "stream_port": self.stream_port,
                "mediamtx_http": self.mediamtx_http,
                "mediamtx_whep_path": self.mediamtx_whep_path
            }, 200); return

        self.send_response(404); self.end_headers()

    def do_POST(self):
        p=self._clean_path()

        # Local WHEP reverse proxy to avoid cross-origin issues
        if p == "/whep" or p == "/whep/":  # WHEP FIX: guard empty path
            msg=b"missing WHEP path (expected /whep/<path>)"
            self.send_response(400)
            self.send_header("Content-Type","text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        if p.startswith("/whep/"):
            path = unquote(p[len("/whep/"):].lstrip("/"))  # WHEP FIX
            sdp = self._read_bytes()
            target = self.mediamtx_http.rstrip("/") + "/" + path + "/whep"
            try:
                LOGGER.debug("WHEP proxy -> %s", target)
                rr = requests.post(target, data=sdp, headers={"Content-Type":"application/sdp"}, timeout=10)
                body = rr.text.encode("utf-8")
                self.send_response(rr.status_code)
                self.send_header("Content-Type","application/sdp")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                msg=f"whep proxy error: {e}".encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type","text/plain; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            return

        if p.startswith("/api/upload"):
            try:
                environ = {
                    'REQUEST_METHOD': self.command,
                    'CONTENT_TYPE': self.headers['Content-Type'],
                }
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)
                if 'file' not in form:
                    self._send_json({"ok": False, "error": "no file"}, 400)
                    return
                file_item = form['file']
                if not file_item.filename:
                    self._send_json({"ok": False, "error": "no filename"}, 400)
                    return
                ext = os.path.splitext(file_item.filename)[1] or ".mp4"
                upload_file = f"/tmp/rf_uploaded_{int(time.time())}{ext}"
                with open(upload_file, 'wb') as f:
                    f.write(file_item.file.read())
                if self.ffmpeg_proc:
                    try:
                        self.ffmpeg_proc.terminate()
                        self.ffmpeg_proc.wait(timeout=5)
                    except Exception as e:
                        LOGGER.warning("[upload] failed to terminate previous ffmpeg: %s", e)
                    self.ffmpeg_proc = None
                rtsp_url = f"rtsp://{self.mediamtx_rtsp_host}/{self.upload_rtsp_path}"
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-stream_loop", "-1",
                    "-re",
                    "-i", upload_file,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-tune", "zerolatency",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-f", "rtsp",
                    "-rtsp_transport", "tcp",
                    rtsp_url
                ]
                LOGGER.info("[upload] starting ffmpeg: %s", " ".join(ffmpeg_cmd))
                proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.ffmpeg_proc = proc
                if not self._is_stream_available(rtsp_url):
                    proc.terminate()
                    try: proc.wait(timeout=5)
                    except Exception: pass
                    try: os.remove(upload_file)
                    except Exception: pass
                    self._send_json({"ok": False, "error": "failed to start stream"}, 500)
                    return
                time.sleep(2)  # additional delay for WebRTC to be ready
                self._send_json({"ok": True, "video": rtsp_url, "rtsp_transport": "tcp"}, 200)
            except Exception as e:
                LOGGER.warning("[upload] error: %s", e)
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        if p.startswith("/api/start"):
            raw_body=self._read_bytes()
            LOGGER.debug("[api/start] Raw body: %s", raw_body)
            try: body=json.loads(raw_body.decode("utf-8") or "{}")
            except Exception: body={}
            LOGGER.debug("[api/start] Parsed body: %s", body)
            video=str(body.get("video","")).strip(); rtsp_transport=body.get("rtsp_transport", None)
            LOGGER.debug("[api/start] Extracted video: %s", video)
            LOGGER.debug("[api/start] rtsp_transport: %s", rtsp_transport)
            if not video: self._send_json({"ok":False,"error":"video is required"}, 400); return
            try:
                vref = int(video)
            except Exception:
                vref = video
                if isinstance(vref,str) and vref.lower().startswith("rtsp://"):
                    vref = _prepare_rtsp_url(vref, rtsp_transport)
            LOGGER.debug("[api/start] vref after prepare: %s", vref)
            vref = _adjust_for_docker(vref)
            LOGGER.debug("[api/start] vref after adjust: %s", vref)
            st=self.manager.status()
            image_field=(st.get("config") or {}).get("image_field") or "rendered_output_hq"
            rbs=(st.get("config") or {}).get("results_buffer_size") or 1
            bct=(st.get("config") or {}).get("batch_collection_timeout") or 0.03
            excluded=(st.get("config") or {}).get("excluded_fields") or []
            cfg=dict(video_reference=vref,image_field=image_field,results_buffer_size=rbs,
                     batch_collection_timeout=bct,excluded_fields=excluded, original_video=video, rtsp_transport=rtsp_transport)
            self.manager.start_async(cfg)
            self._send_json({"ok":True,"accepted":True}, 202); return

        if p.startswith("/api/stop"):
            self.manager.stop_async(grace=True)
            if self.ffmpeg_proc:
                try:
                    self.ffmpeg_proc.terminate()
                    self.ffmpeg_proc.wait(timeout=5)
                except Exception as e:
                    LOGGER.warning("[stop] failed to terminate ffmpeg: %s", e)
                self.ffmpeg_proc = None
            self._send_json({"ok":True,"accepted":True}, 202); return

        self.send_response(404); self.end_headers()

    def _is_stream_available(self, rtsp_url: str, timeout: int = 10) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
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


def parse_args():
    p=argparse.ArgumentParser(description="RF video app: WebRTC Original via existing MediaMTX + MJPEG Workflow")
    p.add_argument("--inference-server-url", required=True, type=str)
    p.add_argument("--api-key", required=True, type=str)
    p.add_argument("--workspace-id", required=True, type=str)
    p.add_argument("--workflow-id", required=True, type=str)
    p.add_argument("--rtsp-transport", choices=["tcp","udp"], default=None)
    p.add_argument("--image-field", default="rendered_output_hq", type=str)
    p.add_argument("--results-buffer-size", default=4, type=int)
    p.add_argument("--batch-collection-timeout", default=0.05, type=float)
    p.add_argument("--http-port", default=8081, type=int)
    p.add_argument("--http-workers", default=3, type=int)
    p.add_argument("--stream-port", default=8082, type=int)
    p.add_argument("--stream-max", default=4, type=int)
    p.add_argument("--mediamtx-http", default="http://127.0.0.1:8889", type=str,
                   help="MediaMTX HTTP base for WHEP (existing)")
    p.add_argument("--mediamtx-api", default="http://127.0.0.1:9997", type=str,
                   help="MediaMTX API base for config (existing)")
    p.add_argument("--mediamtx-rtsp", default="127.0.0.1:8554", type=str,
                   help="MediaMTX RTSP host:port for publishing (existing)")
    p.add_argument("--mediamtx-whep-path", default="original", type=str,
                   help="MediaMTX path already being published (matches your rtsp://...:8554/<path>)")
    p.add_argument("--log-level", default="INFO", type=str)
    p.add_argument("--log-file", default=None, type=str)
    p.add_argument("--log-max-bytes", default=5*1024*1024, type=int)
    p.add_argument("--log-backup-count", default=3, type=int)
    p.add_argument("--exclude", default="", type=str)
    return p.parse_args()


def main():
    args=parse_args()
    setup_logging(args.log_level, args.log_file, args.log_max_bytes, args.log_backup_count)

    LOGGER.info("boot ctrl=%d stream=%d inference=%s ws=%s wf=%s mediamtx_http=%s mediamtx_api=%s whep_path=%s mediamtx_rtsp=%s",
                args.http_port, args.stream_port, args.inference_server_url,
                args.workspace_id, args.workflow_id, args.mediamtx_http, args.mediamtx_api, args.mediamtx_whep_path, args.mediamtx_rtsp)

    client=InferenceHTTPClient(api_url=args.inference_server_url, api_key=args.api_key)
    LOGGER.info("Inference client initialized with url: %s", args.inference_server_url)
    wf_shared=SharedFrame()
    base_cfg = {"workspace_id": args.workspace_id, "workflow_id": args.workflow_id, "mediamtx_http": args.mediamtx_http, "mediamtx_api": args.mediamtx_api}
    manager=PipelineManager(client=client, wf_shared=wf_shared,base_cfg=base_cfg)

    existing = _list_ids(client)
    if existing:
        pid = existing[0]
        LOGGER.info("Attaching to existing pipeline: %s", pid)
        excluded = [s.strip() for s in args.exclude.split(",") if s.strip()]
        vref = ""
        # Fetch pipeline status to get video_reference
        status_url = f"{args.inference_server_url}/inference_pipelines/{pid}/status"
        try:
            resp = requests.get(status_url, params={"api_key": args.api_key}, timeout=5)
            if resp.status_code == 200:
                status_data = resp.json()
                report = status_data.get("report", {})
                sources_metadata = report.get("sources_metadata", [])
                if sources_metadata:
                    source_ref = sources_metadata[0].get("source_reference", "")
                    if source_ref:
                        vref = source_ref
                        LOGGER.info("Fetched video_reference from pipeline status: %s", source_ref)
        except Exception as e:
            LOGGER.warning("Failed to fetch pipeline status for video_reference: %s", e)
        cfg = dict(video_reference=vref, image_field=args.image_field,
                   results_buffer_size=max(1, int(args.results_buffer_size)),
                   batch_collection_timeout=float(args.batch_collection_timeout),
                   excluded_fields=excluded, original_video=vref, rtsp_transport=args.rtsp_transport)
        stop_ev = threading.Event()
        th = threading.Thread(target=poll_worker, name="Poller", daemon=True,
                              kwargs=dict(client=client, pipeline_id=pid, image_field=cfg["image_field"],
                                          shared=wf_shared, stop_ev=stop_ev, excluded_fields=cfg.get("excluded_fields")))
        th.start()
        with manager._lock:
            manager._pipeline_id = pid
            manager._cfg = dict(cfg)
            manager._poll_th = th
            manager._poll_stop = stop_ev
        manager._set_state("running")

    stream_srv=StreamServer(("0.0.0.0", int(args.stream_port)), StreamHandler, max_streams=int(args.stream_max))
    StreamHandler.wf_shared=wf_shared
    StreamHandler.server_ref=stream_srv

    ctrl_srv=ControlServer(("0.0.0.0", int(args.http_port)), ControlHandler, max_workers=int(args.http_workers))
    ControlHandler.manager=manager
    ControlHandler.stream_port=int(args.stream_port)
    ControlHandler.mediamtx_http=args.mediamtx_http
    ControlHandler.mediamtx_whep_path=args.mediamtx_whep_path
    ControlHandler.mediamtx_rtsp_host=args.mediamtx_rtsp

    def _serve_http():
        LOGGER.info("UI/API: http://0.0.0.0:%d/  |  API: /api/config /api/status /api/start /api/stop /api/upload | WHEP proxy: /whep/<path>", args.http_port)
        ctrl_srv.serve_forever()
    def _serve_stream():
        LOGGER.info("Workflow MJPEG: http://0.0.0.0:%d/workflow.mjpg", args.stream_port)
        stream_srv.serve_forever()

    t1=threading.Thread(target=_serve_http, name="HTTP-Ctrl", daemon=True)
    t2=threading.Thread(target=_serve_stream, name="HTTP-Stream", daemon=True)
    t1.start(); t2.start()

    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested")
    finally:
        for srv in [ctrl_srv, stream_srv]:
            try: srv.shutdown(); srv.server_close()
            except Exception: pass
        try: manager.stop_async(grace=True)
        except Exception: pass
        
        if ControlHandler.ffmpeg_proc:
            try: ControlHandler.ffmpeg_proc.terminate()
            except Exception: pass
        LOGGER.info("server stopped")


if __name__ == "__main__":
    main()