import json
import urllib.request
import urllib.error


class ControlStream:
    """
    Minimal controller for MediaMTX (v3 API).
    Manages exactly one active path at a time; every start_* deletes other paths first.
    """

    def __init__(self, api_base_url: str = "http://127.0.0.1:9997", rtsp_output_host: str = "127.0.0.1"):
        """
        api_base_url: MediaMTX API base, e.g. "http://127.0.0.1:9997"
        rtsp_output_host: host where FFmpeg (inside MediaMTX) publishes RTSP (usually 127.0.0.1)
        """
        if not api_base_url.endswith("/"):
            api_base_url += "/"
        self.api_base_url = api_base_url
        self.rtsp_output_host = rtsp_output_host

    def start_rtsp_copy(self, rtsp_url: str, name: str = "rtsp_copy"):
        """
        Pull an external RTSP source and republish it via MediaMTX with video copy (no re-encode).
        """
        self._delete_all_paths()
        ffmpeg_cmd = (
            "ffmpeg -nostdin "
            "-rtsp_transport udp -fflags nobuffer+genpts -use_wallclock_as_timestamps 1 "
            f"-i {self._quote_shell(rtsp_url)} "
            "-c:v libx264 -g 30 -keyint_min 30 -b:v 2000k -profile:v baseline -bf 0 "
            f"-f rtsp -rtsp_transport tcp rtsp://{self.rtsp_output_host}:8554/{name}"
        )
        self._add_path_run_on_init(name, ffmpeg_cmd)

    def start_file(self, file_path_in_media: str, name: str = "file_loop"):
        """
        Loop a local file mounted under /media, transcode to stable H.264 (HLS-friendly).
        """
        self._delete_all_paths()
        ffmpeg_cmd = (
            "ffmpeg -nostdin "
            "-stream_loop -1 -re -fflags +genpts "
            f"-i {self._quote_shell(file_path_in_media)} "
            "-vf format=yuv420p "
            "-c:v libx264 -preset fast -crf 18 -bf 0 -g 60 -sc_threshold 0 -tune zerolatency "
            "-an "
            f"-f rtsp -rtsp_transport tcp rtsp://{self.rtsp_output_host}:8554/{name}"
        )
        self._add_path_run_on_init(name, ffmpeg_cmd)

    def start_video(self, video_device: str = "/dev/video0", name: str = "cam_low_latency"):
        """
        Stream a V4L2 camera (/dev/video*) with a low-latency H.264 encode.
        Assumes the container has access to the device.
        """
        self._delete_all_paths()
        ffmpeg_cmd = (
            "ffmpeg -nostdin "
            "-f v4l2 -thread_queue_size 4096 -input_format mjpeg -framerate 30 -video_size 1280x720 "
            "-fflags nobuffer+genpts -flags low_delay -probesize 32 -analyzeduration 0 "
            "-use_wallclock_as_timestamps 1 "
            f"-i {self._quote_shell(video_device)} "
            "-c:v libx264 -preset veryfast -crf 20 -bf 0 -g 30 -sc_threshold 0 -tune zerolatency "
            "-an "
            f"-f rtsp -rtsp_transport tcp rtsp://{self.rtsp_output_host}:8554/{name}"
        )
        self._add_path_run_on_init(name, ffmpeg_cmd)

    def stop_all(self) -> None:
        """Delete all existing paths."""
        self._delete_all_paths()

    def _add_path_run_on_init(self, name: str, ffmpeg_cmd: str) -> None:
        request_body = {
            "runOnInit": ffmpeg_cmd,
            "runOnInitRestart": True
        }
        self._api_post_json(f"v3/config/paths/add/{name}", request_body)

    def _delete_all_paths(self) -> None:
        paths = self._list_paths()
        for path_item in paths:
            path_name = path_item.get("name")
            if path_name:
                self._api_delete(f"v3/config/paths/delete/{path_name}")

    def _list_paths(self) -> list[dict]:
        try:
            response = self._api_get_json("v3/paths/list")
            return response.get("items", [])
        except Exception:
            return []

    def _api_get_json(self, suffix: str) -> dict:
        url = self.api_base_url + suffix
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def _api_post_json(self, suffix: str, body: dict) -> dict:
        url = self.api_base_url + suffix
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method="POST")
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def _api_delete(self, suffix: str) -> None:
        url = self.api_base_url + suffix
        request = urllib.request.Request(url, method="DELETE")
        try:
            urllib.request.urlopen(request, timeout=5).read()
        except urllib.error.HTTPError as http_error:
            if http_error.code != 404:
                raise

    @staticmethod
    def _quote_shell(string_value: str) -> str:
        """
        Simple quoting for embedding paths/URLs into shell commands used by MediaMTX.
        """
        if "'" not in string_value:
            return f"'{string_value}'"
        return "'" + string_value.replace("'", "'\"'\"'") + "'"
