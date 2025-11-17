import cv2
import logging
import numpy as np
import os


LOGGER = logging.getLogger("stream_reader")

class StreamReader:
    """
    Base class for video ingestion.

    Usage:
        with SomeStreamReader(...) as reader:
            ret, frame = reader.read()
    """

    def __init__(self, width: int | None = None, height: int | None = None, fps: int | None = None) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: cv2.VideoCapture | None = None

    # context manager
    def __enter__(self) -> "StreamReader":
        if not self.open():
            raise RuntimeError("Failed to open capture")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # interface
    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class V4L2StreamReader(StreamReader):
    def __init__(self, device: str, width: int, height: int, fps: int) -> None:
        super().__init__(width=width, height=height, fps=fps)
        self.device = device

    @staticmethod
    def _build_v4l2_gst_pipeline(device: str, width: int, height: int, fps: int) -> str:
        pipeline = (
            f"v4l2src device={device} ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            "jpegdec ! videoconvert ! queue max-size-buffers=1 leaky=downstream ! appsink max-buffers=1 drop=true sync=false"
        )
        LOGGER.info("CAP pipeline: %s", pipeline)
        return pipeline

    def open(self) -> bool:
        gst = self._build_v4l2_gst_pipeline(self.device, self.width, self.height, self.fps)
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        return self.is_opened()


class RTSPStreamReader(StreamReader):
    def __init__(self, rtsp_url: str, fps: int, rtsp_transport: str = "") -> None:
        super().__init__(width=None, height=None, fps=fps)
        self.rtsp_url = rtsp_url
        self.rtsp_transport = rtsp_transport

    def open(self) -> bool:
        url = self.rtsp_url
        if self.rtsp_transport:
            url += f"?rtsp_transport={self.rtsp_transport}"

        self.cap = cv2.VideoCapture(url)
        if not self.cap or not self.cap.isOpened():
            LOGGER.error("Failed to open RTSP via FFmpeg backend: %s", url)
            self.cap = None
            return False
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 1e-3:
            self.fps = fps
        LOGGER.info("RTSP stream opened: %sx%s @ %.2f fps", self.width or "?", self.height or "?", self.fps or "?")
        return True

class __RTSPStreamReader(StreamReader):
    def __init__(self, rtsp_url: str, fps: int, rtsp_transport: str = "") -> None:
        super().__init__(width=None, height=None, fps=fps)
        self.rtsp_url = rtsp_url
        self.rtsp_transport = rtsp_transport

    def _build_rtsp_pipeline(self) -> str:
        protocols_str = ""
        if self.rtsp_transport == "tcp":
            protocols_str = " protocols=4"
        elif self.rtsp_transport == "udp":
            protocols_str = " protocols=1"
        pipeline = (
            f"rtspsrc location={self.rtsp_url} latency=0{protocols_str} ! "
            "decodebin ! videoconvert ! video/x-raw,format=BGR ! queue max-size-buffers=1 leaky=downstream ! appsink sync=false max-buffers=1 drop=true"
        )
        LOGGER.info("CAP pipeline: %s", pipeline)
        return pipeline

    def open(self) -> bool:
        gst = self._build_rtsp_pipeline()
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not self.cap or not self.cap.isOpened():
            LOGGER.error("Failed to open RTSP via GStreamer: %s", self.rtsp_url)
            self.cap = None
            return False
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 1e-3:
            self.fps = fps
        LOGGER.info("RTSP stream opened: %sx%s @ %.2f fps", self.width or "?", self.height or "?", self.fps or "?")
        return True