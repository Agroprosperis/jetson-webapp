import cv2
import logging
import numpy as np

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
        return self.cap.read()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        if self.cap is not None:
            capture = self.cap
            self.cap = None
            capture.release()


class V4L2StreamReader(StreamReader):
    def __init__(self, device: str, width: int, height: int, fps: int, pixel_format: str = "MJPG") -> None:
        super().__init__(width=width, height=height, fps=fps)
        self.device = device
        self.pixel_format = pixel_format

    @staticmethod
    def _build_v4l2_gst_pipeline(device: str, width: int, height: int, fps: int, pixel_format: str) -> str:
        # Determine caps based on requested format:
        # 'image/jpeg' for MJPEG cameras, 'video/x-raw' for uncompressed formats.
        
        if pixel_format == "MJPG" or pixel_format == "MJPEG":
            caps = f"image/jpeg,width={width},height={height},framerate={fps}/1"
            decoder = "jpegdec ! videoconvert" 
        else:
            # Assume raw (YUYV, etc.)
            caps = f"video/x-raw,width={width},height={height},framerate={fps}/1"
            decoder = "videoconvert"

        pipeline = (
            f"v4l2src device={device} ! "
            f"{caps} ! "
            f"{decoder} ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "appsink max-buffers=1 drop=true sync=false"
        )
        LOGGER.info("CAP pipeline: %s", pipeline)
        return pipeline

    def open(self) -> bool:
        gst = self._build_v4l2_gst_pipeline(self.device, self.width, self.height, self.fps, self.pixel_format)
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        return self.is_opened()


class FileReader(StreamReader):
    def __init__(self, file_path, fps):
        super().__init__(width=None, height=None, fps=fps)
        self.file_path = file_path

    def open(self):
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

        LOGGER.info("Opened file %s (%sx%s @ %s fps)", self.file_path, self.width or "?", self.height or "?", self.fps or "?")
        return True
