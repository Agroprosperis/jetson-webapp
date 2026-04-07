import subprocess
import re
import logging

LOGGER = logging.getLogger("camera_manager")
PREFERRED_CAMERA_MODE = {"width": 2592, "height": 1944, "fps": 30, "format": "MJPG"}

class CameraManager:
    @staticmethod
    def _is_preferred_mode(mode):
        return mode == PREFERRED_CAMERA_MODE

    @staticmethod
    def get_default_camera_selection():
        cameras = CameraManager.get_available_cameras()
        if not cameras:
            return None, None

        camera = cameras[0]
        modes = camera.get("modes") or []
        if not modes:
            return camera, None

        return camera, modes[0]

    @staticmethod
    def get_available_cameras():
        """
        Parses v4l2-ctl to get devices and their supported modes.
        Returns a list of dicts:
        [
            {
                "device": "/dev/video0",
                "name": "Cam Name",
                "modes": [
                    {"format": "MJPG", "width": 3840, "height": 2160, "fps": 30},
                    ...
                ]
            },
            ...
        ]
        """
        devices = []
        
        # 1. List devices
        try:
            # --list-devices returns chunks like: "Camera Name (usb-0000:00:14.0-1): \n /dev/video0"
            output = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True)
            
            current_name = None
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                
                if not line.startswith("/dev/"):
                    # It's a camera name header
                    current_name = line.split("(")[0].strip()
                else:
                    # It's a device path
                    dev_path = line
                    if current_name:
                        # Only process if we haven't seen this device path (some cams list /dev/video0 and /dev/video1)
                        # Usually we want the first one per physical device for capture
                        devices.append({"device": dev_path, "name": current_name, "modes": []})
                        current_name = None # Reset so we don't add metadata node as a separate cam if listed implicitly
        except Exception as e:
            LOGGER.error(f"Error listing devices: {e}")
            return []

        # 2. Get formats for each device
        for cam in devices:
            cam["modes"] = CameraManager._get_modes_for_device(cam["device"])
            cam["modes"].sort(key=lambda mode: 0 if CameraManager._is_preferred_mode(mode) else 1)

        # Filter out devices with no capture modes (metadata nodes)
        devices = [c for c in devices if len(c["modes"]) > 0]
        devices.sort(key=lambda cam: 0 if any(CameraManager._is_preferred_mode(mode) for mode in cam["modes"]) else 1)
        return devices

    @staticmethod
    def _get_modes_for_device(device_path):
        modes = []
        try:
            # --list-formats-ext gives detailed resolution and fps info
            cmd = ["v4l2-ctl", "-d", device_path, "--list-formats-ext"]
            output = subprocess.check_output(cmd, text=True)
            
            current_format = None
            
            # Regex to parse: Size: Discrete 1920x1080
            size_re = re.compile(r"Size: Discrete (\d+)x(\d+)")
            # Regex to parse: Interval: Discrete 0.033s (30.000 fps)
            fps_re = re.compile(r"\((\d+\.\d+) fps\)")

            lines = output.splitlines()
            for line in lines:
                line = line.strip()
                
                # Detect Format Header, e.g., "[0]: 'MJPG' (Motion-JPEG)"
                if line.startswith("["):
                    # Extract 'MJPG' or 'YUYV'
                    parts = line.split("'")
                    if len(parts) >= 2:
                        current_format = parts[1]
                
                # Detect Size
                elif line.startswith("Size: Discrete"):
                    match = size_re.search(line)
                    if match and current_format:
                        w, h = int(match.group(1)), int(match.group(2))
                        # Temporarily store size to associate with subsequent FPS lines
                        current_size = (w, h)
                
                # Detect FPS (sub-entry of size)
                elif line.startswith("Interval: Discrete"):
                    match = fps_re.search(line)
                    if match and current_format and 'current_size' in locals():
                        fps = float(match.group(1))
                        # Add entry
                        modes.append({
                            "format": current_format,
                            "width": current_size[0],
                            "height": current_size[1],
                            "fps": int(fps)
                        })
        except Exception as e:
            LOGGER.warning(f"Could not query modes for {device_path}: {e}")
        
        return modes
