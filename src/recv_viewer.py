#!/usr/bin/env python3
import argparse
import cv2
import numpy as np

from datetime import datetime
from hud import draw_hud


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-path", default="/app/recv_output.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    in_pipeline = (
        f"udpsrc port={args.port} "
        "caps=\"video/mpegts, systemstream=true, packetsize=(int)188\" ! "
        "tsdemux ! h264parse ! avdec_h264 ! "
        "videoconvert ! "
        "appsink max-buffers=1 drop=true sync=false"
    )

    print("IN pipeline:", in_pipeline)

    cap = cv2.VideoCapture(in_pipeline, cv2.CAP_GSTREAMER)
    print("IN opened:", cap.isOpened())
    if not cap.isOpened():
        raise RuntimeError("Failed to open UDP input")

    writer = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No frame")
            break

        recv_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        fps_display = None  # or a real number if you measure it

        hud_frame = draw_hud(
            frame,
            send_ts=None,        # SEND text is already burned into the frame
            recv_ts=recv_ts,
            latency_ms=None,     # you can wire this up later
            fps=fps_display,
            panel_bg=(0, 0, 0, 0)
        )

        # lazy-init writer once we know frame size
        if args.save_video and writer is None:
            h, w = hud_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                args.video_path,
                fourcc,
                args.fps,
                (w, h),
                True,
            )
            print("VideoWriter opened:", writer.isOpened())
            if not writer.isOpened():
                raise RuntimeError("Failed to open VideoWriter")

        if writer is not None:
            writer.write(hud_frame)

        cv2.imshow("recv", hud_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
