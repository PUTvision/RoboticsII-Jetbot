import os
import time
import cv2
import numpy as np


class Recorder:
    def __init__(self, record: bool = False):
        self.video_capture = cv2.VideoCapture(
            gstreamer_pipeline(flip_method=0, display_width=224, display_height=224),
            cv2.CAP_GSTREAMER,
        )

        self.key = str(time.time())
        self.index = 0

        if record:
            os.makedirs(f"./dataset/{self.key}")

    def get_frame(self) -> np.ndarray:
        ret, frame = self.video_capture.read()

        if not ret:
            raise Exception("No camera")

        return frame

    def record(self, forward: float, left: float):
        with open(f"./dataset/{self.key}.csv", "a") as f:
            f.write(f"{self.index},{forward},{left}\n")

        image = self.get_frame()

        cv2.imwrite(f"./dataset/{self.key}/{self.index:04d}.jpg", image)

        self.index += 1


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )
