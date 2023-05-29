import cv2
from jetbot import Robot


class PUTDriver:
    def __init__(self, config: dict):
        self.robot = Robot()

        self.max_speed = config['robot']['max_speed']
        self.max_steering = config['robot']['max_steering']

        self.left_c = config['robot']['differential']['left']
        self.right_c = config['robot']['differential']['right']

    def update(self, forward, left):
        
        left_speed = forward * self.left_c
        right_speed = forward * self.right_c

        if left > 0:
            left_speed -= left*self.max_steering
        elif left < 0:
            right_speed += left*self.max_steering

        self.robot.set_motors(left_speed=left_speed*self.max_speed, right_speed=right_speed*self.max_speed)


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
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
