from typing import Tuple, override
from inputs import get_gamepad

from src.driving.controller import Controller


class XboxController(Controller):
    def __init__(self) -> None:
        self.left_y = 0.0
        self.right_x = 0.0

    @property
    def instructions(self) -> str:
        return "Use the left stick to move forward and the right stick to turn."

    @override
    def read(self) -> Tuple[float, float]:
        forward = (self.left_y - 128) / (-128)
        left = (self.right_x - 127) / (-128)
        return forward, left

    @override
    def monitor(self) -> None:
        while not self.stop_monitoring:
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_Y":
                    self.left_y = event.state
                elif event.code == "ABS_Z":
                    self.right_x = event.state
