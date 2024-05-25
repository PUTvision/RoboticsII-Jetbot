from typing import Tuple, override
from sshkeyboard import listen_keyboard

from src.driving.controller import Controller


class KeyboardController(Controller):
    def __init__(self) -> None:
        self.forward_duration = 0.0
        self.left_duration = 0.0

    @property
    def instructions(self) -> str:
        return "Use 'W' to move forward, 'S' to move backward, 'A' to turn left, 'D' to turn right, and 'SPACE' to stop."

    @override
    def read(self) -> Tuple[float, float]:
        forward = max(min(self.forward_duration, 1.0), -1.0)
        left = max(min(self.left_duration, 1.0), -1.0)
        return forward, left

    @override
    def monitor(self) -> None:
        listen_keyboard(
            on_press=self._handle_press,
            sequential=False,
            delay_second_char=0.05,
            delay_other_chars=0.05,
        )

    def _handle_press(self, key: str, interval: float = 0.1) -> None:
        if key == "w":
            self.forward_duration += interval
        elif key == "s":
            self.forward_duration -= interval
        elif key == "a":
            self.left_duration += interval * 2
        elif key == "d":
            self.left_duration -= interval * 2
        elif key == "space":
            self.forward_duration = 0.0
            self.left_duration = 0.0
