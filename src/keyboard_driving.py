"""
With this script, you can control jetbot movement with your keyboard (instead of gamepad).
"""
import asyncio
import cv2
import click
import time
from pathlib import Path
import yaml

from PUTDriver import PUTDriver, gstreamer_pipeline

try:
    from sshkeyboard import listen_keyboard
except ImportError:
    print('sshkeyboard module not found. Install it with:')
    print('pip install sshkeyboard')
    exit(1)

class KeyboardControl:
    def __init__(self, config) -> None:
        self.press_durations = {
            'forward': 0.0,
            'left': 0.0,
        }

        self.driver = PUTDriver(config=config)

    def press(self, key, interval=0.1):
        
        if key == 'w':
            self.press_durations['forward'] += interval
        elif key == 's':
            self.press_durations['forward'] -= interval
            
        if key == 'a':
            self.press_durations['left'] += interval*2
        elif key == 'd':
            self.press_durations['left'] -= interval*2

        # clear on space
        if key == 'space':
            for k in self.press_durations.keys():
                self.press_durations[k] = 0.0

        forward = max(min(self.press_durations['forward'], 1.0), -1.0)
        left = max(min(self.press_durations['left'], 1.0), -1.0)

        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')

        self.driver.update(forward, left)

def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    keyboard_control = KeyboardControl(config=config)
    
    input('Robot is ready to ride. Press enter to start, and then control with WSAD...')

    listen_keyboard(
        on_press=keyboard_control.press,
        sequential=False,
        delay_second_char=0.05,
        delay_other_chars=0.05,
    )

if __name__ == '__main__':
    main()

