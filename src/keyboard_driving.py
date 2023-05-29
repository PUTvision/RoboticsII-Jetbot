"""
With this script, you can control jetbot movement with your keyboard (instead of gamepad).
"""

import cv2
import click
import time
from pathlib import Path
import yaml

from PUTDriver import PUTDriver, gstreamer_pipeline

try:
    import keyboard
except ImportError:
    print('keyboard module not found. Install it with:')
    print('pip install keyboard')
    exit(1)


class KeyboardControl:
    def __init__(self) -> None:
        self.press_durations = {
            'w': 0.0,
            's': 0.0,
            'a': 0.0,
            'd': 0.0,
        }

    def update(self, interval=0.1):
        for key in self.press_durations.keys():
            if keyboard.is_pressed(key):
                self.press_durations[key] += interval
            else:
                self.press_durations[key] -= interval
                if self.press_durations[key] < 0:
                    self.press_durations[key] = 0.0

        # clear on space
        if keyboard.is_pressed('space'):
            for key in self.press_durations.keys():
                self.press_durations[key] = 0.0

        time_to_force_multiplier = 5  # how many seconds needs a button to be pressed to reach max speed
        forward = (self.press_durations['w'] - self.press_durations['s']) / time_to_force_multiplier
        left = (self.press_durations['d'] - self.press_durations['a']) / time_to_force_multiplier

        return forward, left


def save_data(key, index, image, forward, left):
    with open(f'./dataset/{key}.csv', 'a') as f:
        f.write(f'{index},{forward},{left}\n')

    cv2.imwrite(f'./dataset/{key}/{index:04d}.jpg', image)


@click.command()
@click.option('--record', is_flag=True)
def main(record):
    keyboard_control = KeyboardControl()

    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    driver = PUTDriver(config=config)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224), cv2.CAP_GSTREAMER)

    if record:
        prev_image = video_capture.read()[1]
        key = str(time.time())
        Path(f'./dataset/{key}/').mkdir()
        index = 0

        print(f'Robot is ready to ride. Grab to: ./dataset/{key}/')

    input('Robot is ready to ride. Press enter to start, and then control with WSAD...')

    while True:
        ret, image = video_capture.read()
        if not ret:
            print(f'No camera')
            break

        forward, left = keyboard_control.update()
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')

        driver.update(forward, left)

        time.sleep(0.1)

        if record:
            save_data(key, index, prev_image, forward, left)

            prev_image = image
            index += 1


if __name__ == '__main__':
    main()
