import cv2
from inputs import get_gamepad
import click
import math
import threading
import time
from pathlib import Path
import yaml

from PUTDriver import PUTDriver, gstreamer_pipeline


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0.0
        self.RightJoystickX = 0.0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self): # return the buttons/triggers that you care about in this methode
        forward = (self.LeftJoystickY-128)*(-1)/128
        left = (self.RightJoystickX-127)*(-1)/128
        return [forward, left]


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = 0.0
                    self.LeftJoystickY = event.state
                elif event.code == 'ABS_Z':
                    self.RightJoystickX = 0.0
                    self.RightJoystickX = event.state


def save_data(key, index, image, forward, left):
    with open(f'./dataset/{key}.csv', 'a') as f:
        f.write(f'{index},{forward},{left}\n')

    cv2.imwrite(f'./dataset/{key}/{index:04d}.jpg', image)


@click.command()
@click.option('--record', is_flag=True)
def main(record):

    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    joy = XboxController()
    driver = PUTDriver(config=config)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224), cv2.CAP_GSTREAMER)

    print('Turn gamepad sticks around to center.')
    while True:
        forward, left = joy.read()
        # "\r" to print on the same line
        print(f'  Forward: {forward:.4f}\tLeft: {left:.4f}  ', end='\r', flush=True)
        if abs(forward) < 0.05 and abs(left) < 0.05:
            print('\n\nCentered')
            break
    input('Press enter> ')

    if record:
        prev_image = video_capture.read()[1]
        key = str(time.time())
        Path(f'./dataset/{key}/').mkdir()
        index = 0

        print(f'Robot is ready to ride. Grab to: ./dataset/{key}/')

    input('Robot is ready to ride. Press both gamepad knobs and then enter to start...')

    while True:
        ret, image = video_capture.read()
        if not ret:
            print(f'No camera')
            break

        forward, left = joy.read()
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')

        driver.update(forward, left)

        time.sleep(0.1)

        if record:
            save_data(key, index, prev_image, forward, left)

            prev_image = image
            index += 1


if __name__ == '__main__':
    main()
