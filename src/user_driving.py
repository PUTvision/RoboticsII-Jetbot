import cv2
from inputs import get_gamepad
import click
import math
import threading
import time
from jetbot import Robot, Camera
from pathlib import Path
import yaml

from PUTDriver import PUTDriver


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

    camera = Camera.instance(width=224, height=224)

    if record:
        prev_image = camera.value
        key = str(time.time())
        Path(f'./dataset/{key}/').mkdir()
        index = 0

        print(f'Robot is ready to ride. Grab to: ./dataset/{key}/')


    while True:
        image = camera.value

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
