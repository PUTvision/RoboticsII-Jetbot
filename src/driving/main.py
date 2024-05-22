import threading
import yaml

from time import sleep
from typing import Dict

from src.driving import Controller, AIController, KeyboardController, XboxController
from src.driving.recorder import Recorder
from src.driving.put_jetbot import PUTJetbot


def get_controller(config: Dict, recorder: Recorder) -> Controller:
    name = config["driving"]["controller"]

    if name == "ai":
        return AIController(config["model"]["path"], recorder)
    elif name == "xbox":
        return XboxController()
    elif name == "keyboard":
        return KeyboardController()

    raise ValueError(f"Not a proper controller name: {name}")


if __name__ == "__main__":
    with open("src/config.yml", "r") as f:
        config = yaml.safe_load(f)

    robot = PUTJetbot(config)
    recorder = Recorder(record=config["driving"]["record"])
    controller = get_controller(config, recorder)

    if config["driving"]["record"]:
        print(f"Created a directory under: ./dataset/{recorder.key}")
        print("Starting to record...")

    print(f"Robot is ready {controller.instructions}")
    input("Press enter to start...")

    monitor_thread = threading.Thread(target=controller.monitor)
    monitor_thread.setDaemon(True)
    monitor_thread.start()

    try:
        while True:
            forward, left = controller.read()

            print(f"Forward: {forward:.4f}\tLeft: {left:.4f}")

            recorder.record(forward, left)
            robot.update(forward, left)

            sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop_monitoring = True
        monitor_thread.join()
