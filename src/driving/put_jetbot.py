from jetbot import Robot


class PUTJetbot:
    def __init__(self, config: dict):
        self.robot = Robot()

        self.max_speed = config["robot"]["max_speed"]
        self.max_steering = config["robot"]["max_steering"]

        self.left_c = config["robot"]["differential"]["left"]
        self.right_c = config["robot"]["differential"]["right"]

    def update(self, forward, left):
        left_speed = forward * self.left_c
        right_speed = forward * self.right_c

        if left > 0:
            left_speed -= left * self.max_steering
        elif left < 0:
            right_speed += left * self.max_steering

        self.robot.set_motors(
            left_speed=left_speed * self.max_speed,
            right_speed=right_speed * self.max_speed,
        )
