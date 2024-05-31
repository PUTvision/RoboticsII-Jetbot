import onnxruntime as rt
import numpy as np

import cv2

from typing import Tuple, override, List

from src.driving.controller import Controller
from src.driving.recorder import Recorder


class AIController(Controller):
    def __init__(self, model_path: str, recorder: Recorder) -> None:
        self.sess = rt.InferenceSession(
            model_path,
            providers=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.recorder = recorder

        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
        self.outputs = 0.0, 0.0
        self.weights = [0.4, 0.4, 0.2]

    @override
    def read(self) -> Tuple[float, float]:
        return self.outputs

    @override
    def monitor(self) -> None:
        while not self.stop_monitoring:
            inputs = self.recorder.get_frame()
            inputs = preprocess(inputs)

            detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
            self.outputs = postprocess(detections, self.weights)


def preprocess(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (224, 224))


def postprocess(detections: np.ndarray, weights: List[float]) -> Tuple[float, float]:
    weights_sum = sum(weights)
    forward, right = 0, 0
    for i in range(0, 3, 2):
        forward += weights[i] * detections[i]
        right += weights[i] * detections[i + 1]
    forward /= weights_sum
    right /= weights_sum
    return (forward, right)
