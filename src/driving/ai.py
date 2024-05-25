import onnxruntime as rt
import numpy as np

from typing import Tuple, override

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

    @override
    def read(self) -> Tuple[float, float]:
        return self.outputs

    @override
    def monitor(self) -> None:
        while not self.stop_monitoring:
            inputs = self.recorder.get_frame()
            inputs = preprocess(inputs)

            detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
            self.outputs = postprocess(detections)


def preprocess(img: np.ndarray) -> np.ndarray:
    return img


def postprocess(detections: np.ndarray) -> Tuple[float, float]:
    return detections[0], detections[1]
