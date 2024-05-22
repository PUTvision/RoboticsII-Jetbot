from abc import ABC, abstractmethod
from typing import Tuple


class Controller(ABC):
    """Controller interface"""

    stop_monitoring = False

    @property
    def instructions(self) -> str:
        return ""

    @abstractmethod
    def read(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def monitor(self) -> None:
        pass
