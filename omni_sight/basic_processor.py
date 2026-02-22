from abc import ABC
from abc import abstractmethod


class BasicProcessor(ABC):
    def __init__(self, device: str, model_name: str = None, model_path: str = None):
        self.device = device

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def model_infer(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass

    @abstractmethod
    def run(self):
        pass
