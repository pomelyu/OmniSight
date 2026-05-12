from abc import ABC
from abc import abstractmethod


class BasicProcessor(ABC):
    """Abstract base class for all OmniSight model processors.

    Subclasses must implement the four-stage pipeline:
    ``preprocess`` → ``model_infer`` → ``postprocess``, orchestrated by ``run``.
    """

    def __init__(self, device: str, model_name: str = None, model_path: str = None) -> None:
        """Initialize the processor.

        Args:
            device: Target inference device (e.g. ``"cpu"`` or ``"cuda"``).
            model_name: Optional model identifier used to locate a checkpoint.
            model_path: Optional explicit path to the model file.
        """
        self.device = device

    @abstractmethod
    def preprocess(self):
        """Prepare raw input for model inference.

        Returns:
            A structure of tensors or metadata ready for ``model_infer``.
        """

    @abstractmethod
    def model_infer(self):
        """Run model inference on preprocessed input.

        Returns:
            Raw model output to be consumed by ``postprocess``.
        """

    @abstractmethod
    def postprocess(self):
        """Decode raw model output into usable predictions.

        Returns:
            Final prediction results (format defined by the subclass).
        """

    @abstractmethod
    def run(self):
        """Execute the full pipeline: preprocess → model_infer → postprocess.

        Returns:
            Final prediction results (format defined by the subclass).
        """
