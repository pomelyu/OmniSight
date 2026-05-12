from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import onnxruntime as ort

from .file_loader import FileLoader


class OnnxLoader(FileLoader):
    """FileLoader subclass that creates ONNX Runtime inference sessions.

    Example:
        >>> loader = OnnxLoader()
        >>> loader.set_file(
        ...     "model-abc123.onnx",
        ...     urls=["https://example.com/model.onnx"],
        ...     identifier="my_model",
        ... )
        >>> session = loader.get_onnx_session("my_model", device="cuda")
    """

    def get_onnx_session(
        self,
        identifier: str = "default",
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> ort.InferenceSession:
        """Return an ONNX Runtime session for the specified model.

        If ``model_path`` is not provided, the file is resolved via
        :meth:`~omni_sight.utils.file_loader.FileLoader.get_path`, which
        downloads the model on first access.

        Args:
            identifier: Key previously registered with
                :meth:`~omni_sight.utils.file_loader.FileLoader.set_file`.
                Defaults to ``"default"``.
            model_path: Explicit path to the ``.onnx`` file. When given,
                ``identifier`` is ignored. Defaults to ``None``.
            device: Target device — ``"cpu"`` or ``"cuda"``.
                Defaults to ``"cpu"``.

        Returns:
            Configured :class:`onnxruntime.InferenceSession` ready for inference.
        """
        if not model_path:
            model_path = self.get_path(identifier)

        providers = self._build_providers(device)
        session = ort.InferenceSession(
            model_path,
            providers=providers,
        )
        return session

    @staticmethod
    def _build_providers(device: str) -> List[str]:
        """Select ONNX Runtime providers from device string.

        Args:
            device: Device string (e.g. ``"cpu"`` or ``"cuda"``).

        Returns:
            List of ONNX Runtime provider strings ordered by preference.
        """
        normalized = (device or "cpu").lower()
        available = ort.get_available_providers()

        if normalized.startswith("cuda"):
            requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return [provider for provider in requested if provider in available]

        return ["CPUExecutionProvider"]
