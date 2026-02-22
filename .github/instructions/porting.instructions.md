# OmniSight Model Porting Instructions

Use this guide when porting an open-source model into OmniSight.

## Porting Principles
- Keep only the minimum files required for inference-only usage.
- Prefer ONNX over PyTorch whenever ONNX inference is available and quality is acceptable.
- Keep dependencies minimal and avoid introducing packages unless clearly required.

## Required Third-Party Folder Layout
Place original upstream files under `third_party/` using this structure:

```text
third_party/
	model_name/
		model_name.py
		networks/
			# pytorch model files if needed
		utils/
			# model-specific utils if needed
```

Notes:
- Keep only files needed for inference.
- Include `networks/` and `utils/` only when needed.

## Processor Contract
New model processors must inherit `BasicProcessor` and implement all required methods. Put model processor on the top of the file.

```python
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
        # preprocess
        # model_infer
        # postprocess
        pass
```

Implementation requirements:
- `preprocess`: validate and transform raw inputs into model-ready tensors/arrays.
- `run`: execute inference (prefer ONNX runtime path when available).
- `postprocess`: convert model outputs into standardized OmniSight outputs.

## Integration Checklist
When adding a new model, complete all items below:

1. Port only inference-related code and assets.
2. Implement ONNX inference path first; add PyTorch path only if necessary.
3. Place upstream source files in `omni_sight/third_party/model_name/` with the required structure.
4. Implement a processor that inherits `BasicProcessor` and defines `preprocess`, `run`, and `postprocess`.
5. Add runnable demo code in `demo.py`.
6. Add the demo usage script/command to `README.md`.
7. Add a new skill in `.agents/skills` describing how to use this model in OmniSight.

## Definition of Done
A model port is complete only when:
- Inference works with the preferred ONNX path (or documented PyTorch fallback).
- `demo.py` runs successfully for the model.
- `README.md` includes the demo script/usage.
- A model usage skill exists under `.agents/skills`.
