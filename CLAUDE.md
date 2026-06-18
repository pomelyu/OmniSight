# OmniSight — Developer Guide

## Project Overview

OmniSight is a Python package for porting open-source ML vision models, with a focus on image processing and face-related detection/tracking/reconstruction. The goal is to standardize interfaces across models, minimize dependencies, and expose tools as AI agent skills for use in this project and others.

## Directory Layout

```
omni_sight/
  <TOOL_CATEGORY>/   # e.g. face_detection — the importable public API
  third_party/       # original third-party code + wrappers for unified interface
  onnx/              # ONNX Runtime utilities (session management, model loading)
  utils/             # shared utilities (algo, hash, visual, file loading)
scripts/             # standalone file/data utility scripts
tests/               # pytest test suite
  resources/         # test fixtures (images, etc.)             
demo/                # runnable CLI examples (will eventually merge into skills)
checkpoints/         # model weight files (not committed)
outputs/             # result output directory (not committed)
docs/                # Splinx document
.claude/
  skills/            # AI agent skill definitions, one per tool category
```

## Core Abstraction

All model wrappers extend `BasicProcessor` (`omni_sight/basic_processor.py`):

```python
class BasicProcessor(ABC):
    def __init__(self, device: str, model_name: str = None, model_path: str = None): ...
    def preprocess(self): ...   # prepare input tensors
    def model_infer(self): ...  # run inference
    def postprocess(self): ...  # decode raw outputs
    def run(self): ...          # end-to-end entry point
```

Every new model wrapper **must** subclass `BasicProcessor` and implement all four abstract methods.

## Adding a New Model

1. Place original/third-party source under `omni_sight/third_party/<model_name>/`
2. Write a wrapper that subclasses `BasicProcessor` with full type annotations and docstrings
3. Expose the wrapper through the appropriate `TOOL_CATEGORY` directory (e.g. `omni_sight/face_detection/`)
4. Add a runnable demo under `demo/`
5. Write tests under `tests/` — use `tests/resources/` for fixture files
6. Modify files under `docs` if a new tool category, a third_party wrapper or runtime dependencies are added.
7. Add a row for each new model that is ready for use to `README.md`

## Dependencies

- **Runtime baseline**: `opencv-python-headless`, `onnxruntime`
- **Dev**: `pytest`, `pylint`, `isort` (see `requirements.dev.txt`)
- Additional dependencies (e.g. `torch`, `transformers`) are acceptable when porting new models.
- **Before adding any new dependency**: check `requirements.txt` first. Only add if not already satisfiable by an existing package.

## Code Style

- **Type annotations are required** on all public functions, methods, and class attributes.
- **Docstrings are required** on all public classes, methods, and functions. The project uses Sphinx for auto-documentation — use Google-style docstrings:
  ```python
  def foo(x: int) -> str:
      """One-line summary.

      Args:
          x: Description of x.

      Returns:
          Description of return value.
      """
  ```
- Import order is enforced by `isort` (config in `setup.cfg`). Run `isort .` before committing.
- Linting via `pylint` (config in `setup.cfg`; `no-member` and `no-name-in-module` are disabled for third-party compatibility).

## Testing

```bash
pytest tests/
```

Test fixtures (sample images, etc.) live in `tests/resources/`.

## Skills

Each tool category should have a corresponding skill under `./claude/skills/` that describes how an AI agent invokes it. The `demo/` scripts serve as interim references until skills are fully defined.

## Documentation

Docstring changes never require editing the docs files. Update the following only when the public module surface changes.

### `docs/api.rst`

Add a section for each new tool category:

```rst
Landmark Detection
------------------

.. automodule:: omni_sight.landmark_detection
   :members:
   :show-inheritance:
```

Add an `automodule` entry for each new third-party wrapper module:

```rst
.. automodule:: omni_sight.third_party.mediapipe.mediapipe_face_mesh
   :members:
   :show-inheritance:
```

### `docs/conf.py`

Add new runtime dependencies to `autodoc_mock_imports`:

```python
autodoc_mock_imports = ["cv2", "numpy", "onnxruntime", "torch"]
```

### `README.md`

Add a row to the "Available Models" table for each new model.
