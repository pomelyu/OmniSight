---
description: 'General guidelines'
applyTo: "**"
---

# OmniSight Copilot Instructions

## Coding Style and Naming
- Keep code clear, consistent, and minimal.
- Follow existing project naming and module structure patterns.
- Prefer explicit, readable APIs so model backends can be swapped with minimal code changes.
- Use English for all code comments.
- When writing Python code, always follow the guidance in `instructions/python.instructions.md`.

## Technology Stack and Libraries
- This project collects open-source ML models and exposes a unified interface.
- Primary frameworks are **PyTorch** and **ONNX**.
- Keep dependencies minimal and only add packages when clearly required.

## Architecture Patterns
- Maintain consistent abstractions across model backends.
- Avoid backend-specific behavior leaking into shared interfaces.
- For Python imports, use relative imports when the target module is in the same directory or a subdirectory; use absolute imports in other cases.
- Keep each file's dependencies on other repository files to a minimum.
- Consult relevant tools in `.agents/skills` before implementation.
- For new model integrations, check and follow `instructions/porting.instruction.md` first.

## Security and Error Handling
- Validate external inputs and model/config parameters.
- Fail with clear, actionable error messages.
- Do not silently ignore exceptions that affect correctness or reproducibility.

## Dependency and Package Files
- Keep dependency declarations alphabetically sorted in package manager files.
- Apply sorting whenever editing files such as `requirements.txt` and `requirements.dev.txt`.

## Documentation Standards
- Keep docstrings and README updates concise and task-focused.
- Document any new model support, required assets, and backend-specific constraints.
