# Contributing

Thank you for your interest in improving this project.

## Before contributing

Please read:

- `README.md` for the system architecture and supported workflows.
- `PROJECT_INFO.md` for project scope and engineering constraints.
- `SECURITY.md` for security-related reporting.

## Development workflow

1. Create a focused branch for your change.
2. Keep changes small and technically justified.
3. Preserve deterministic behavior where the project currently requires it.
4. Add or update tests for behavioral changes.
5. Run the Python test suite:
   ```powershell
   python -m pytest tests\python tests\integration -v
   ```
6. Build and test the embedded target:
   ```powershell
   cmake -S .\embedded -B .\build\embedded
   cmake --build .\build\embedded --config Release
   ctest --test-dir ".\build\embedded" --output-on-failure
   ```
7. Regenerate model artifacts when the model or quantization pipeline changes:
   ```powershell
   python -m python.export.export_model
   ```
8. Review generated files and Git status before committing.

## Pull requests

A useful pull request should describe:

- What changed.
- Why it changed.
- How it was tested.
- Any impact on model accuracy, memory, execution, or deployment artifacts.

Avoid committing generated build directories, caches, local IDE state, or large untracked datasets.

## Code quality

For C/C++ sources, use the repository `.clang-format` configuration. Keep interfaces explicit,
validate inputs, avoid unnecessary dynamic allocation in embedded code, and document assumptions
that affect deployment behavior.
