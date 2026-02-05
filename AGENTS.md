# AGENTS.md

## Scope
- Applies to entire repository.
- No other AGENTS.md files exist yet.
- These notes are for coding agents working here.

## Quick Facts
- Language: Python 3.13 (see `pyproject.toml`).
- Package layout: source code lives in `src/`.
- Entry point: `main.py` (no CLI yet).
- Execution uses `uv` for environment management.
- Documentation is in Spanish under `docs/`.

## Commands (Build / Run / Test / Lint)
- Install dependencies: `uv sync`
- Run the runner script: `uv run main.py`
- Alternative run: `uv run python main.py`
- There is no build step beyond `uv sync`.
- There are currently no tests in the repo.
- If you add tests, use pytest via `uv run pytest`.
- Single test example (when tests exist):
  - `uv run pytest path/to/test_file.py::test_name`
- Keyword test selection example:
  - `uv run pytest -k "metric"`
- No lint/format tools are configured.
- If the team adopts linting, prefer `ruff` + `black`.
- No type-checker is configured (e.g., mypy/pyright).

## Cursor / Copilot Rules
- No `.cursorrules` file found.
- No `.cursor/rules/` directory found.
- No `.github/copilot-instructions.md` found.

## Repo Layout Guide
- `main.py`: simple entry point for running the runner.
- `src/runner/`: CSV-driven experiment orchestration.
- `src/models/`: model adapters for ViT + backends.
- `src/metrics/`: metric definitions and calculators.
- `src/dataset_loaders/`: dataset I/O adapters.
- `src/utils/`: enums, shared types, helpers.
- `docs/`: Spanish design documentation.

## Python Version and Packaging
- Use Python >= 3.13 features (pattern matching, unions).
- Dependencies are managed in `pyproject.toml`.
- Prefer adding new dependencies there, not in ad-hoc files.
- Keep importable code under `src/`.

## Formatting and Layout
- Follow existing 4-space indentation.
- Keep blank lines between import blocks and code.
- Use single blank line between top-level defs.
- Prefer line lengths around 88–100 chars.
- Use parentheses for long expressions rather than backslashes.
- Keep docstrings short and action-oriented.
- Inline comments are ok but avoid noise.

## Imports
- Order imports: standard lib, third-party, local.
- Use explicit relative imports within `src/` (e.g., `from ..utils...`).
- Avoid unused imports; prune when refactoring.
- Prefer importing modules at top-level unless conditional is required.

## Types and Annotations
- Use type hints for public functions and class attributes.
- Use `Protocol` for interfaces (see `BaseModelAdapter`).
- Use `dataclass` for lightweight data containers.
- Prefer modern union syntax (`A | B`) when possible.
- Use `Optional[T]` only when readability improves.
- Use precise container types: `list[str]`, `dict[str, Any]`.
- Preserve `ForwardOutputs` and `ArrayLike` aliases.

## Naming Conventions
- Classes: `PascalCase` (e.g., `SaliencyMetricsCalculator`).
- Functions/methods: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Enum members: `UPPER_SNAKE_CASE` values.
- Filenames: `snake_case.py`.
- Metric names: lower snake with prefixes (e.g., `saliency_auc_judd`).

## Error Handling and Assertions
- Raise `ValueError` for unsupported backends.
- Use `assert` for invariant checks inside core flows.
- Avoid bare `except` or silent failures.
- `NotImplementedError` is used for abstract methods.
- When returning NaNs, use explicit `float("nan")`.

## Backend Handling
- Keep backend matching explicit via `BackendEnum`.
- Use `match`/`case` for backend dispatch.
- Ensure metrics and calculators use the same backend.
- `get_backend_device` is the single source for devices.

## Dataset Loading
- Dataset loaders expose `get_iterator()`.
- Torch loaders return `torch.utils.data.DataLoader`.
- Keep dataset paths configurable when possible.
- Avoid hardcoding local absolute paths when adding new loaders.

## Metric Design
- Metric classes implement `calculate`, `finalize`, `reset`.
- Calculators should call `metric.reset()` before loops.
- Calculators return a `{metric_name: value}` dict.
- Saliency calculators may return extra outputs if needed.

## Model Adapters
- Adapters should expose `forward_features` and `forward_saliency`.
- Keep `forward()` returning a standardized dict.
- Model adapters should set `self.device` from backend.
- Favor `torch.no_grad()` for inference.

## CSV Runner Behavior
- CSV is source of truth for experiment specs.
- Metrics columns use the `metric_` prefix.
- `Runner` updates the CSV in-place unless output is set.
- Treat empty cells as pending metrics (including "nan").

## Documentation
- Documentation is maintained in Spanish.
- Use Mermaid for diagrams in `docs/`.
- Keep docs aligned with actual code behavior.
- Avoid adding new docs unless requested.

## Tests (When Added)
- Use pytest naming: `test_*.py`.
- Keep tests close to feature modules.
- Prefer small, deterministic fixtures.
- If GPU tests are added, mark them clearly.

## Development Hygiene
- Do not introduce new CLIs without alignment.
- Keep changes minimal and focused.
- Avoid large refactors unless requested.
- Avoid magic numbers; define named constants where helpful.
- Prefer explicit over implicit behaviors.

## Suggested Future Tooling (Non-binding)
- `ruff` for linting and import cleanup.
- `black` for formatting if a formatter is desired.
- `pyright` or `mypy` for static typing.
- These are not currently configured; add only on request.

## Agent Checklist
- Read relevant module docs in `docs/` first.
- Keep Spanish comments and docstrings consistent.
- Update `pyproject.toml` if dependencies change.
- Run `uv sync` after adding dependencies.
- Update `README.md` if usage changes.
- Verify `uv run main.py` still works.
- Note any breaking changes in your response.

## End
