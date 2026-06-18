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

## JSON Experiment Config
- The JSON experiment flow must remain backward-compatible with the legacy CSV flow.
- Treat `src/runner/config_validation.py` as the current source of truth for JSON config validation.
- Treat `src/runner/json_source.py` as the place where `defaults`, `dataset_defaults`, and per-experiment config are merged.
- `dataset_defaults` is scoped by metric family (`levels`, `visturing`, `tid`, `nights`, `saliency`); do not mix family-specific keys into global `defaults`.
- `results.csv` is only for the JSON flow; do not change legacy CSV persistence unless explicitly requested.
- Validation errors should fail early and include valid options whenever the domain is closed.

## Extending Metrics And Models
- If you add a new metric, update all of the following together:
  - metric loading in `src/metrics/__init__.py`
  - JSON validation rules in `src/runner/config_validation.py`
  - metric family resolution in `src/runner/json_source.py` if it belongs to a new family
  - runner dispatch in `src/runner/base.py` if the execution path changes
  - JSON validation tests in `tests/unit/test_json_experiment_source.py`
  - runner tests in `tests/unit/test_runner_json.py` when config is consumed by calculators
- Prefer one declarative source of truth for valid config keys and valid values; avoid scattering the same options across multiple files.
- If a metric uses family-specific dataset paths, make sure the valid keys align with `dataset_defaults` and the calculator/factory signature.
- For `visturing_*` metrics, avoid instantiating metric classes through the generic path if that would trigger default downloads or side effects.
- If you add or expand model support, update both model resolution and model validation together.
- If model support becomes dynamic (for example `timm` model names), add a dedicated resolver/registry abstraction instead of encoding the logic directly in many validation sites.

## Agent Change Checklist
- When adding a metric, do not stop at the metric class itself; wire validation, runner dispatch, and tests in the same change.
- When adding config options, make sure invalid keys and invalid values produce actionable `ValueError` messages with allowed options.
- When adding a new dataset family, update `dataset_defaults` handling and add tests proving the merge behavior.
- When adding model aliases or new backends, add acceptance tests and invalid-input tests.
- Prefer tests that prove no unintended downloads or generated artifacts happen during unit/integration runs.

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
- Update JSON config validation and tests when adding metrics/models/config options.
- Verify `uv run main.py` still works.
- Note any breaking changes in your response.

## End
