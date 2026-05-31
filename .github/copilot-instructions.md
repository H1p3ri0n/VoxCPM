# VoxCPM — Copilot Instructions

VoxCPM is a **tokenizer-free** Text-to-Speech system that generates continuous speech
representations directly via a diffusion-autoregressive architecture (no discrete audio
tokens). The installable package is `voxcpm`, with source under `src/voxcpm/`
(src-layout, configured in `pyproject.toml`).

## Build, test, lint

- Python `>=3.10,<3.12`; runtime needs PyTorch `>=2.5.0` and CUDA `>=12.0` for real inference.
- Editable install with dev tools: `pip install -e ".[dev]"` (the `dev` extra adds
  `pytest`, `pytest-cov`, `black`, `flake8`, `pre-commit`).
- The repo ships a `uv.lock`; `uv` is the intended dependency manager and pins Torch to the
  `pytorch-cu128` index (see `[tool.uv.sources]` in `pyproject.toml`).
- Run the whole test suite: `pytest` (from repo root).
- Run a single test file / test: `pytest tests/test_model_utils.py` /
  `pytest tests/test_model_utils.py::test_resolve_runtime_device_auto_falls_back_to_cpu`.
- Format: `black .` (line length **120**, target py310 — configured in `pyproject.toml`,
  not the black default of 88). Lint: `flake8`.

## Architecture (the big picture)

- **`src/voxcpm/core.py` → `VoxCPM`** is the public façade (re-exported from
  `voxcpm/__init__.py`). It does **not** implement generation; it reads `config.json` from
  the model directory and dispatches on the `"architecture"` field to one of two backends:
  - `"voxcpm2"` → `model/voxcpm2.py::VoxCPM2Model` (current 2B model, 48kHz, 30 languages,
    Voice Design + Controllable Cloning).
  - `"voxcpm"` → `model/voxcpm.py::VoxCPMModel` (legacy 0.5B / 1.5).
  When adding model-level features, change the right backend **and** keep the two parallel
  in shape, since `core.py` calls them through the same method names.
- The generation pipeline (latent space of **AudioVAE V2**) is four stages:
  **LocEnc → TSLM → RALM → LocDiT**. These map to `src/voxcpm/modules/`:
  `locenc/` (local encoder), `minicpm4/` (the MiniCPM-4 LM backbone), `locdit/` (local
  diffusion transformer / CFM), and `audiovae/` (encode/decode to waveform).
  `layers/` holds shared building blocks including LoRA (`layers/lora.py`).
- **Entry points** all funnel into `VoxCPM`:
  - CLI `voxcpm` → `voxcpm/cli.py` (argparse subcommands: `design`, `clone`, `batch`).
  - Web demo: `app.py` (Gradio). `app_old.py` is a prior version — prefer `app.py`.
  - Fine-tuning: `scripts/train_voxcpm_finetune.py` + the WebUI `lora_ft_webui.py`.
- **Training** lives in `src/voxcpm/training/` (`accelerator.py`, `data.py`, `packers.py`,
  `validate.py`, `tracker.py`, …) and is driven by YAML configs in `conf/<version>/`
  (`voxcpm_v2`, `voxcpm_v1.5`, `voxcpm_v1`), e.g.
  `python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml`.

## Key conventions

- **Tests stub heavy dependencies.** Test modules load the target file in isolation via
  `importlib.util.spec_from_file_location` and register fake `voxcpm`/`transformers` modules
  in `sys.modules`, so they run **without** torch/transformers installed (see
  `tests/test_model_utils.py`, `tests/test_cli.py`, `tests/test_validate.py`). When writing
  new tests for a low-level module, follow this pattern instead of importing the full package.
- **Both model backends mirror the same API**: `generate`, `generate_streaming`,
  `generate_with_prompt_cache`, `generate_with_prompt_cache_streaming`, and
  `load_lora_weights`. `core.py` relies on this symmetry — keep new methods present on both.
- **LoRA config is per-version.** There are two distinct `LoRAConfig` classes
  (`model/voxcpm.py` and `model/voxcpm2.py`); the training script imports them as
  `LoRAConfigV1` / `LoRAConfigV2`. Don't assume a single shared config.
- **Device resolution is centralized.** Use the helpers in `model/utils.py`
  (`resolve_runtime_device`, `pick_runtime_dtype`, `get_dtype`) rather than checking
  `torch.cuda.is_available()` inline. Accepted device values: `auto`, `cpu`, `mps`,
  `cuda`, `cuda:N`; `auto`/`None` falls back CUDA → MPS → CPU.
- **`safetensors` is optional at import time.** Code guards it with a
  `try/except ImportError` setting `SAFETENSORS_AVAILABLE`; preserve that fallback rather
  than importing it unconditionally.
- **Diagnostic prints go to stderr** (`print(..., file=sys.stderr)`) so they don't corrupt
  audio/stdout pipelines — follow this for new logging in the inference path.
- **Text-prefix control syntax**: a leading parenthesized description in `text`
  (e.g. `"(A young woman, gentle voice)Hello"`) drives Voice Design / style control. This is
  a string-level convention parsed downstream, not a separate argument.
- Apache-2.0 licensed; source files carry the OpenBMB license header.
