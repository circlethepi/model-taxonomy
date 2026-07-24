# Changelog

<!-- When opening a PR, always add an entry under "Unreleased" describing what changed and why. -->

## Unreleased

### LoRACache — multi-config storage + adapter listing

**`src/cache/lora_cache.py`**

- New storage layout: extracted representations now live in a `{config_hash}/` subdir
  under each adapter folder, so multiple extraction configurations can coexist.
  Raw PEFT files (`adapter_model.safetensors`, `adapter_config.json`) are untouched.
- `exists()`, `load()`, `load_config()`, `save()` now take an `extraction_config: dict`
  parameter to identify which config-hash subdir to use.
- `save()` now accepts and stores `layer_lengths: list[int]` in `config.json`.
- New methods: `list_raw_adapters()`, `adapter_status()`, `adapter_count()`,
  `list_representations()`.
- `list_adapters()` updated to detect processed adapters via config-hash subdirs.

**`src/taxonomy/structural.py`**

- Removed `n_components` parameter and `_truncate_pad` — LoRA weight vectors are
  now stored at their full natural length. Rows of different length are zero-padded
  to the longest so the matrix can be stacked; original lengths saved in `config.json`.
- Added shorthand layer/projection selectors: `layer_indices` and `projections`
  (matching `load_lora_weights` conventions). `layer_names` still works for explicit
  full-prefix control and takes precedence.
- `_find_lora_pairs` updated to apply the new filters via architecture-agnostic
  regex on actual parameter names.
- `extract()` now builds a single `extraction_config` dict and passes it to all
  `LoRACache` calls so the right config-hash subdir is always targeted.

**`src/metrics/cka.py`, `src/metrics/frobenius.py`**

- Added TODO comments noting that these metrics assume uniform-row
  `ModelRepresentation.matrix` and will need updating before structural taxonomy
  representations can flow through the full pipeline.
