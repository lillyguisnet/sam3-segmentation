# Testing and reproducibility

## Checks performed locally — 23 October 2026

- **106 CPU tests passed** on Linux with Python **3.12.12 and 3.13.13**.
- Lint and Python formatting checks passed.
- The complete copied skill ran its demo outside the repository; the installer
  preserved locks/scripts/licence and refused overwriting an existing skill.
- A **fresh isolated model environment** was installed from the supplied lock.
  SAM 3 ran on an **RTX 3090**, Torch **2.10.0+cu128**, BF16, CUDA **12.8**.
- A real-model generated-shape control exercised text inference and four
  independent point prompts with different click counts: **12 batched and 12
  singleton hypotheses**. Per-hypothesis batch/single IoU had median **1.000**
  and minimum **0.922**. This is an implementation diagnostic, not mask accuracy.
- The real end-to-end pipeline on a generated two-oval fixture saved raw masks,
  selected proposals, retained two bounded boundary updates, exported native
  PNG/COCO masks, and resumed cached inference without invalidating unchanged
  downstream evidence. It did not use the protected corn photograph.
- An isolated Chromium check passed **desktop, mobile/no horizontal overflow,
  theme variables, selection, original comparison, color fill, flags, category
  changes, zoom, pan and download**, with no page errors.

See the small, privacy-filtered [validation record](tests/evidence/validation.json).
It contains implementation hashes and model/checkpoint identifiers, not private
photographs, mask pools, credentials or GPU UUIDs. The actual private run records
remain excluded from the repository.

A new-account Hugging Face acceptance flow was **not** automated or tested; the
model checks reused an authorized local checkpoint. The repository has not been
published to GitHub, so no GitHub-hosted CI run has occurred. Windows/macOS base
compatibility, MPS and CPU model inference were not tested here.

Known warnings: pycocotools 2.0.11 emits an array-copy deprecation warning with
newer NumPy; exact mask round trips passed. Upstream SAM/timm emit deprecation
warnings as well. These warnings were not treated as evidence of semantic success
or hidden by the tests.

## Base tests

```sh
uv run --locked --project skills/detailed-segmentation --group dev pytest tests
uv run --locked --project skills/detailed-segmentation --group dev ruff check skills/detailed-segmentation/scripts tests install.py
uv run --locked --project skills/detailed-segmentation --group dev ruff format --check skills/detailed-segmentation/scripts tests install.py
```

These use synthetic masks and a deterministic fake inference adapter. They never
download weights, contact a provider, or establish model quality. They cover:

- Pixel-tight bbox/IoU, including one-pixel and last-row/column cases.
- Complete tiling, explicit EXIF policy, native RLE and PNG round trips.
- Independent prompt grouping without artificial mixed-arity padding.
- Strict config/review schemas, finite values, bounds and image identity.
- Cache identity/checksum rejection, runtime/code changes and interrupted runs.
- Archived stale outputs, provenance binding, bounded refinement and writer locks.
- Safe export IDs, proposal-only status, copied-skill portability and local links.

The included CI workflow is configured to run CPU/base checks only. It does not
require gated weights, a GPU or provider credentials.

## Real-model smoke check

After authorized setup, with `S` set to the absolute skill folder:

```sh
uv run --locked --project "$S/environments/sam3" python "$S/scripts/model_smoke.py" --out /absolute/new-model-check --device cuda:0
```

Use `--checkpoint` for an existing weight file. The tool generates its own simple
input, preserves the hypotheses and reports each batch/single IoU. A large
individual disagreement fails the check even if the median looks good. Retain
failed records and test `--batch-size 1` in a new directory before blaming the
photo or model version. This control does **not** measure natural-image accuracy.

For an end-to-end control, prepare the generated `synthetic-input.png` with
`configs/shapes-smoke.json`, then run infer, audit, select, bounded refine,
validate and export. Repeat infer with exactly the same settings to check cache
reuse. `configs/smoke.json` is a separate sparse corn API control, not a dense
annotation recipe. Do not commit real photographs or private run outputs.

## Optional browser automation

```sh
uv run --locked --project "$S" --with playwright playwright install chromium
uv run --locked --project "$S" --with playwright python "$S/scripts/browser_smoke.py" /path/run/export --out /absolute/new-browser-check
```

This starts a loopback-only test server and a dedicated browser/profile. Pass
`--browser-executable` for an appropriate isolated browser on systems such as
NixOS. Do not take over somebody's shared browser or debugging connection.
Playwright/browser downloads are optional, not base runtime dependencies.

## What these results do not establish

No independent reference masks or public corn photo are included. The historical
1,310 proposal count is not a regression target. Passing these checks does not
establish correct semantics, completeness, cross-hardware bit identity, support
for every GPU/OS, or public redistribution rights. Those require separate evidence.
