# Testing and reproducibility

## Release checks for 0.1.0 — 24 October 2026

- **150 CPU tests passed** (Python 3.12, Linux). New: the skill header parsed by a
  strict YAML parser and checked against the Agent Skills rules, versions that
  must agree across `SKILL.md`, the plugin manifest, both `pyproject.toml` files
  and the changelog, and the `scripts/run` launcher (environment location, lock
  checksum, bad arguments, and a real `doctor` run that writes nothing into the
  skill folder).
- `skills-ref validate` (the Agent Skills reference validator) and
  `claude plugin validate --strict` passed. The previous header failed the
  reference validator, and Pi's strict YAML parser would have skipped the skill.
- The model environment was rebuilt through `scripts/run` outside the skill
  folder (`~/.local/share/sam3-segmentation/envs/`). The generated-shape model
  control gave the same numbers as before: batch/single IoU median **1.000**,
  minimum **0.922**. A direct `sam.py find "red circle"` call on generated shapes
  returned the two circles and not the square, with pixel-exact boxes.
- The full workflow's model steps (no review) on a 960 x 640
  [CC0 corn photo](https://commons.wikimedia.org/wiki/File:Corn_ready_for_removing_kernels_from_cob.jpg)
  produced 1,397 kernel and 11 ear proposals and 594 boundary updates: about 2
  minutes of inference and 1 of refinement on one RTX 3090. The default corn
  budget refused this larger photo before loading weights, as intended; the
  run used the larger budget in [docs/hero/config.json](docs/hero/config.json). A
  quick `find "corn kernel"` on the whole photo returned **no** masks: small
  objects need the crops of the full workflow.
- Agent evals with Claude Haiku, Sonnet and Opus 5.5: see
  [evals/README.md](evals/README.md).
- Six agents (GPT-6 Luna, GPT-6.1 Sol, GPT-6 Astra, Claude Haiku 5.5, Sonnet 5
  and Opus 5.5), each alone with the skill and the original corn photo, ran the
  full workflow on 23 October; results are summarised in the [README](README.md#does-it-work).

## Checks performed locally — 23 October 2026

- The original **106 CPU tests passed** on Linux with Python **3.12.12 and 3.13.13**.
  Five additional non-corn configuration checks brought the suite to **111**; four
  checks for containment matching, cropped geometry and review flags brought it to **115**;
  the direct SAM tool, recorded review decisions and the no-bytecode rule bring it to **144**.
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
- A later local rerun of the **whole corn workflow on the original photograph**
  (copied skill, fresh environments, RTX 3090, BF16) produced 1,350 kernel and 19
  ear proposals with 879 boundary updates; review sheets, validation and export
  passed. Only scalar counts are recorded, in the [reproduction notes](skills/sam3-segmentation/references/lessons.md#reproducing-the-original-corn-run).
  No photo or mask from that run is in the repository.
- A real corn run from a freshly installed copy, with environments outside the
  skill folder, recorded 5 exclusions and 1 uncertain mark before refinement.
  Excluded proposals were not refined and appeared only in `export/excluded.json`;
  no `__pycache__` was written into the skill folder.
- An independent agent (separate folder, only this skill, GPT-6.1 Sol, high
  reasoning) ran the full corn workflow in about 26 minutes: 1,331 kernel and 11
  ear proposals, 872 boundary updates. 98% of the original run's 1,310 kernel
  masks have a match at IoU >= 0.5 (80% at >= 0.9). It had to write its own
  script to remove 19 wrong proposals, which led to the `decide` command.
- An isolated Chromium check passed **desktop, mobile/no horizontal overflow,
  theme variables, selection, original comparison, color fill, flags, category
  changes, zoom, pan and download**, with no page errors.

See the small, privacy-filtered [validation record](tests/evidence/validation.json).
It contains implementation hashes and model/checkpoint identifiers, not private
photographs, mask pools, credentials or GPU UUIDs. The actual private run records
remain excluded from the repository.

A new-account Hugging Face acceptance flow was **not** automated or tested; the
model checks reused an authorized local checkpoint. Windows/macOS base
compatibility were not tested here. CPU and MPS model inference are not implemented.

Known warnings: pycocotools 2.0.11 emits an array-copy deprecation warning with
newer NumPy; exact mask round trips passed. Upstream SAM/timm emit deprecation
warnings as well. These warnings were not treated as evidence of semantic success
or hidden by the tests.

## Base tests

```sh
uv run --locked --project skills/sam3-segmentation --group dev pytest tests
uv run --locked --project skills/sam3-segmentation --group dev ruff check skills/sam3-segmentation/scripts tests install.py
uv run --locked --project skills/sam3-segmentation --group dev ruff format --check skills/sam3-segmentation/scripts tests install.py
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
sh "$S/scripts/run" model model_smoke.py --out /absolute/new-model-check --device cuda:0
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
sh "$S/scripts/run" base --with playwright playwright install chromium
sh "$S/scripts/run" base --with playwright python "$S/scripts/browser_smoke.py" /path/run/export --out /absolute/new-browser-check
```

This starts a loopback-only test server and a dedicated browser/profile. Pass
`--browser-executable` for an appropriate isolated browser on systems such as
NixOS. Do not take over somebody's shared browser or debugging connection.
Playwright/browser downloads are optional, not base runtime dependencies.

## Non-corn qualitative trials

The same implementation was also run on a dog, glass jars, pinned insects,
overlapping mushroom caps and cattle, using explicit image-specific settings.
All exported masks passed native/RLE/PNG checks, but semantic false positives,
parent/part duplicates and fine-boundary defects remained. Independent model
audits are unverified judgments, not ground truth. The [trial report](docs/cross-domain-trials.md)
records both useful masks and failures; the [scalar evidence](tests/evidence/cross-domain-trials.json)
contains no input photos or private masks. A rerun after the containment rule
and review flags were added is summarized at the end of the trial report. The five new tests only validate
those configs and confirm they replace corn-specific semantic/scale fields.

## What these results do not establish

No independent reference masks or public corn photo are included. The historical
1,310 proposal count is not a regression target. Passing these checks does not
establish correct semantics, completeness, cross-hardware bit identity, support
for every GPU/OS, or public redistribution rights. Those require separate evidence.
