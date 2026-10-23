# Detailed Segmentation Skill

**Give an agent a reproducible way to build detailed, inspectable instance masks—not just a convincing overlay.**

A self-contained agent skill plus working tools for multi-scale SAM 3 inference,
raw-candidate preservation, threshold diagnostics, independent visual review,
guarded boundary refinement and pixel-checked exports.

Built from an iterative corn-kernel experiment and its subsequent audit. The
original draft had 1,310 kernel proposals, not 1,310 verified kernels. This repo
shares the approach and its failure lessons, not a claim of perfect annotations.

## What another person gets

- A standard [`SKILL.md`](skills/detailed-segmentation/SKILL.md) that tells the
  agent what to do, when to inspect, when to stop, and how to report uncertainty.
- First-machine setup: no existing model service, private paths, dataset,
  provider subscription or API key assumed.
- A small **CPU-only synthetic demo** before any model download.
- Separate locked base and Linux/NVIDIA SAM 3 environments; pinned code/weights.
- Seeded image selection, explicit orientation, generic overlapping crops and
  request budgets checked before loading weights.
- Resumable raw hypotheses saved before downstream filters, score sweeps,
  duplicate provenance, native coordinates and regression-tested pixel overlap.
- Coordinate-labelled review sheets and a provider-independent review schema.
- An interactive viewer and individual PNG / COCO-RLE / full-proposal exports.
- No automatic “ground truth” labels or changes to existing accepted annotations.

## Install into your agent

Clone this repository, then copy the **whole skill**, not just `SKILL.md`:

```sh
python3 install.py --target codex   # ~/.agents/skills/detailed-segmentation
# or --target claude, --target pi, or --dest /your/skills/detailed-segmentation
```

The installer refuses to overwrite an existing skill and does not install
models, read credentials, or alter agent settings. Use `--dry-run` to inspect it.
The copied folder is self-contained; the repository need not remain at its old
location. A custom destination must keep the directory name `detailed-segmentation`,
matching the [Agent Skills specification](https://agentskills.io/specification). For other agents, point them at the skill file and retain its sibling
scripts/references/environments. Skill discovery varies by agent/version.

Example request:

> Use the detailed-segmentation skill on this photo. Segment each visible corn
> kernel and the whole ears separately. Keep overlapping levels, inspect every
> region, and leave unresolved silk/occluded fragments explicit. Check setup and
> get my approval before large downloads or hosted photo review. Deliver a
> zoomable viewer, binary masks, provenance and an honest review draft.

## Try the no-account demo

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git if
needed. From the repository root:

```sh
export S="$PWD/skills/detailed-segmentation"
export R="$PWD/runs"
uv python install 3.12
uv sync --locked --project "$S"
uv run --locked --project "$S" python "$S/scripts/segment.py" doctor
uv run --locked --project "$S" python "$S/scripts/segment.py" demo --out "$R/demo"
uv run --locked --project "$S" python -m http.server 8000 --bind 127.0.0.1 --directory "$R/demo/export"
```

Open **http://127.0.0.1:8000**. This checks the workflow and viewer using synthetic
shapes, **not model segmentation quality**. No GPU, model weights or credentials.

## Use a real photograph

Read [setup](skills/detailed-segmentation/references/setup.md) first. The supplied
model path targets **Linux x86-64 + NVIDIA**, including Windows through WSL2.
Base tools also work without a GPU. macOS/MPS inference is not implemented.
Allow substantial disk/GPU headroom; obtain download approval and accept Meta's
gated SAM 3 terms yourself. Never send tokens to an agent in chat.

```sh
uv sync --locked --project "$S/environments/sam3"
uv run --locked --project "$S/environments/sam3" hf auth login
# Optional real-model API check on generated shapes, before using a photograph:
uv run --locked --project "$S/environments/sam3" python "$S/scripts/model_smoke.py" --out "$R/model-check"

uv run --locked --project "$S" python "$S/scripts/segment.py" prepare /path/corn.jpg \
  --out "$R/corn" --config "$S/configs/corn-starting-point.json" --orientation raw
uv run --locked --project "$S" python "$S/scripts/segment.py" plan "$R/corn"
uv run --locked --project "$S/environments/sam3" python "$S/scripts/segment.py" infer "$R/corn" --device cuda:0
uv run --locked --project "$S" python "$S/scripts/segment.py" audit "$R/corn"
uv run --locked --project "$S" python "$S/scripts/segment.py" select "$R/corn"
uv run --locked --project "$S" python "$S/scripts/segment.py" review-sheets "$R/corn"
# Inspect original/overlay pairs, validate independent review points, diagnose mistakes.
uv run --locked --project "$S/environments/sam3" python "$S/scripts/segment.py" refine "$R/corn" --device cuda:0 --limit 200
uv run --locked --project "$S" python "$S/scripts/segment.py" validate "$R/corn"
uv run --locked --project "$S" python "$S/scripts/segment.py" export "$R/corn"
```

Serve `runs/corn/export` the same way as the demo. Repeat inference to resume
completed requests; a changed model/runtime belongs in a new run. Do not expect
the historical count: different images, crop grids and settings change outputs.
The example area limits assume small corn-image pixels; inspect before reusing
those settings for other images or categories.
`--limit 200` refines only the first 200 masks from the top of the image; omit it
to refine everything, as the original run did. To repeat the original corn
experiment, see the [stage-by-stage reproduction notes](skills/detailed-segmentation/references/lessons.md#reproducing-the-original-corn-run).

## Does it work beyond corn?

The model and mask tools are not corn-specific, but the original default config
still is. Use the complete [non-corn example profiles](skills/detailed-segmentation/configs/non-corn/)
rather than changing only the object name.

We ran five qualitative trials on a dog, jars, insects, mushrooms and cattle.
Useful masks appeared in every case, but so did background false positives,
part/whole duplicates and missed fine boundaries. More crops were **not always
better**. These are review proposals, not a general automatic annotation result.
See the [full trial report](docs/cross-domain-trials.md), including failures and
a diagnostic score sweep. No private photos or mask sets are in the repository.

## The approach, and its limits

1. Inventory visible objects and nesting **before** inspecting predictions.
2. Freeze canonical pixels and native coordinates. Never invent resolution.
3. Compare full-image, coarse and fine views. Specify the intended object level.
4. Preserve raw candidates before gates; inspect score distributions and misses.
5. Match same-category duplicates by exact mask IoU or near-total containment of a
   similar-sized mask; retain parent/part overlap and flag likely parts.
6. Audit every region independently. Validate coordinates and then inspect them.
7. Refine with positive/neighbor-negative prompts; preserve rejected alternatives.
8. Validate file/pixel consistency and export everything as an unaccepted draft.

The [later audit](skills/detailed-segmentation/references/lessons.md) found that
prompt ambiguity and score filtering strongly affected apparent SAM-version
rankings, and corrected a last-pixel overlap bug. The skill incorporates those
lessons. More masks do not establish higher accuracy; identical confidence
cutoffs do not imply equivalent model selection. A second agent is not an oracle.

This is **not** a model benchmark, general automatic annotation guarantee,
medical annotation tool, or human acceptance interface. No independent reference
masks are included. Pixel-perfect file round trips are not pixel-perfect semantics.

## Documentation

- [Agent workflow](skills/detailed-segmentation/SKILL.md)
- [Setup and troubleshooting](skills/detailed-segmentation/references/setup.md)
- [Method and coordinate contract](skills/detailed-segmentation/references/method.md)
- [Configuration, budgets and run files](skills/detailed-segmentation/references/configuration.md)
- [Independent review protocol and JSON](skills/detailed-segmentation/references/review.md)
- [Historical mistakes and lessons](skills/detailed-segmentation/references/lessons.md)
- [Five non-corn trials, with failures](docs/cross-domain-trials.md)
- [Testing and reproducibility status](TESTING.md)

## Development

```sh
uv sync --locked --project skills/detailed-segmentation --group dev
uv run --locked --project skills/detailed-segmentation --group dev pytest tests
uv run --locked --project skills/detailed-segmentation --group dev ruff check skills/detailed-segmentation/scripts tests install.py
```

Ordinary tests need no weights, GPU, provider accounts or network inference.
Generated runs, local environments and secrets are excluded from git. CI covers
the base tools; real-model and browser smoke checks are documented separately.

## Licence and privacy

[MIT](LICENSE) for this repository's original skill/tools. SAM code, gated
weights, dependency licences, image rights, annotation rights and provider terms
are separate. No weights, private photograph, private masks or credentials are
included. Review current terms before publishing annotations or training models.
`export --omit-image` excludes the source photograph, not the need for rights
review. Optional cloud review requires explicit photo-upload permission.
