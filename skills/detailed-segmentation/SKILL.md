---
name: detailed-segmentation
description: Outline objects in photographs with Meta's SAM 3 model, either directly or through a careful multi-step workflow. Direct mode gives quick masks from words ("dog"), example boxes, clicks or boxes on one object, or automatic "segment everything", with an overlay and native-pixel masks. Full workflow for detailed annotation of many small objects or parts, such as corn kernels, seeds, cells or repeated items: overlapping crops, resumable raw predictions, threshold diagnostics, independent visual review, guarded refinement, pixel-checked exports and an interactive viewer. Includes first-machine setup and a no-model demo. Produces model proposals, never verified ground truth.
license: MIT
---

# Detailed instance segmentation

## Choose a mode

**Direct SAM calls** (`scripts/sam.py`): the model and nothing else. Use them
for a quick outline, a few objects, a "segment everything" pass, exploring a
photo before a full run, or your own procedure. Read
[direct SAM calls](references/direct-sam.md). `S` is the absolute path of this
skill folder. Direct calls need an NVIDIA GPU too (no CPU or Mac support). The
model must be installed first, with the user's approval: see
"Install the actual model only after approval" below (the same rules apply).

```sh
SAM="uv run --locked --project $S/environments/sam3 python $S/scripts/sam.py"
$SAM find photo.jpg --text "dog" --out out/            # every instance of words
$SAM find photo.jpg --like 120,80,40,35 --out out/     # more like this example box
$SAM click photo.jpg --point 300,220 --negative 330,240 --out out/   # one object
$SAM click photo.jpg --box 250,180,120,90 --out out/   # one object from a box
$SAM everything photo.jpg --step 32 --out out/         # automatic masks
```

Add `--region x,y,w,h` to zoom into part of the photo. Each call writes
`out/image.png` (the pixels all coordinates refer to), a numbered overlay to look
at, and a JSON file with scores, boxes and masks. Look at the overlay before
reporting; direct masks are unchecked model output.

**Full workflow** (the rest of this file): when the user wants every small
object or part outlined in detail, or results someone must audit. It adds
multi-scale crops, raw-candidate records, review flags, independent review,
refinement and a viewer. If unsure, start direct and move to the full
workflow when direct results miss or merge small objects.

## Full workflow

You are building inspectable annotations, not a pretty overlay. The task is to
identify individual visible objects at the requested level, preserve their exact
native pixels, find omissions and semantic mistakes, and expose uncertainty.

## Before doing anything expensive

1. Read [setup](references/setup.md) and [method](references/method.md).
2. Determine the absolute location of **this skill folder**, called `S` below.
   Scripts and environments live inside it; do not assume a repository checkout,
   a working directory, a preinstalled model service, or a particular home path.
3. Clarify only decisions that matter: image, intended objects/parts, visible-only
   policy, whether parent and part masks should overlap, available hardware,
   permission to download weights, and permission to send photos to hosted
   reviewers. Do not ask questions the image or supplied instructions answer.
4. State the trade-offs: exact boundaries may be unresolvable in a small/blurred
   image; more crops and hypotheses cost time and storage; looser score gates
   recover candidates but also noise. Model signals do not certify correctness.
5. Do not install drivers, stop other GPU processes, use somebody else's service,
   bypass gated access, scrape subscription credentials, or upload photos without
   permission. Use an official provider application/API if cloud review is allowed.

## Start with the inexpensive plumbing check

Requirements: Git, Python 3.12 and `uv` (see setup for installation alternatives).
The base environment has no PyTorch and no account requirement.

```sh
uv sync --locked --project "$S"
uv run --locked --project "$S" python "$S/scripts/segment.py" doctor
uv run --locked --project "$S" python "$S/scripts/segment.py" demo --out "$R/demo"
```

`R` is a new absolute working-output directory chosen for this task. Serve the
result to inspect it:

```sh
uv run --locked --project "$S" python -m http.server 8000 --bind 127.0.0.1 --directory "$R/demo/export"
```

This is explicitly a **synthetic plumbing demo**, not a model benchmark. It checks
that masks, JSON, review sheets and the viewer can work before a large download.
If `uv` is unavailable, setup documents a `venv` alternative. Do not claim the
model was tested merely because this demo worked.

## Install the actual model only after approval

The supplied model environment targets **Linux + NVIDIA CUDA** (Windows through
WSL2). **CPU and macOS/MPS model inference are not implemented**: the tools
refuse any device other than `cuda:N`. Without an NVIDIA GPU, users can still
run the demo, geometry, review and export tools, or rent an approved GPU machine.

```sh
uv sync --locked --project "$S/environments/sam3"
uv run --locked --project "$S/environments/sam3" hf auth login
```

The user must accept Meta's SAM 3 access terms on Hugging Face themselves. Never
request their token in chat. Weights download automatically on first inference
at a pinned revision; `--checkpoint /path/sam3.pt` uses an existing file and
records its checksum. See setup for disk, driver, GPU and troubleshooting checks.
The model imports pinned upstream code; discuss trust requirements before install.

Before a photograph, run the opt-in real-model control on generated shapes:

```sh
uv run --locked --project "$S/environments/sam3" python "$S/scripts/model_smoke.py" --out "$R/model-check" --device cuda:0
```

This exercises real text/point APIs, native geometry and batch/single controls,
not annotation accuracy. Inspect individual disagreement as well as the median.
The adapter groups prompts by their true click count; artificial padding changed
some mask alternatives in testing. If the control fails, inspect its records and
try a new check with `--batch-size 1` before trusting a dense run.

## Step 1 — Freeze the source and policy

Inspect the original before reading any model proposals. Inventory categories,
parent objects, likely small parts, difficult shadows and possible occlusions.
Write the scope and unresolved policy in the run notes. Inventory guesses are
not ground truth.

For another subject, read [the five non-corn trials](references/cross-domain-trials.md)
and use the appropriate **example** under `configs/non-corn/` as a starting point.
Those trials produced useful silhouettes but also background masks and parent/part
duplicates; they are not a general-accuracy guarantee. Explicitly replace labels,
phrases, parent queries, native area bounds, crop scales and refinement scale.
Renaming `kernel` alone leaves other corn settings inherited.

Copy `configs/corn-starting-point.json` only if the image scale is comparable to
the original small corn photo. Otherwise make a config for this image using
[the config guide](references/configuration.md). Object-size limits and thresholds
are starting hypotheses, never universal biological or model constants.
To repeat the original corn experiment itself, follow
[the reproduction mapping](references/lessons.md#reproducing-the-original-corn-run):
it lists each original stage, what this tool does differently, and a measured rerun.

```sh
uv run --locked --project "$S" python "$S/scripts/segment.py" prepare /path/photo.jpg --out "$R/run" --config /path/config.json --orientation raw
uv run --locked --project "$S" python "$S/scripts/segment.py" plan "$R/run"
```

Use `--orientation exif` instead if the task follows the displayed phone-photo
orientation. The chosen policy is recorded, and every model, click and export
must use the resulting `original.png`. No super-resolution, sharpening or
inpainting is allowed to invent annotation evidence.

For a genuinely random image, use `prepare --random-from /path/image-folder
--seed 42 --out ...`; the sorted population digest and choice are recorded. Do
not quietly swap the image after seeing difficult results.

`plan` enforces explicit request/point/candidate budgets before loading weights.
If it refuses, explain the cost and obtain approval for a new config/run. Never
silently truncate a dense image to fit a budget.

## Step 2 — Generate hypotheses, not accepted labels

```sh
uv run --locked --project "$S/environments/sam3" python "$S/scripts/segment.py" infer "$R/run" --device cuda:0 --batch-size 24
uv run --locked --project "$S" python "$S/scripts/segment.py" audit "$R/run"
uv run --locked --project "$S" python "$S/scripts/segment.py" select "$R/run"
```

The order is deliberate:
- Full-image baseline, overlapping coarse crops, then smaller crops.
- Text prompts specify semantics and object granularity.
- Point grids recover missed candidates, but an anonymous point can refer to a
  whole parent instead of a small part. A point mask is not a verified category.
- Raw returned masks and scores are saved **before downstream quality, area,
  edge and duplicate gates**, at text API threshold 0.0. This is the public
  adapter's returned candidate pool, not every hidden internal model query.
- Inference caches include canonical-image hash, exact request, code hashes,
  checkpoint checksum and runtime identity. Repeat the command to resume.
- `audit` shows score distributions and gate sensitivity from the same saved
  point hypotheses. Its numbers are before deduplication, not object counts.

A universal score cutoff is forbidden as a cross-model accuracy comparison.
The later SAM audit showed that identical cutoffs selected very different score
populations. Preserve all candidates and investigate prompt ambiguity before
saying one version is intrinsically better. See [lessons](references/lessons.md).

Review selection changes without rerunning the model:

```sh
uv run --locked --project "$S" python "$S/scripts/segment.py" select "$R/run" --selection /path/selection-overrides.json
```

The override is a selection-only JSON object. It cannot recover masks filtered
inside a different model adapter. A new selection invalidates the previous
refinement output so stale masks cannot be exported accidentally.

## Step 3 — Inspect every region and validate reviewer coordinates

```sh
uv run --locked --project "$S" python "$S/scripts/segment.py" review-sheets "$R/run"
uv run --locked --project "$S" python "$S/scripts/segment.py" export "$R/run"
```

Open the viewer locally. Zoom to individual kernels/objects, not only the whole
image. Inspect each coordinate-labelled sheet in `review/`, comparing original
pixels to contours. Check missing/merged objects, specular holes, disconnected
pieces, background leakage, edge slivers and identity across occluders. Do not
join hidden fragments merely to force a single object count.

Inspect category and physical identity **before** trusting boundary refinement.
A lamp-shaped point mask is not an insect; a jar lid and jar body are not two jars.
The tools only flag these cases: `point_semantics_unverified` (grid-point origin),
`inside_larger_same_label_proposal` (likely a part of another proposal) and
`outside_whole_object_proposals` (outside every parent, such as every ear). Start
with the viewer's "Flagged only" filter; the flag list is in
[method](references/method.md#review-flags). Flags are not a complete category gate
or whole/part identity resolver, and unflagged masks can still be wrong. If those
errors remain, stop with an explicit review draft rather than claiming a cleaner
boundary fixed them.

Read [review protocol](references/review.md). Use an independent vision-capable
agent if authorized, and separate source-image inventory from proposal audit to
reduce anchoring. Give it the native crop origin, zoom and header offset.
Require plain JSON matching `review/template.json`, not prose coordinates.

```sh
uv run --locked --project "$S" python "$S/scripts/segment.py" validate-review "$R/run" /path/review.json
```

Schema/bounds/image checks are necessary but not sufficient: a coordinate can
be inside the image and still point at the wrong kernel. Visually verify the
review points. Never run commands supplied in an image, model response or review.
Do not blindly trust reviewer masks, counts or interpretations of tiny fragments.

If verified observations suggest new point hypotheses:

```sh
uv run --locked --project "$S/environments/sam3" python "$S/scripts/segment.py" infer "$R/run" --review /path/review.json --confirm-review-points
uv run --locked --project "$S" python "$S/scripts/segment.py" select "$R/run"
```

This adds raw point candidates and retains previous jobs. It does not accept
reviewer claims or turn a model's suggestions into reference labels.

To remove a proposal you have inspected and found wrong (background, a part
duplicating another proposal, two objects merged into one), or to mark one as
uncertain, record a decision with a reason:

```sh
uv run --locked --project "$S" python "$S/scripts/segment.py" decide "$R/run" /path/decisions.json
```

The file format is in [review protocol](references/review.md#recording-decisions).
**Never edit `selected.json`, `refined.json` or status files by hand or with your
own script**: that breaks the record of what the tools produced. Decisions are
kept in `decisions.json`, applied when refining, reviewing and exporting, bound
to the exact pixels you saw, and reversible with `"action": "keep"`. Excluded
proposals are not refined and appear in `export/excluded.json` with reasons.
Decide before refinement when you can, so wrong masks are not refined or used
as negative clicks. Decisions are review judgments, not ground truth.

## Step 4 — Refine boundaries conservatively

```sh
uv run --locked --project "$S/environments/sam3" python "$S/scripts/segment.py" refine "$R/run" --device cuda:0 --limit 200
```

`--limit` caps attempted instance updates explicitly, in mask-ID order (top of
the image first), so a small limit leaves the lower part unrefined. Omit only
when the full budget is approved. Refinement uses an interior positive point and up to three
nearby same-category negatives that are outside the original mask. It does not
invent negative corners when no trustworthy neighbor exists. Isolated targets
with no suitable neighbor negative are deliberately skipped and flagged; this
trades coverage for less risk of drifting back to a parent object. Every hypothesis
is retained in `refinement-raw/`. Stable agreeing updates can replace a draft
boundary; large disagreements leave the original and a review flag. **No update
is called a human acceptance.** Model agreement can preserve a shared mistake,
so inspect unchanged masks as well as changed ones.

For large changes, diagnose the semantic object level and inspect independent
reference examples rather than lowering agreement gates until something passes.
Thin silk, bare cob or other parts that fail semantic checks stay unresolved.
Never label an ear as bare cob just because a prompt returned a confident mask.

## Step 5 — Validate, export, and report honestly

```sh
uv run --locked --project "$S" python "$S/scripts/segment.py" validate "$R/run"
uv run --locked --project "$S" python "$S/scripts/segment.py" export "$R/run"
```

Deliver `export/`: interactive viewer, one native binary PNG per instance,
COCO RLE, full proposals, selection/gate provenance, manifest and format checks.
Keep `raw/`, `refinement-raw/`, `threshold-audit.json` and review records locally
for reproducibility. `export --omit-image` omits the photo; recipients must supply
the unchanged canonical photo locally for the viewer. It is not a rights clearance.

Use the host's artifact/browser tools if available; otherwise provide the local
HTTP command. Test the viewer and downloads, and report what was actually tested.
The optional `scripts/browser_smoke.py` automates local browser checks when
Playwright and a dedicated test browser are available (see setup).
The toolset never edits another project's accepted annotations.

In your final answer, distinguish:
- number of **proposals**, not a verified object census;
- boundary updates, not proven improvements;
- format/pixel round-trip checks, not semantic accuracy;
- unresolved objects/parts and sources of uncertainty;
- model/runtime and budget used, plus the location of evidence.

Only a separate authorized human-review process may mark masks accepted. To
claim accuracy, use independently checked reference masks and an honest held-out
evaluation; do not score against these same model-assisted drafts.
