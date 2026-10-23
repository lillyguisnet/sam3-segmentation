# Configuration and run files

Configuration files are JSON overrides merged into `DEFAULT_CONFIG` in
`scripts/core.py`. Top-level/setting typos are rejected. `passes` replaces the
entire pass list. Inspect the saved complete config in `manifest.json`.

## Minimal example

```json
{
  "label": "seed",
  "phrases": ["seed", "individual seed"],
  "whole_objects": [],
  "passes": [
    {"name": "whole", "size": null, "overlap": 0, "grid_step": null},
    {"name": "local", "size": 160, "overlap": 50, "grid_step": 8}
  ],
  "selection": {
    "min_area": 5,
    "max_area": 900,
    "text_score_min": 0.2,
    "point_quality_min": null,
    "point_stability_min": 0.83,
    "dedupe_iou": 0.52,
    "exclude_internal_crop_edges": true
  },
  "budget": {"max_jobs": 256, "max_grid_points": 25000, "max_candidates": 100000}
}
```

These are proposal-selection starting settings, not validated accuracy settings.
`null` disables the relevant gate; it does not mean a zero probability. Scores
from different model heads must not be pooled as accuracy probabilities.

## Fields

| Field | Meaning |
|---|---|
| `label` | Main small-object category; grid candidates are **unverified** category hypotheses |
| `phrases` | Short noun phrases, not long instructions to the segmentation model |
| `whole_objects` | Additional `{label, phrase}` full-image parent/part queries; these are not subject to the small-object area cap |
| `passes[].name` | Unique stable pass identifier |
| `passes[].size` | Native square crop side, or `null` for the full image |
| `passes[].overlap` | Native overlap, strictly smaller than crop size |
| `passes[].grid_step` | Native point spacing, or `null` to skip point-grid inference |
| `selection.min_area`, `max_area` | Small-object native mask area gates; `max_area: null` removes the upper cap |
| `text_score_min` | Text-detector score gate, applied after raw outputs are saved |
| `point_quality_min` | Point-head quality gate; default `null` because the historical cross-version audit invalidated a universal cutoff |
| `point_stability_min` | Point mask threshold-sensitivity gate, not semantic certainty |
| `dedupe_iou` | Exact same-category mask IoU suppression threshold |
| `exclude_internal_crop_edges` | Exclude masks clipped by artificial crop boundaries, not true photo borders |
| `refinement.crop_size` | Complete local view used for interior positives and neighbor negatives |
| `refinement.quality_min`, `stability_min` | SAM 3-specific heuristic gates for boundary updates |
| `refinement.agreement_min` | Minimum IoU between update and initial mask |
| `refinement.supported_agreement_min` | Stricter IoU gate when original has multiple-pass support |
| `refinement.area_ratio_min`, `area_ratio_max` | Allowed updated/initial area change |
| `budget.max_jobs` | Maximum planned text/grid request jobs |
| `budget.max_grid_points` | Maximum independent point prompts; refinement checks the same prompt cap separately |
| `budget.max_candidates` | Maximum estimated/actual returned hypotheses in inference |
| `seed` | Model random seed, recorded; not a guarantee of cross-hardware bit identity |

The model text adapter exposes up to its available query slots at API threshold
0.0. The planner uses 200 queries per text job as an upper-bound estimate for the
pinned model. Actual counts are also checked. No cap silently clips an object set.
Raw storage can be large even when selection retains few masks.

## Choosing crop and area scales

Inspect several typical and small targets at native resolution. If a typical
object has diameter `d`, start with crop side roughly `8d–16d`, overlap larger
than `d` plus boundary margin, and a grid spacing that can enter the smallest
resolvable instance. These are operational heuristics, not guarantees. Do not
choose area limits from a displayed browser zoom or carry corn's 650px cap to
another photograph without checking it.

The defaults are for a small image. On a very large photo they intentionally
hit the budget guard. Use an approved ROI strategy or a different coarse-to-fine
plan rather than issuing millions of requests blindly. The current tool does
not infer automatic semantic ROIs; do not describe that feature as implemented.

## Diagnostic selection profiles

A selection override contains only fields from `selection`, for example:

```json
{"point_quality_min": null, "point_stability_min": null}
```

Run `select RUN --selection profile.json` to inspect the same saved pool under
another policy. `audit` sweeps point quality/stability without rerunning the
model and reports **hypotheses passing gates before duplicate suppression**.
That is intentionally not a comparison of verified objects or model accuracy.

The historical `.78` quality gate is in `configs/corn-historical-gates.json` as
a diagnostic profile, not the recommended universal default. Keep this profile
separate from threshold calibration on independent checked masks.

## Run layout

```text
manifest.json                 immutable canonical checksum, dimensions, config, seed
original.png                  canonical source pixels; private unless rights permit
raw/index.json                partial/complete request index and checksums
raw/<content hash>.json.gz    returned masks before downstream selection gates
threshold-audit.json          signal distribution and gate sensitivity
selected.json                 selection settings, gates, suppressed links, proposals
review/*.png                  coordinate-labelled source/overlay pairs
review/template.json          review schema skeleton with exact image checksum
review/sheets.json            full crop/render coordinate mappings
refinement-raw/*.json.gz       all returned update hypotheses, prompts and identity
refined.json                  initial masks, chosen updates, failures and proposals
export/                       viewer, masks.zip, COCO, full proposals, checks
```

Inference can resume only with the same backend/runtime identity. New review jobs
are added without losing previous jobs. Different model/precision/batch/device
settings require a new run so records cannot be mixed invisibly. Selection may
be rerun from raw masks without redownloading weights. A new selection removes
the obsolete `refined.json` from the active run; prior derived files are archived
under `history/`. New raw evidence likewise archives stale selection, refinement,
review sheets and exports. Selections bind to the raw index checksum; refinements
bind to the selected-result checksum. Exports reject stale or incomplete stages.
Mutating CLI commands use a per-run lock and refuse simultaneous writers.

`export` validates the current result first. It always keeps proposal status;
it never overwrites another annotation system's accepted source of truth. The
viewer groups categories without forcing their pixels to be mutually exclusive.

## Adapting another model

Implement the narrow backend protocol shown by `sam3_backend.py`:
- `identity`: model/source/checkpoint/runtime fingerprint, without secrets;
- `text(PIL image, phrase)`: native local RLE, score and declared signal meaning;
- `points(PIL image, prompts)`: independent object hypotheses with prompt index,
  hypothesis index, native local RLE, score and stability when supported.

This is a documented extension point, **not** a currently shipped SAM 1/2,
Moondream, hosted segmentation or generic remote-service adapter. Test coordinates,
labels, resize transforms, batches, mask membership, empty outputs and an easy
single object before comparing models. Preserve each head's native score meaning.
