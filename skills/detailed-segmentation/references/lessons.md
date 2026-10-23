# Lessons from the corn experiment and subsequent audit

These are observations from one small, unverified corn photograph, not a
published benchmark. No protected image or generated mask set is distributed here.

## What the first run did

A seeded draw selected one 480×500 photograph from a 121-image corn collection.
The first whole-image SAM 3.1 text/video-service path returned whole ears but
zero kernel masks. Cropping that path recovered only 91 kernel proposals.
Switching to the dedicated SAM 3 still-image path, overlapping crops, text and
point proposals produced 924 retained coarse proposals. Smaller crops produced
1,309; matching and guarded refinement left 1,310 with 854 boundary updates.
Those counts were always **proposals**, not a verified census or accuracy result.

A photo that looks densely outlined can still contain false positives, missed
kernels, internal fragments, merged identities, semantic errors and invented
interpretations of occlusion. Thin silk and exposed-cob identity remained
unresolved. Semantic inspection rejected “bare cob” masks that actually outlined
kernels, ears or background. A model returning a mask is not proof it understood
the requested part.

## Reviewer mistakes were real too

An independent reviewer misread a whole-photo montage and assigned a tiny ear
fragment to the wrong location. Individual native close-ups corrected that
interpretation. Another returned an x coordinate outside the 480px image. Bounds
validation discarded it. The experience motivated explicit crop origins, header
and zoom metadata, strict review JSON, image checksums and visual point checks.
A second agent's agreement is useful evidence, not an oracle.

## The SAM-version comparison needed correction

A later matched audit used the same 6,640 whole-image grid points, one positive
per independent prompt, three returned hypotheses per point and no negatives or
text. It preserved 19,920 hypotheses per model before downstream gates.

An error in the initial SAM 1/2 duplicate-removal code treated a geometric box's
width/height as pixel-slice extents, omitting the last occupied row and column
from intersections. Corrected overlap calculation reduced the SAM 1 small-proposal
count from 1,300 to 1,240 under the original gates. This was a postprocessing
error, not an upstream model error. This repository tests full-array IoU against
cropped IoU, including one-pixel and last-row/column cases.

On the same corrected saved pools, retained small proposals (8–650 native pixels)
changed substantially with filtering:

| Gate policy | SAM 1 | SAM 2.1 | SAM 3 |
|---|---:|---:|---:|
| Quality ≥0.78, stability ≥0.83 | 1,240 | 138 | 113 |
| No quality cutoff, same stability | 1,333 | 562 | 407 |
| Neither cutoff | 2,006 | 1,857 | 2,248 |

These heavily overlapping candidates are **not verified kernels**. Removing gates
also admits fragments and noise. The empirical small-candidate quality-score
populations differed greatly. That does not prove the newer quality heads are
miscalibrated: low scores might correctly identify poor masks. It does prove
that identical numerical cutoffs cannot be treated as equivalent selection or
an intrinsic accuracy comparison.

The original SAM 1 path used FP32; SAM 2.1/3 used BF16. A small FP32 sensitivity
check did not explain the count gap, but it did not establish universal precision
invariance. The appropriate model-specific input preprocessing was retained.

## A point is not an object-level instruction

A point inside a kernel is also inside an ear. On twelve provisional targets,
the highest-scored mask was small for 9/12, 2/12 and 4/12 targets for SAM 1,
SAM 2.1 and SAM 3. Adding three neighboring negatives changed this to 9/12,
9/12 and 10/12. A tight box produced small masks for all twelve for all versions.

This measured object **scale**, not correct boundaries. The targets came from
the earlier model-assisted draft, not independent labels; boxes encoded strong
prior extent information. The supported conclusion was prompt ambiguity and
selection effects, not an intrinsic accuracy winner.

## Other models were controls, not magic repairs

In the local tests, Florence-2/YOLOE mostly returned parent regions, a direct
language-generated polygon route produced schematic geometry, and Moondream's
native outline path failed even on an easy control. Boxes and points did not
become valid masks merely by being rasterized. The Moondream failure remained a
local generation-path problem; it did not justify a general claim that the model
could not segment. This skill therefore ships one tested still-image adapter
and does not promise that every installed model is interchangeable.

## Fresh-adapter controls found additional implementation issues

A new isolated environment exposed two practical problems worth checking on any
port: the pinned upstream builder only moved weights for the literal device
string `cuda`, not `cuda:0`, and artificially padding mixed click-count prompts
changed some mask alternatives. The portable adapter now explicitly moves the
complete model onto the selected device and groups independent prompts by their
true click count. Its optional real-model smoke tool compares every hypothesis
in batch versus singleton calls, reporting the minimum as well as the median.
Remaining numerical differences are recorded, not described as perfect identity.
These are implementation checks on generated shapes, not a natural-image benchmark.

## What changed in the portable skill

- No family service, private cache path, private dataset or subscription assumed.
- Separate small base environment and pinned, gated model environment.
- Generic full-coverage tiling instead of hardcoded image-specific starts.
- Raw outputs retained before selection, with inspectable gate counts/sweeps.
- Correct pixel-tight overlap, regression tests and overlap-preserving exports.
- No universal point-quality gate by default; historical gates remain diagnostic.
- Scores from separate heads are not treated as calibrated probabilities.
- Invalid review coordinates are errors, not clipped or silently used.
- Missing neighbor negatives remain missing rather than fabricated.
- Replacement decisions are called boundary updates, never automatic acceptances.
- No protected corn image or historical counts used as a regression target.

The cost of these choices is extra storage, explicit setup, more review and no
promise of the same final count. That is preferable to a seemingly easy one-shot
recipe that conceals the uncertainty and cannot be audited.
