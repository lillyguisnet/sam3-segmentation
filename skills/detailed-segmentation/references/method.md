# Method: detailed masks through explicit, reviewable stages

## The loop

**Inventory → canonical pixels → full-image baseline → overlapping crops → raw
hypotheses → score/scale diagnosis → selection → independent crop review →
positive/negative refinement → re-review → pixel-checked exports.**

Do not collapse these stages into one opaque “segment everything” command.
The skill supplies an implementation of this loop, not a calibrated annotation
system or a universal best model.

## Image and coordinate contract

- `original.png` is the exact canonical RGB pixel array used throughout.
- `raw` orientation follows decoded file pixels; `exif` explicitly applies EXIF
  display orientation first. Both are supported, but never mixed in one run.
- Coordinates are full-image `[x,y]`, with origin at the upper left. Continuous
  points obey `0 <= x < width`, `0 <= y < height`.
- Boxes are **pixel-tight `[x,y,width,height]`**. If the last occupied column is
  `x_max`, width is `x_max - x_min + 1`, not `x_max - x_min`.
- Crop-local coordinates become global by adding the crop origin. Normalized
  coordinates are used only if a particular adapter requires them. The SAM 3
  image adapter expects native local pixels and performs its prescribed mapping.
- Masks are binary native-size arrays. No inferred hidden surfaces, recreated
  texture, sharpened evidence or smoothed “prettier” contours.
- Different physical levels overlap: a kernel belongs inside an ear mask. Never
  suppress masks across categories or flatten them into one exclusive label map.
- A visibly interrupted instance may have several components. Fragment flags are
  review cues; they do not justify deleting real visible pieces.

## Why multiple scales

Full-image inference supplies context but can miss tiny parts or return their
parents. Crops change the apparent size and reduce the number of competing
objects. The photo is not enhanced; the model still sees only the source pixels.

The defaults illustrate 200px and 120px native crops for a small photograph.
Choose these from the actual object size, not a fixed screen zoom. Adjacent tiles
must overlap more than a typical target's diameter plus boundary margin. The
generic tiler covers the last row/column and includes tiles smaller than the
requested size when the entire image is small. It does not pad invented pixels.

A mask touching an **internal** crop boundary is a likely truncated observation.
The selection recipe can exclude it while an overlapping tile recovers it.
Touching a **true image boundary** is not the same failure: retain genuine slivers.
If overlaps are inadequate, internal-edge exclusion can lose objects; inspect
coverage instead of treating this rule as a proof of completeness.

## Raw candidate pool versus selection

Keep the public adapter's returned candidates before downstream filtering.
Each raw request records model/checkpoint/runtime identity, image hash, crop,
text or point prompts, candidate masks, model signals and duration. Checksum
raw files. Cache keys include code hashes so code changes cannot reuse stale
predictions accidentally.

The pinned SAM 3 text adapter uses detector threshold 0.0. This exposes available
positive-score detector outputs; it is not a claim to dump every hidden internal
query or arbitrary alternative model decoding. Point inference saves all three
returned hypotheses, including empty/low-score masks.

Selection is a separate stage with visible gate counts: empty, area,
text-score, point-quality, point-stability, positive-point membership, and
internal crop-edge gates. Duplicate suppression is same-category only and uses
exact mask IoU, plus a containment rule taken from the original corn merge: a
mask with at least 87% of its pixels inside a same-label mask at most 2.2 times
its size is treated as another copy of it (for example a kernel and the same
kernel with a leaky tip). IoU alone keeps both. The size cap keeps real
parent/part pairs. Every suppressed candidate is linked to the retained
proposal with the rule that matched. All raw masks remain available for a
different policy (`dedupe_containment: null` turns the rule off).

Scores from text detection and point-mask quality are not the same quantity.
The implementation uses an explicit heuristic ordering—text before points,
finer crops before coarse crops, score only within that ordering—not a claim
that the values are comparable probabilities. This is a trade-off: it favors
semantic specificity and local edges but may prefer a fine-crop semantic error.
Check alternative candidates in difficult regions.

## Review flags

Flags mark where to look first. They never delete or accept a mask. Geometry
flags are recomputed after every stage, so a boundary update cannot leave a
stale warning behind.

| Flag | Meaning |
|---|---|
| `inside_larger_same_label_proposal` | ≥87% of the mask lies inside a larger proposal of the same label: often a part (wing, lid, half kernel), not another instance |
| `outside_whole_object_proposals` | More than 15% of a main-label mask lies outside every whole-object proposal (for corn: every ear). Often background or gap fill; also fires where a parent mask is itself too tight. Only computed when whole-object proposals exist |
| `point_semantics_unverified` | Came from an anonymous grid point, not a text prompt |
| `single_pass_only` | No matching proposal from another crop scale |
| `fragmented_boundary` | More than 4% of pixels lie outside the largest connected piece |
| `tiny_instance` | Fewer than 30 pixels |
| `no_trustworthy_neighbor_negatives`, `no_complete_refinement_crop` | Refinement was not attempted |
| `no_stable_agreeing_refinement` | Refinement was attempted; no hypothesis passed the gates, original kept |

Stability is `area(logit > +1) / area(logit > -1)`. It describes threshold
sensitivity, not whether the mask is a corn kernel. For text detections it is
not substituted for the point head's measure. A shared number such as 0.78 must
not be carried between SAM versions as though it selected equal accuracy.

## Refinement is a guarded proposal replacement

1. Choose an interior point using distance to mask background (with explicit
   image padding so full-image masks have sensible interiors).
2. Find nearby same-category proposal centers outside that mask. These are
   candidate negatives, not ground-truth labels. If none exists, do not invent
   negative corners merely to fill a fixed batch shape.
3. Use a complete overlapping local view and independent object prompts.
4. Preserve all returned hypotheses, then test score, stability, source agreement,
   area-change bounds and internal crop clipping.
5. Retain a bounded agreeing update or keep the original with a review flag.
   Both the original mask and the raw refinement remain inspectable.

Agreement is a protection against catastrophic drift, not independent evidence
of correctness. A large erroneous mask may agree with another large erroneous
mask. A mask that changes little can still be wrong. Revisit semantic scope and
use authorized hand edits or independently checked reference examples when a
case cannot be resolved automatically.

A single point can belong to a kernel **and** an ear. Negatives or an instance
box can specify the intended scale, but a tight box supplies strong extent
information. Do not count a small returned mask as evidence of accurate edges.
SAM 3 concept exemplar boxes and interactive instance boxes are different APIs;
this implementation does not silently substitute one for the other.

## What validation proves

It proves checksum consistency, native dimensions, unique instance IDs,
pixel-tight boxes, nonempty masks, area consistency, binary PNG/COCO-RLE round
trips, and unaccepted status. It does **not** prove correct object names, one
mask per physical object, complete recall, accurate boundaries, or safe public
licensing. Those require different evidence.

## Reproducibility versus exact replay

The historical corn draft had 1,310 kernel proposals and 854 retained boundary
updates. It was not human-verified and those numbers are not a target metric.
This portable implementation is a refactoring of the approach with corrected
geometry, generic coverage, independent environments and stronger provenance.
It deliberately does not promise bit-for-bit recreation of that one-off run.
Different crop positions, precision, hardware, versions and review points may
change the output. The protected photo and its derived masks are not included.
See [lessons](lessons.md#reproducing-the-original-corn-run) for a step-by-step
mapping from the original scripts to this tool and a measured rerun.

For exact replay on your own image, keep the canonical photo, immutable config,
locks, model checksum, raw candidate files, selection policy, review JSON and
refinement records together. A folder of final PNGs alone cannot explain how
those masks were obtained.
