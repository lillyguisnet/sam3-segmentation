# Five non-corn trials: what transfers, and what does not

## Answer in brief

The SAM 3 image backend and mask/coordinate tools are not corn-specific. The
original **default configuration is**: kernel labels, corn text prompts, an ear
query, a 650-native-pixel area cap, 120/200px tiles, and a 120px refinement view.
These must not be silently carried over to unrelated tasks.

The approach can produce useful masks for other subjects, but these trials do
**not** establish a general automatic high-quality annotation system. Distinct
objects, parts of those objects, background regions and duplicate hypotheses
can all have stable, attractive contours. Semantic and physical-instance review
must come before trusting the output or its count.

## What was actually run

Five deliberately varied, non-random images, all at their native resolution:

| Scene | Native dimensions | Intended instances | Important configuration changes |
|---|---|---|---|
| Dog | 800×534 | One visible dog, not the person/bowl | Dog phrase; no area ceiling; whole/768px views; no grid |
| Jars | 1015×457 | Physical jars, including lids and visible glass | Jar phrases; much larger area cap; whole/640/384px views; no grid |
| Insect display | 743×1319 | Each insect with wings/body/visible appendages | Insect/butterfly/beetle phrases; 512/320px crops; 32px coarse grid |
| Mushrooms | 1015×1015 | Each visible cap, plus a separate cluster-level proposal | Cap-specific phrases; larger scales; 48px coarse grid; nested labels |
| Cows | 1439×810 | Visibly supported near/far cattle | Cow/cattle phrases; 640/384px crops; no grid; small instances retained |

Complete tested profiles are in
[`configs/non-corn/`](../configs/non-corn/).
They are **image-specific examples, not recommended universal settings**.

All used the same pinned SAM 3 checkpoint, without training or weight changes.
The resident model was reused across cases for convenience, through the shipped
inference/selection/refinement/export functions. One GPU handled dog/jars/mushrooms;
the other handled insects/cows. No input image was enhanced or resized for
inference. Independent source inventories were requested without showing masks.
The automatic draft passes were followed by visual inspection and independent
model audits; their mistakes were intentionally not removed to make the result
look better. These are not completed accepted annotations.

## Observed first-draft results

These numbers are **retained proposals, not verified object counts**:

| Scene | Whole-image text baseline | Multi-scale + guarded refinement | Visually evident issues |
|---|---:|---:|---|
| Dog | 2 | 2 | Good dog silhouette, plus a false patch on a background chair |
| Jars | 14 | 22 | Useful jar bodies, but also stand/support pieces, lids and body fragments |
| Insects | 69 | 117 | Many useful insect masks, plus individual wings, merged specimens, reflections and a wall switch |
| Mushrooms | 28 caps + 1 cluster | 87 caps + 1 cluster | Useful cap surfaces, but many background leaves/hands and duplicate cap/gill/stalk parts |
| Cows | 11 | 24 | Useful nearby/distant cattle, but also deadwood and duplicate body/tail fragments |

All five cases passed the binary/native-coordinate export checks. That does not
make any of the five automatically acceptable. A valid PNG can faithfully encode
a confidently wrong object.

Concrete examples visible in the saved per-instance crops:

- **Dog:** `mask-00001` is background furniture; `mask-00002` is the visible dog.
- **Jars:** `mask-00009` is a stand support. `mask-00021` is a plausible whole
  foreground jar, while other candidates isolate lids or parts of already-covered jars.
- **Insects:** `mask-00001` is a light/reflection, and `mask-00117` is a wall switch.
  Several point masks select individual wings inside a whole-insect proposal.
- **Mushrooms:** `mask-00002` is a background leaf, despite its assigned cap label.
  `mask-00007` is a plausible cap surface; it also has a smaller competing part mask.
- **Cows:** `mask-00005` is deadwood. `mask-00018` is a plausible foreground cow.
  A body, a tail and another fragment may be separate proposals for one animal.

These are qualitative observations, not exhaustive error labels. Fine fur,
antennae, glass boundaries and vegetation occlusions still need careful review.

## Diagnostic threshold probe

The exact same saved candidates were reselected with text-score cutoff **0.5**
instead of **0.2**, with the point gates unchanged. No model was rerun for this
probe, and no result was marked accepted.

| Scene | Whole-image proposals at 0.5 | Multi-scale proposals at 0.5 |
|---|---:|---:|
| Dog | 1 | 1 |
| Jars | 13 | 14 |
| Insects | 55 | 109 |
| Mushrooms | 23 caps + 1 cluster | 77 caps + 1 cluster |
| Cows | 9 | 11 |

The mushroom cluster is a separate category, not another individual cap.
All table entries are diagnostic proposals; no accuracy or recall follows from them.

The threshold removes the obvious dog false positive in this photo, but it does
not solve the overall problem. Point masks still have no independent category
verification. When a text mask is filtered away, an overlapping point candidate
that it previously suppressed can survive instead; point-derived proposals can
therefore become **more numerous** under a stricter text gate. Raising one
confidence threshold is not a semantic repair, and may also remove real targets.

## Why this matters for the skill

1. **A point mask does not prove its assigned category.** A grid finds leaves,
   fingers and lamps as easily as target objects. The current tool flags these
   as unverified point candidates but does not automatically classify them.
2. **Mask IoU is not physical identity.** A complete insect and its single wing,
   or a jar and its lid, can have too little IoU for duplicate suppression even
   though they are not separate instances at the requested level.
3. **Finer views can lose semantic context.** More crops can recover real small
   objects, but also promote parent/part fragments into supposed new objects.
4. **Boundary refinement cannot repair every semantic mistake.** A stable,
   agreeing update may make a wrong leaf mask look cleaner, not turn it into a cap.
5. **Whole-image masks may already be preferable.** More passes should be earned
   by evidence of missed detail, not treated as mandatory quality improvement.
6. **Independent reviewers are fallible.** The mushroom auditor added an extra
   zero to its mask IDs. Those strings were checked against the actual IDs and
   logged as format-only normalizations; no reviewer judgment became ground truth.

The high-level skill calls for semantic/instance review before trust. The
shipped automatic tools are still a **proposal generator**, not a substitute for
that part of the agent's work. A stronger future implementation should require
explicit task scope, gate point candidates by category, track whole/part identity,
and support recorded proposal rejection before boundary refinement. Those
features are not claimed as implemented by these trials.

## Reproduce the approach on another image

Use an explicit complete profile; do not merely rename `kernel` and leave the
inherited corn size/parent settings:

```sh
uv run --locked --project "$S" python "$S/scripts/segment.py" prepare /path/image.jpg \
  --out /path/new-run --config "$S/configs/non-corn/jars.json"
```

First inspect object sizes and change the profile for your native pixels. Then
follow the skill's plan → infer → audit → select → semantic/instance inspection →
refine → re-review → export workflow. Do not use these first-draft proposal counts
as a target. The scalar validation record is in
[the scalar validation record](cross-domain-trials.json).

## Evidence and limits

- One hand-selected image per category; no statistical generalization claim.
- No independent human reference masks, so no precision, recall or boundary-IoU
  accuracy is reported. Model-review inventories are explicitly unverified.
- Native masks, all returned candidate pools, inventories, audits, alternative
  profiles, overlays and per-instance crops were retained in local trial records.
- Input photographs and derived masks are **not** included in the repository.
  Four source photos are labelled CC BY 4.0 in the local collection (credit:
  segbench authors); the dog is an official SAM demo asset whose picture-specific
  redistribution terms were not independently cleared. Private previews are not
  a blanket public release permission.
- The tested implementation was commit `e1fcd71`. Later versions add a same-label
  containment rule to duplicate matching and two review flags (nested part, outside
  every parent). Those were not rerun on these five images, so the counts above
  describe the older selection. This report adds evidence,
  examples and warnings, not a claim that the observed failures have been fixed.
