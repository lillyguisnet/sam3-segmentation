# Direct SAM 3 calls

`scripts/sam.py` gives an agent the model itself, without the crop plan, review
loop or refinement of the full workflow. Use it when a quick outline is enough,
to explore a photo before writing a full config, or to build your own procedure.

It needs the model environment from [setup](setup.md) (Linux + NVIDIA, accepted
SAM 3 terms). Below, `SAM` stands for:

```sh
uv run --locked --project "$S/environments/sam3" python "$S/scripts/sam.py"
```

Loading the model takes about 20-30 seconds per command. For many requests, use
`batch` or the Python class (bottom of this page).

## Coordinates

- Every command saves the exact pixels it used as `OUT/image.png`. All points,
  boxes and masks refer to that image's native pixels, origin top-left.
- Points are `x,y`. Boxes are `x,y,w,h` (top-left corner, width, height).
- `--orientation exif` (default) turns phone photos upright, as image viewers
  show them. The full workflow defaults to `raw`; pick one per photo and keep it.
- The photo is never resized, sharpened or enhanced by this tool.

## Commands

**find** - every instance of a concept, from words, example boxes, or both.

```sh
$SAM find photo.jpg --text "dog" --out out/
$SAM find photo.jpg --text "butterfly" --threshold 0.3 --out out/   # more, less sure
$SAM find photo.jpg --like 120,80,40,35 --out out/                  # "more like this box"
$SAM find photo.jpg --text "jar" --unlike 400,10,60,40 --out out/   # "not like this"
```

Use short noun phrases ("corn kernel", "jar lid"), not instructions. Default
score threshold is 0.5, the model's own default.

**click** - one object from clicks and/or a box.

```sh
$SAM click photo.jpg --point 300,220 --out out/
$SAM click photo.jpg --point 300,220 --negative 330,240 --out out/
$SAM click photo.jpg --box 250,180,120,90 --out out/
$SAM click photo.jpg --point 300,220 --all-hypotheses --out out/
```

A single click is ambiguous: a point on a kernel is also on the ear. The tool
then asks for three alternatives and returns the highest-scored one; use
`--all-hypotheses` to see all three (often small part, middle, whole object).
Negative clicks on neighbours or a tight box pin down the intended object.

**everything** - automatic masks from a grid of single clicks.

```sh
$SAM everything photo.jpg --step 32 --out out/
```

`--step` is the grid spacing in pixels; objects smaller than it can be missed.
Masks below `--min-quality` (0.8) or `--min-stability` (0.9) are dropped, and
masks overlapping a kept mask by IoU of `--overlap-iou` (0.7) or more count as
duplicates. Parts and wholes both appear. No names are attached.

**--region x,y,w,h** (all three commands) runs on that part only. The model then
sees it larger, which helps small objects. Masks come back in full-image pixels;
`touches_region_edge: true` means the object may continue outside the region.

**--png** also writes one black-and-white PNG per mask.

## What worked in tests (23 October 2026)

One RTX 3090; one photo each; not an accuracy benchmark.

| Photo | Request | Result |
|---|---|---|
| Dog | `find --text dog` | The dog only. The 0.5 default left out the chair patch that the full workflow's 0.2 cutoff kept |
| Dog | one `click`, or a `box` | The dog (score 0.98-0.99) |
| Dog | `everything --step 32` | 17 masks: dog, bowl, food, legs, feet, floor, wall, chair parts |
| Butterflies | `find --text butterfly` | 36 masks |
| Butterflies | `find --like` (box of one butterfly) | 56 masks, including beetles; the lamp and light switch were not picked |
| Corn kernels (~20 px) | `find --text "corn kernel"`, whole photo | **Nothing** at 0.5; at 0.2 only two large wrong masks |
| Corn kernels | same, `--region` 200 px | Nothing at 0.5; 145 masks at 0.2 |
| Corn kernels | same, `--region` 120 px | 80 masks at 0.5 |
| Corn kernels | `--like` one kernel, `--region` 120 px | 69 masks, highest scores |

Lessons:

- **Small objects need zooming.** The model resizes the photo (or region) to a
  fixed internal size. When objects are tiny in that view, words find nothing.
  Use `--region` so a typical object is roughly a sixth of the region width or
  more, and cover the photo with overlapping regions. At that point the
  [full workflow](../SKILL.md) is usually the better tool.
- **An example box often beats words** for repeated objects. Draw it tightly
  around one clear instance.
- **A lower threshold finds more, including more wrong things.** Look at the
  overlay after changing it.
- A single click's best mask may be a part or a whole. Use `--all-hypotheses`
  when it matters.

## Outputs

For a request named `find` (later ones become `find-2`, `find-3`, ...):

| File | Contents |
|---|---|
| `image.png` | The pixels used. A different photo in the same `--out` is refused |
| `find-overlay.png` | Coloured, numbered masks over the photo: look at this |
| `find.json` | Request, model identity, and per mask: `id`, `score`, `stability` (click/everything), `bbox`, `area`, `touches_region_edge`, COCO `rle` |
| `find-masks/` | With `--png` only |

The command also prints a short summary with the top 25 masks. Read masks back
in Python with `sam.load_masks("out/find.json")`.

## Batch: one model load, many requests

```sh
cat > requests.jsonl <<'EOF'
{"command": "find", "text": "mushroom cap"}
{"command": "click", "points": [[300, 220]], "negatives": [[330, 240]], "name": "cap-3"}
{"command": "everything", "step": 24, "region": [0, 0, 500, 400]}
EOF
$SAM batch photo.jpg requests.jsonl --out out/
```

Fields match the command-line options (`text`, `like`, `unlike`, `threshold`,
`points`, `negatives`, `box`, `all_hypotheses`, `step`, `min_quality`,
`min_stability`, `min_area`, `overlap_iou`, `region`, optional `name`). A JSON
list works too, and `-` reads from standard input.

## Python: keep the model loaded

From an interpreter started in the model environment:

```python
import sys; sys.path.insert(0, f"{S}/scripts")
from sam import Sam
from core import canonical_image
sam = Sam(device="cuda:0")              # checkpoint="/path/sam3.pt" to skip download
image = canonical_image("photo.jpg", "exif")
dogs = sam.find(image, text="dog")
one = sam.click(image, points=[[300, 220]], negatives=[[330, 240]])
parts = sam.click(image, points=[[300, 220]], all_hypotheses=True)
auto = sam.everything(image, step=24, region=[0, 0, 500, 400])
# each result: dict(id, score, stability, bbox, area, mask=<full-size bool array>, ...)
```

## What this mode does not do

- No checking. Scores are the model's own confidence, not accuracy. A
  confident mask can be the wrong object (a chair called a dog).
- No multi-scale crops, raw-candidate archive, review sheets, refinement or
  viewer. For dense small objects or work others must audit, use the full
  workflow in `SKILL.md`.
- The same photo may give slightly different masks on different hardware.

Always look at the overlay before reporting results, and call them model
proposals, not verified outlines.
