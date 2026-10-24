#!/usr/bin/env python3
"""Direct SAM 3 calls: find by words or example boxes, click/box one object, or
outline everything. No crops plan, review loop or refinement - just the model.

Run inside the model environment:
  sh "$S/scripts/run" model sam.py find photo.jpg --text "dog" --out out/
Many requests on one photo (model loaded once): the ``batch`` command, or the
Python class ``Sam`` from a long-lived interpreter.

All coordinates are native pixels of the saved ``image.png``: points are
``x,y``; boxes are ``x,y,w,h`` (top-left corner, width, height). Masks are
proposals from a model, never verified labels.
"""

from __future__ import annotations

import sys

# Do not write __pycache__ into the (possibly shared or read-only) skill folder.
sys.dont_write_bytecode = True

import argparse
import colorsys
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from core import (
    bbox,
    canonical_image,
    decode,
    dedupe,
    encode,
    file_sha,
    grid_points,
    lift,
    record,
    stability,
    write_json,
)
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

SCHEMA = "sam-direct/1"


def parse_pair(text):
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected x,y but got {text!r}") from error
    return [x, y]


def parse_box(text):
    try:
        x, y, w, h = (float(v) for v in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected x,y,w,h but got {text!r}") from error
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"Box width and height must be positive: {text!r}")
    return [x, y, w, h]


def resolve_region(region, width, height):
    """Integer crop [x, y, w, h] inside the image; None means the whole image."""
    if region is None:
        return [0, 0, width, height]
    x, y, w, h = (int(round(v)) for v in region)
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
        raise ValueError(f"Region {region} is outside the {width}x{height} image")
    return [x, y, w, h]


def to_local_point(point, region):
    x, y = point[0] - region[0], point[1] - region[1]
    if not (0 <= x < region[2] and 0 <= y < region[3]):
        raise ValueError(f"Point {point} is outside the region {region}")
    return [x, y]


def to_local_box(box, region):
    x, y, w, h = box[0] - region[0], box[1] - region[1], box[2], box[3]
    if x < 0 or y < 0 or x + w > region[2] or y + h > region[3]:
        raise ValueError(f"Box {box} is outside the region {region}")
    return [x, y, w, h]


def normalized_center_box(box, width, height):
    """SAM 3 example boxes are [center x, center y, w, h] divided by image size."""
    x, y, w, h = box
    return [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]


class Sam:
    """Keeps one SAM 3 model loaded. Each method takes a PIL image and returns
    a list of dicts with a full-image boolean ``mask`` and native ``bbox``."""

    def __init__(self, device="cuda:0", checkpoint=None, precision="bf16", batch_size=24):
        from sam3_backend import Sam3Backend

        self.backend = Sam3Backend(
            device=device, checkpoint=checkpoint, precision=precision, batch_size=batch_size
        )
        self.identity = {
            key: self.backend.identity[key]
            for key in ["backend", "source_commit", "checkpoint_sha256", "device", "precision"]
        }

    # Model calls on a crop. Tests replace these three with fakes.
    def _find_local(self, crop, text, like, unlike):
        backend, torch = self.backend, self.backend.torch
        with torch.inference_mode(), backend.context():
            state = backend.image_state(crop)
            backend.processor.reset_all_prompts(state)
            if text:
                state = backend.processor.set_text_prompt(text, state)
            for box, positive in [(b, True) for b in like] + [(b, False) for b in unlike]:
                state = backend.processor.add_geometric_prompt(
                    normalized_center_box(box, crop.width, crop.height), positive, state
                )
            masks = state["masks"].squeeze(1).cpu().numpy()
            scores = state["scores"].float().cpu().numpy()
        return [(mask, float(score), None) for mask, score in zip(masks, scores, strict=True)]

    def _click_local(self, crop, points, labels, box, multimask):
        backend, torch = self.backend, self.backend.torch
        kwargs = dict(multimask_output=multimask, return_logits=True)
        if points:
            kwargs.update(
                point_coords=np.array(points, np.float32),
                point_labels=np.array(labels, np.int32),
            )
        if box is not None:
            x, y, w, h = box
            kwargs["box"] = np.array([x, y, x + w, y + h], np.float32)
        with torch.inference_mode(), backend.context():
            state = backend.image_state(crop)
            logits, scores, _ = backend.model.predict_inst(state, **kwargs)
        return [
            (logit > 0, float(score), stability(logit))
            for logit, score in zip(logits, scores, strict=True)
        ]

    def _points_local(self, crop, points):
        prompts = [dict(points=[p], labels=[1]) for p in points]
        return [
            (decode(c["rle"]), c["score"], c["stability"], c["prompt_index"])
            for c in self.backend.points(crop, prompts)
        ]

    # Public calls -------------------------------------------------------------
    def find(self, image, text=None, like=(), unlike=(), threshold=0.5, region=None):
        """Every instance of a concept: words, example boxes, or both."""
        if not text and not like:
            raise ValueError("find needs --text, --like, or both")
        region = resolve_region(region, image.width, image.height)
        crop = image.crop(_corners(region))
        like = [to_local_box(b, region) for b in like]
        unlike = [to_local_box(b, region) for b in unlike]
        results = []
        for mask, score, stab in self._find_local(crop, text, like, unlike):
            if score >= threshold:
                results.append(_result(mask, score, stab, region, image, "find"))
        return _ranked(results)

    def click(self, image, points=(), negatives=(), box=None, all_hypotheses=False, region=None):
        """One object from positive clicks, negative clicks and/or a box."""
        if not points and box is None:
            raise ValueError("click needs at least one --point or a --box")
        region = resolve_region(region, image.width, image.height)
        crop = image.crop(_corners(region))
        local = [to_local_point(p, region) for p in list(points) + list(negatives)]
        labels = [1] * len(points) + [0] * len(negatives)
        local_box = to_local_box(box, region) if box is not None else None
        # A lone click is ambiguous (kernel or ear?): ask for three alternatives.
        multimask = all_hypotheses or (len(points) == 1 and not negatives and box is None)
        hypotheses = self._click_local(crop, local, labels, local_box, multimask)
        results = _ranked(
            [_result(mask, score, stab, region, image, "click") for mask, score, stab in hypotheses]
        )
        return results if all_hypotheses else results[:1]

    def everything(
        self,
        image,
        step=32,
        min_quality=0.8,
        min_stability=0.9,
        min_area=16,
        overlap_iou=0.7,
        region=None,
    ):
        """Automatic masks from a grid of single clicks, duplicates removed."""
        region = resolve_region(region, image.width, image.height)
        crop = image.crop(_corners(region))
        points = grid_points(crop.width, crop.height, step)
        candidates = []
        for mask, score, stab, index in self._points_local(crop, points):
            px, py = points[index]
            if score < min_quality or stab < min_stability or mask.sum() < min_area:
                continue
            if not mask[min(int(py), mask.shape[0] - 1), min(int(px), mask.shape[1] - 1)]:
                continue
            item = _result(mask, score, stab, region, image, "everything")
            item["candidate_id"] = str(len(candidates))
            candidates.append(item)
        # Prefer confident, stable masks; duplicates only by overlap of the same mask.
        candidates.sort(key=lambda r: -(r["score"] + r["stability"]))
        records = [
            dict(
                record(r["mask"], "object", {}, r["score"], candidate_id=r["candidate_id"]),
                keep=r,
            )
            for r in candidates
        ]
        kept, _ = dedupe(records, overlap_iou)
        return _ranked([r["keep"] for r in kept])


def _corners(region):
    x, y, w, h = region
    return (x, y, x + w, y + h)


def _result(local_mask, score, stab, region, image, kind):
    full, cut = lift(np.asarray(local_mask, bool), region, (image.height, image.width))
    return dict(
        kind=kind,
        mask=full,
        score=float(score),
        stability=None if stab is None else float(stab),
        area=int(full.sum()),
        bbox=bbox(full),
        touches_region_edge=cut,
    )


def _ranked(results):
    results = [r for r in results if r["area"]]
    results.sort(key=lambda r: -r["score"])
    for i, result in enumerate(results, 1):
        result["id"] = f"mask-{i:03d}"
    return results


# Files and command line -------------------------------------------------------
def colors(count):
    return [
        tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618) % 1, 0.85, 1.0))
        for i in range(count)
    ]


def overlay(image, results):
    pixels = np.array(image.convert("RGB")).astype(float)
    for result, color in zip(results, colors(len(results)), strict=True):
        mask = result["mask"]
        pixels[mask] = 0.55 * pixels[mask] + 0.45 * np.array(color)
        pixels[mask & ~ndi.binary_erosion(mask)] = color
    picture = Image.fromarray(pixels.astype(np.uint8))
    draw = ImageDraw.Draw(picture)
    for result in results:
        if result["area"]:
            y, x = ndi.center_of_mass(result["mask"])
            draw.text(
                (x - 6, y - 6),
                result["id"][-3:].lstrip("0") or "0",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
    return picture


def save(out, image, source, request, results, identity, png=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    image_path = out / "image.png"
    if not image_path.exists():
        image.save(image_path)
    name = request.get("name")
    if not name:
        # Never overwrite an earlier result in the same folder.
        name, number = request["command"], 1
        while (out / f"{name}.json").exists():
            number += 1
            name = f"{request['command']}-{number}"
    overlay(image, results).save(out / f"{name}-overlay.png")
    if png:
        (out / f"{name}-masks").mkdir(exist_ok=True)
        for result in results:
            Image.fromarray(result["mask"].astype(np.uint8) * 255).save(
                out / f"{name}-masks" / f"{result['id']}.png"
            )
    payload = dict(
        schema=SCHEMA,
        created_at=datetime.now(timezone.utc).isoformat(),
        source=str(source),
        image="image.png",
        image_sha256=file_sha(image_path),
        width=image.width,
        height=image.height,
        request=request,
        model=identity,
        note="Model proposals in native pixels of image.png. Not verified labels.",
        masks=[
            {key: value for key, value in result.items() if key != "mask"}
            | dict(rle=encode(result["mask"]))
            for result in results
        ],
    )
    write_json(out / f"{name}.json", payload)
    return dict(
        request=name,
        masks=len(results),
        overlay=str(out / f"{name}-overlay.png"),
        result=str(out / f"{name}.json"),
        top=[
            dict(id=r["id"], score=round(r["score"], 3), bbox=r["bbox"], area=r["area"])
            for r in results[:25]
        ],
    )


def load_masks(path):
    """Read a result file back as full-image boolean masks (for later code)."""
    payload = json.loads(Path(path).read_text())
    return [dict(item, mask=decode(item["rle"])) for item in payload["masks"]]


FIELDS = {
    "find": {"text", "like", "unlike", "threshold"},
    "click": {"points", "negatives", "box", "all_hypotheses"},
    "everything": {"step", "min_quality", "min_stability", "min_area", "overlap_iou"},
}


def _number(value):
    return type(value) in (int, float) and np.isfinite(value)


def check_request(request):
    """Batch requests skip the command-line parser, so check their shape here."""
    if not isinstance(request, dict) or request.get("command") not in FIELDS:
        raise ValueError("Each request needs a command: find, click or everything")
    command = request["command"]
    unknown = set(request) - FIELDS[command] - {"command", "name", "region"}
    if unknown:
        raise ValueError(f"Unknown fields for {command}: {sorted(unknown)}")

    def box(value, field):
        if not (
            isinstance(value, list)
            and len(value) == 4
            and all(_number(v) for v in value)
            and value[2] > 0
            and value[3] > 0
        ):
            raise ValueError(f"{field} must be a box [x, y, w, h] with w, h > 0; got {value!r}")

    def boxes(field):
        value = request.get(field, [])
        if not isinstance(value, list) or (value and not isinstance(value[0], list)):
            raise ValueError(f"{field} must be a list of boxes, e.g. [[x, y, w, h]]")
        for item in value:
            box(item, field)

    def points(field):
        value = request.get(field, [])
        if not isinstance(value, list) or not all(
            isinstance(p, list) and len(p) == 2 and all(_number(v) for v in p) for p in value
        ):
            raise ValueError(f"{field} must be a list of points, e.g. [[x, y]]")

    if request.get("region") is not None:
        box(request["region"], "region")
    if command == "find":
        boxes("like")
        boxes("unlike")
        if request.get("text") is not None and not isinstance(request["text"], str):
            raise ValueError("text must be a string")
    if command == "click":
        points("points")
        points("negatives")
        if request.get("box") is not None:
            box(request["box"], "box")
    for field in ["threshold", "min_quality", "min_stability", "overlap_iou"]:
        if field in request and not (_number(request[field]) and 0 <= request[field] <= 1):
            raise ValueError(f"{field} must be a number between 0 and 1")
    for field in ["step", "min_area"]:
        if field in request and not (type(request[field]) is int and request[field] >= 1):
            raise ValueError(f"{field} must be a positive whole number")
    return request


def run_request(sam, image, request):
    check_request(request)
    command = request["command"]
    region = request.get("region")
    if command == "find":
        return sam.find(
            image,
            text=request.get("text"),
            like=request.get("like", []),
            unlike=request.get("unlike", []),
            threshold=request.get("threshold", 0.5),
            region=region,
        )
    if command == "click":
        return sam.click(
            image,
            points=request.get("points", []),
            negatives=request.get("negatives", []),
            box=request.get("box"),
            all_hypotheses=request.get("all_hypotheses", False),
            region=region,
        )
    if command == "everything":
        options = {
            key: request[key]
            for key in ["step", "min_quality", "min_stability", "min_area", "overlap_iou"]
            if key in request
        }
        return sam.everything(image, region=region, **options)
    raise ValueError(f"Unknown command {command!r}; use find, click or everything")


def request_from_args(args):
    if args.command == "find":
        return dict(
            command="find",
            text=args.text,
            like=args.like,
            unlike=args.unlike,
            threshold=args.threshold,
            region=args.region,
        )
    if args.command == "click":
        return dict(
            command="click",
            points=args.point,
            negatives=args.negative,
            box=args.box,
            all_hypotheses=args.all_hypotheses,
            region=args.region,
        )
    return dict(
        command="everything",
        step=args.step,
        min_quality=args.min_quality,
        min_stability=args.min_stability,
        min_area=args.min_area,
        overlap_iou=args.overlap_iou,
        region=args.region,
    )


def read_requests(path):
    text = Path(path).read_text() if str(path) != "-" else sys.stdin.read()
    stripped = text.strip()
    if stripped.startswith("["):
        return json.loads(stripped)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("image", type=Path)
        p.add_argument(
            "--out", required=True, type=Path, help="Folder for image.png, JSON, overlay"
        )
        p.add_argument(
            "--orientation",
            choices=["exif", "raw"],
            default="exif",
            help="exif (default) turns phone photos upright, as image viewers show them",
        )
        p.add_argument("--png", action="store_true", help="Also write one PNG per mask")
        p.add_argument(
            "--device", default="cuda:0", help="NVIDIA GPU, e.g. cuda:0 (no CPU support)"
        )
        p.add_argument("--precision", choices=["bf16"], default="bf16")
        p.add_argument("--batch-size", type=int, default=24)
        p.add_argument("--checkpoint", type=Path, help="Existing sam3.pt instead of downloading")

    def region(p):
        p.add_argument("--region", type=parse_box, help="x,y,w,h: run on this part only (zooms in)")

    p = sub.add_parser("find", help="Every instance matching words and/or example boxes")
    common(p)
    region(p)
    p.add_argument("--text", help='Short noun phrase, e.g. "dog" or "corn kernel"')
    p.add_argument("--like", type=parse_box, action="append", default=[], help="Example box")
    p.add_argument("--unlike", type=parse_box, action="append", default=[], help="Counter-example")
    p.add_argument("--threshold", type=float, default=0.5, help="Minimum score (default 0.5)")

    p = sub.add_parser("click", help="One object from clicks and/or a box")
    common(p)
    region(p)
    p.add_argument("--point", type=parse_pair, action="append", default=[], help="x,y inside")
    p.add_argument("--negative", type=parse_pair, action="append", default=[], help="x,y outside")
    p.add_argument("--box", type=parse_box, help="x,y,w,h around the object")
    p.add_argument("--all-hypotheses", action="store_true", help="Return all three alternatives")

    p = sub.add_parser("everything", help="Automatic masks for everything visible")
    common(p)
    region(p)
    p.add_argument("--step", type=int, default=32, help="Grid spacing in pixels (smaller = more)")
    p.add_argument("--min-quality", type=float, default=0.8)
    p.add_argument("--min-stability", type=float, default=0.9)
    p.add_argument("--min-area", type=int, default=16)
    p.add_argument("--overlap-iou", type=float, default=0.7, help="Duplicate threshold")

    p = sub.add_parser("batch", help="Many requests on one image, model loaded once")
    common(p)
    p.add_argument("requests", help="JSON list or JSON-lines file, or - for stdin")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    image = canonical_image(args.image, args.orientation)
    if (args.out / "image.png").exists():
        previous = np.array(Image.open(args.out / "image.png").convert("RGB"))
        if not np.array_equal(previous, np.array(image)):
            raise ValueError("--out already holds results for a different image; use a new folder")
    requests = (
        read_requests(args.requests) if args.command == "batch" else [request_from_args(args)]
    )
    for request in requests:
        check_request(request)
    names = [r["name"] for r in requests if r.get("name")]
    if len(set(names)) != len(names) or any("/" in n or n.startswith(".") for n in names):
        raise ValueError("Request names must be unique simple file names")
    sam = Sam(
        device=args.device,
        checkpoint=args.checkpoint,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    summaries = []
    for request in requests:
        results = run_request(sam, image, request)
        summaries.append(
            save(args.out, image, args.image, request, results, sam.identity, args.png)
        )
    print(json.dumps(dict(width=image.width, height=image.height, results=summaries), indent=1))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Error: {error}") from error
